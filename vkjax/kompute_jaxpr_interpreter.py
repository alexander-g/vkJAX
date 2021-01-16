import typing as tp
import time

from . import ops

import kp
import numpy as np
import jax



class JaxprInterpreter:
    def __init__(self, jaxpr:jax.core.ClosedJaxpr, static_argnums=()):
        self.jaxpr          = jaxpr
        self.static_argnums = static_argnums
        #dict mapping from jax.core.Var or int (for jax.core.Literal) to Buffer
        self.buffers   = {}
        self.mgr       = kp.Manager()

        self.analyze_closed_jaxpr(jaxpr)
    
    def analyze_closed_jaxpr(self, jaxpr:jax.core.ClosedJaxpr):
        '''Starts the analysis of the top level jaxpr.
           Records operations into a kp.Sequence and creates required buffers.'''
        self.sequence    = self.mgr.create_sequence("main")
        self.sequence.begin()
        
        assert len(jaxpr.consts) == len(jaxpr.jaxpr.constvars)
        for constvar, constval in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
            self.get_or_create_buffer(constvar, initial_value=constval)

        self.all_ops = self.analyze_jaxpr(jaxpr.jaxpr)
        for op in self.all_ops:
            self.sequence.record_algo_data(op.tensors, op.shader)
    
        self.output_tensors = [self.get_or_create_buffer(var).tensor for var in jaxpr.jaxpr.outvars]
        self.sequence.record_tensor_sync_local(self.output_tensors)
        self.sequence.end()
    
    def analyze_jaxpr(self, jaxpr:jax.core.Jaxpr):
        '''Analyzes (possibly inner) jaxprs'''
        all_ops = []
        for eq in jaxpr.eqns:
            if all([str(v)=='_' for v in eq.outvars]):
                #seems like a redundant operation
                continue
            opname = eq.primitive.name.replace('-','_')
            method = getattr(ops, opname, None)
            if method is None:
                raise NotImplementedError(eq)
            method_ops = method(self, eq)
            all_ops   += method_ops
        return all_ops


    def run(self, *X, return_all=False, profile=False):
        '''Executes a previously recorded sequence with actual data'''
        input_tensors = [self.get_or_create_buffer(var).tensor for var in self.jaxpr.jaxpr.invars]
        X             = jax.tree_leaves([x for i,x in enumerate(X) if i not in self.static_argnums])
        assert len(input_tensors) == len(X)
        for input_tensor,x in zip(input_tensors,X):
            input_tensor.set_data(np.ravel(x))
        
        if len(input_tensors)>0:
            #transfer input data to device
            self.mgr.eval_tensor_sync_device_def(input_tensors)

        if not profile:
            self.sequence.eval()
        else:
            timings = self.profiling_eval()
            return timings
        
        output_bufs    = [self.get_or_create_buffer(var) for var in self.jaxpr.jaxpr.outvars]
        output_values  = [buf.numpy() for buf in output_bufs]
        output_values  = [np.asarray(x, dtype=var.aval.dtype) for x,var in zip(output_values, self.jaxpr.jaxpr.outvars)]
        output_values  = tuple(output_values)

        if len(output_values)==1:
            output_values = output_values[0]
        
        if not return_all:
            return output_values
        else:
            all_tensors = [b.tensor for b in self.buffers.values() if b is not None]
            self.mgr.eval_tensor_sync_local_def(all_tensors)
            all_arrays = [(var, buf.numpy() if buf is not None else None) for var,buf in self.buffers.items()]
            return output_values, dict(all_arrays)

    def get_or_create_buffer(self, var:tp.Union[jax.core.Var, jax.core.Literal], initial_value:tp.Optional[np.ndarray] = None):
        if isinstance(var, jax.core.Literal):
            #literals are for some reason not hashable but have a hash
            varhash = var.hash
        else:
            varhash = var
        
        if varhash not in self.buffers:
            #kompute always uses f32
            dtype = np.float32

            #create new tensor
            if hasattr(var, 'val'):
                #literals
                initial_value = var.val
            elif initial_value is None:
                initial_value = np.zeros(var.aval.shape, dtype)
            else:
                assert initial_value.shape == var.aval.shape
                if initial_value.shape == (0,):
                    #zero sized array, would cause kompute to segfault
                    #expand it, but dont update the buffer shape
                    initial_value = np.empty((1,), dtype)
                if initial_value.dtype != dtype:
                    initial_value = initial_value.astype(dtype)
            tensor = kp.Tensor( np.ravel(initial_value) )
            self.mgr.eval_tensor_create_def([tensor])
            self.mgr.eval_tensor_sync_device_def([tensor])
            self.buffers[varhash] = ops.Buffer(tensor, dtype, var.aval.shape)
        return self.buffers[varhash]
    

    def profiling_eval(self):
        timings = []
        for op in self.all_ops:
            t0 = time.time()
            self.mgr.eval_algo_data_def(op.tensors, op.shader)
            t1 = time.time()
            timings.append(t1-t0)
        self.mgr.eval_tensor_sync_local_def(self.output_tensors)
        return timings

