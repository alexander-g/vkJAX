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
            workgroup = (len(op.tensors[0])//32,1,1)
            self.sequence.record_algo_data(op.tensors, op.shader, workgroup)
    
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
            input_tensor.set_data(maybe_pad(x))
        
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

        
        if not return_all:
            return output_values
        else:
            all_tensors = [b.tensor for b in self.buffers.values() if b is not None]
            self.mgr.eval_tensor_sync_local_def(all_tensors)
            all_arrays = [(var, buf.numpy() if buf is not None else None) for var,buf in self.buffers.items()]
            print( [[(var, buf.numpy() if buf is not None else '"None"') for var,buf in self.buffers.items() if str(var)=='c']] )
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
                if initial_value.dtype != dtype:
                    initial_value = initial_value.astype(dtype)
            
            #pad to (currently) 32 if needed
            initial_value = maybe_pad(initial_value)

            tensor = kp.Tensor( initial_value )
            self.mgr.eval_tensor_create_def([tensor])
            self.mgr.eval_tensor_sync_device_def([tensor])
            self.buffers[varhash] = ops.Buffer(tensor, dtype, var.aval.shape)
        return self.buffers[varhash]
    

    def profiling_eval(self):
        timings = []
        for op in self.all_ops:
            seq = self.mgr.create_sequence()
            seq.begin()
            workgroup = (len(op.tensors[0])//32,1,1)
            seq.record_algo_data(op.tensors, op.shader, workgroup)
            seq.end()

            t0 = time.time()
            seq.eval()
            t1 = time.time()
            timings.append(t1-t0)
        self.mgr.eval_tensor_sync_local_def(self.output_tensors)
        return timings



def maybe_pad(x, pad_to=32):
    x          = np.ravel(x)
    remainder  = x.size % pad_to
    if remainder != 0 or x.size==0:
        x = np.pad(x, (0,pad_to-remainder))
    return x
