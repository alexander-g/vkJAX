import typing as tp

from . import shaders

import kp
import numpy as np
import jax


class Buffer(tp.NamedTuple):
    tensor : kp.Tensor
    dtype  : np.dtype
    shape  : tp.Tuple


class JaxprInterpreter:
    def __init__(self, jaxpr:jax.core.ClosedJaxpr):
        self.jaxpr     = jaxpr
        #dict mapping from jax.core.Var or int (for jax.core.Literal) to Buffer
        self.buffers   = {}
        self.mgr       = kp.Manager()

        self.analyze_closed_jaxpr(jaxpr)
    
    def analyze_closed_jaxpr(self, jaxpr:jax.core.ClosedJaxpr):
        '''Starts the analysis of the top level jaxpr.
           Records operations into a kp.Sequence and creates required buffers.'''
        self.sequence    = self.mgr.create_sequence("")
        self.sequence.begin()

        for invar in jaxpr.jaxpr.invars:
            self.get_or_create_buffer(invar)
        
        self.analyze_jaxpr(jaxpr.jaxpr)
    
        output_tensors = [self.buffers[var].tensor for var in jaxpr.jaxpr.outvars]
        self.sequence.record_tensor_sync_local(output_tensors)
        self.sequence.end()
    
    def analyze_jaxpr(self, jaxpr:jax.core.Jaxpr):
        '''Analyzes (possibly inner) jaxprs'''
        for eq in jaxpr.eqns:
            method = getattr(self, eq.primitive.name, None)
            if method is None:
                raise NotImplementedError(eq.primitive.name)
            method(eq)

    def run(self, *X):
        '''Executes a previously recoreded sequence with actual data'''
        input_tensors = [self.buffers[var].tensor for var in self.jaxpr.jaxpr.invars]
        assert len(input_tensors) == len(X)
        for input_tensor,x in zip(input_tensors,X):
            input_tensor.set_data(np.ravel(x))

        self.sequence.eval()
        
        output_vars    = [self.buffers[var] for var in self.jaxpr.jaxpr.outvars]
        output_values  = [var.tensor.numpy().reshape(var.shape) for var in output_vars]

        if len(output_values)==1:
            output_values = output_values[0]
        return output_values


    
    def add(self, equation:jax.core.JaxprEqn):
        assert equation.params=={}
        assert len(equation.invars)==2
        assert len(equation.outvars)==1
        
        outvar      = equation.outvars[0]

        intensors   = []
        for invar in equation.invars:
            tensor = self.get_or_create_buffer(invar).tensor
            assert invar.aval.dtype == outvar.aval.dtype
            if invar.aval.shape != outvar.aval.shape:
                tensor = self.broadcast(tensor, invar, outvar.aval.shape)
            intensors.append(tensor)
        
        shape,dtype = outvar.aval.shape, outvar.aval.dtype
        outtensor   = kp.Tensor( np.ones(shape, dtype).ravel() )
        self.mgr.eval_tensor_create_def([outtensor])
        shader_bytes = shaders.get_shader('add')
        self.sequence.record_algo_data(intensors+[outtensor], shader_bytes)
        self.buffers[outvar] = Buffer(outtensor, dtype, shape)

    def broadcast(self, tensor:kp.Tensor, var:jax.core.Var, newshape:tp.Tuple):
        assert var.aval.shape == (), 'broadcasting of non-scalars not implemented'
        #FIXME: this should not be a shader call
        result       = kp.Tensor( np.ones( newshape, var.aval.dtype ).ravel() )
        self.mgr.eval_tensor_create_def([result])
        shader_bytes = shaders.get_shader('broadcast')
        self.sequence.record_algo_data([result, tensor], shader_bytes)
        return result
    
    def xla_call(self, equation:jax.core.JaxprEqn):
        assert equation.params['device'] == None
        assert equation.params['backend'] == None

        jaxpr = equation.params['call_jaxpr']
        assert len(equation.invars) == len(jaxpr.invars)
        assert len(equation.outvars) == len(jaxpr.outvars)
        for eq_var, jaxpr_var in zip(equation.invars, jaxpr.invars):
            self.buffers[jaxpr_var] = self.get_or_create_buffer(eq_var)
        
        self.analyze_jaxpr(jaxpr)
        
        for eq_var, jaxpr_var in zip(equation.outvars, jaxpr.outvars):
            self.buffers[eq_var] = self.buffers[jaxpr_var]
    

    def get_or_create_buffer(self, var:tp.Union[jax.core.Var, jax.core.Literal]):
        if isinstance(var, jax.core.Literal):
            #literals are for some reason not hashable but have a hash
            varhash = var.hash
        else:
            varhash = var
        
        if varhash not in self.buffers:
            #create new tensor
            if hasattr(var, 'val'):
                #literals
                initial_value = var.val
            else:
                initial_value = np.empty(var.aval.shape, var.aval.dtype)
            tensor = kp.Tensor( np.ravel(initial_value) )
            self.mgr.eval_tensor_create_def([tensor])
            self.sequence.record_tensor_sync_device([tensor])
            self.buffers[varhash] = Buffer(tensor, var.aval.dtype, var.aval.shape)
        return self.buffers[varhash]
