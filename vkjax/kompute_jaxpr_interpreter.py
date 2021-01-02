import typing as tp
from collections import namedtuple

from . import shaders

import kp
import numpy as np
import jax


Variable = namedtuple('Variable', ['tensor', 'dtype', 'shape'])


class JaxprInterpreter:
    def __init__(self, jaxpr):
        self.jaxpr     = jaxpr
        self.variables = dict()
        self.mgr       = kp.Manager()

        self.analyze_jaxpr(jaxpr)
    
    def analyze_jaxpr(self, jaxpr):
        self.sequence    = self.mgr.create_sequence("")
        self.sequence.begin()

        for invar in jaxpr.jaxpr.invars:
            dtype = invar.aval.dtype
            shape = invar.aval.shape
            var   = self.create_input(dtype, shape)
            self.variables[invar] = var
        
        for eq in jaxpr.jaxpr.eqns:
            method = getattr(self, eq.primitive.name, None)
            if method is None:
                raise NotImplementedError(eq.primitive.name)
            method(eq)
    
        output_tensors = [self.variables[var].tensor for var in jaxpr.jaxpr.outvars]
        self.sequence.record_tensor_sync_local(output_tensors)
        self.sequence.end()

    def run(self, *X):
        input_tensors = [self.variables[var].tensor for var in self.jaxpr.jaxpr.invars]
        assert len(input_tensors) == len(X)
        for input_tensor,x in zip(input_tensors,X):
            input_tensor.set_data(np.ravel(x))

        self.sequence.eval()
        
        output_vars    = [self.variables[var] for var in self.jaxpr.jaxpr.outvars]
        output_values  = [var.tensor.numpy().reshape(var.shape) for var in output_vars]

        if len(output_values)==1:
            output_values = output_values[0]
        return output_values


    
    def create_input(self, dtype, shape):
        tensor = kp.Tensor(np.ones(shape, dtype).ravel())
        self.mgr.eval_tensor_create_def([tensor])
        self.sequence.record_tensor_sync_device([tensor])
        return Variable(tensor, dtype, shape)
    
    def add(self, equation):
        assert equation.params=={}
        assert len(equation.invars)==2
        assert len(equation.outvars)==1
        
        outvar      = equation.outvars[0]

        intensors   = []
        for invar in equation.invars:
            if isinstance(invar, jax.core.Literal):
                #literals are for some reason not hashable
                tensor = kp.Tensor( np.ravel(invar.val) )
                print('FIXME: literals cleanup')
                self.mgr.eval_tensor_create_def([tensor])
                self.sequence.record_tensor_sync_device([tensor])
            elif invar in self.variables:
                tensor = self.variables[invar].tensor
            else:
                raise NotImplementedError(invar)
            assert invar.aval.dtype == outvar.aval.dtype
            if invar.aval.shape != outvar.aval.shape:
                tensor = self.broadcast(tensor, invar, outvar.aval.shape)
            intensors.append(tensor)
        
        shape,dtype = outvar.aval.shape, outvar.aval.dtype
        outtensor   = kp.Tensor( np.ones(shape, dtype).ravel() )
        self.mgr.eval_tensor_create_def([outtensor])
        shader_bytes = shaders.get_shader('add')
        self.sequence.record_algo_data(intensors+[outtensor], shader_bytes)
        self.variables[outvar] = Variable(outtensor, dtype, shape)

    def broadcast(self, tensor, val, newshape):
        assert val.aval.shape == ()
        #FIXME: this should not be a shader call
        result       = kp.Tensor( np.ones( newshape, val.aval.dtype ).ravel() )
        self.mgr.eval_tensor_create_def([result])
        shader_bytes = shaders.get_shader('broadcast')
        self.sequence.record_algo_data([result, tensor], shader_bytes)
        return result
    
    def xla_call(self, equation):
        assert equation.params['device'] == None
        assert equation.params['backend'] == None

        jaxpr = equation.params['call_jaxpr']
        assert len(equation.invars) == len(jaxpr.invars)
        assert len(equation.outvars) == len(jaxpr.outvars)
        for eq_var, jaxpr_var in zip(equation.invars, jaxpr.invars):
            self.variables[jaxpr_var] = self.variables[eq_var]
        
        #TODO: code duplication
        for eq in jaxpr.eqns:
            method = getattr(self, eq.primitive.name, None)
            if method is None:
                raise NotImplementedError(eq.primitive.name)
            method(eq)
        
        for eq_var, jaxpr_var in zip(equation.outvars, jaxpr.outvars):
            self.variables[eq_var] = self.variables[jaxpr_var]