import os
import typing as tp

import numpy as np
import jax, jax.numpy as jnp
import kp

from .kompute_jaxpr_interpreter import JaxprInterpreter
from .buffers                   import Buffer, BufferPool

class Function:
    def __init__(
        self, 
        function:       tp.Callable, 
        static_argnums: tp.Tuple[int] = (),
        profiling:      bool          = False, 
        reuse_buffers:  bool          = True,
        *args, 
        **kwargs
    ):
        self.jaxpr_function      = jax.make_jaxpr(function, static_argnums, return_shape=True)
        device                   = int(os.environ.get('VKJAX_DEVICE', 0))
        self.mgr                 = kp.Manager(device)
        self.workgroup_size      = _get_maximum_workgroup_size(self.mgr)
        self.bufferpool          = BufferPool(self.mgr, self.workgroup_size, reuse_buffers)
        self._jaxpr_interpreters = dict()
        self._output_shapes      = dict()
        self._profiling          = profiling
        kwargs['profiling']      = profiling
        kwargs['static_argnums'] = static_argnums
        kwargs['workgroup_size'] = self.workgroup_size
        self._interpreter_args   = (args, kwargs)
    
    def __call__(self, *args:tp.Any, **kwargs:tp.Any) -> tp.Any:
        jaxpr_interpreter, output_shapes = self._get_or_create_jaxpr_interpreter(args)
        output = jaxpr_interpreter.run(*args, **kwargs)
        output = _restore_shapes(output, output_shapes)
        if not self._profiling:
            return output
        else:
            return output, jaxpr_interpreter.get_profiling_info()
    
    def _get_or_create_jaxpr_interpreter(self, args:tp.Tuple[tp.Any]) -> JaxprInterpreter:
        leaves,structure = jax.tree_flatten(args)
        args_shape       = tuple(jax.tree_map(np.shape, leaves))
        args_dtype       = tuple(jax.tree_map(lambda x: np.asarray(x).dtype, leaves))
        preloaded_args   = tuple((i,a) for i,a in enumerate(leaves) if isinstance(a, Buffer) )
        call_signature   = (args_shape, args_dtype, structure, preloaded_args)
        if call_signature not in self._jaxpr_interpreters:
            #new input shapes or structure, need to re-trace
            jaxpr, output_shapes = self.jaxpr_function(*args)
            args,kwargs          = self._interpreter_args
            interpreter          = JaxprInterpreter(jaxpr, self.mgr, self.bufferpool, dict(preloaded_args), *args, **kwargs)
            self._jaxpr_interpreters[call_signature] = interpreter
            self._output_shapes[call_signature]      = output_shapes
        return self._jaxpr_interpreters[call_signature], self._output_shapes[call_signature]
    
    def to_buffers(self, *args):
        if len(args)==1:
            args = args[0]
        tensors = jax.tree_map(lambda x: self.bufferpool._create_tensor(var=None, initial_value=x), args)
        buffers = jax.tree_multimap(lambda x,t: Buffer(t, jnp.asarray(x).dtype, np.shape(x)), args, tensors)
        return buffers

def _restore_shapes(x, targetshapes):
    structure = jax.tree_structure(targetshapes)
    x         = jax.tree_unflatten(structure, x)
    x         = jax.tree_multimap(lambda a,shapestruct: np.asarray(a).reshape(shapestruct.shape), x, targetshapes )
    return x

def _get_maximum_workgroup_size(mgr:kp.Manager):
    devprops = mgr.get_device_properties()
    return min(devprops['max_work_group_invocations'], devprops['max_work_group_size'][0])



def wrap(function:tp.Callable, *args, **kwargs):
    return Function(function, *args, **kwargs)
