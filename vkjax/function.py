from .kompute_jaxpr_interpreter import JaxprInterpreter
import numpy as np
import jax, jax.numpy as jnp

import typing as tp

class Function:
    def __init__(self, function:tp.Callable, static_argnums:tp.Tuple[int]=(), profiling:bool=False, *args, **kwargs):
        self.jaxpr_function = jax.make_jaxpr(function, static_argnums, return_shape=True)
        self._jaxpr_interpreters = dict()
        self._output_shapes      = dict()
        self._profiling          = profiling
        kwargs['profiling']      = profiling
        kwargs['static_argnums'] = static_argnums
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
        shape_structure  = (args_shape, args_dtype, structure)
        if shape_structure not in self._jaxpr_interpreters:
            #new input shapes or structure, need to re-trace
            jaxpr, output_shapes = self.jaxpr_function(*args)
            args,kwargs          = self._interpreter_args
            self._jaxpr_interpreters[shape_structure] = JaxprInterpreter(jaxpr, *args, **kwargs)
            self._output_shapes[shape_structure]      = output_shapes
        return self._jaxpr_interpreters[shape_structure], self._output_shapes[shape_structure]

def _restore_shapes(x, targetshapes):
    structure = jax.tree_structure(targetshapes)
    x         = jax.tree_unflatten(structure, x)
    x         = jax.tree_multimap(lambda a,shapestruct: np.asarray(a).reshape(shapestruct.shape), x, targetshapes )
    return x


def wrap(function:tp.Callable, *args, **kwargs):
    return Function(function, *args, **kwargs)
