import vkjax
import numpy as np
import jax

GLOBAL_VAR = 0

def func0(x):
    global GLOBAL_VAR
    GLOBAL_VAR = x
    return x+1


def test_function_basic():
    vk_func = vkjax.Function(func0)
    assert vk_func(65)==66
    assert vk_func(-5)==-4


def test_shape_checking():
    global GLOBAL_VAR

    vk_func = vkjax.Function(func0)
    assert len(vk_func._jaxpr_interpreters)==0

    vk_func(65)
    assert len(vk_func._jaxpr_interpreters)==1
    assert isinstance(GLOBAL_VAR, jax.core.Tracer)
    GLOBAL_VAR = 0

    vk_func(77)
    assert len(vk_func._jaxpr_interpreters)==1
    assert GLOBAL_VAR == 0 

    vk_func(np.zeros([4,4]))
    assert len(vk_func._jaxpr_interpreters)==2
    assert isinstance(GLOBAL_VAR, jax.core.Tracer)
    GLOBAL_VAR = 0

    vk_func(np.zeros([4,4]))
    assert len(vk_func._jaxpr_interpreters)==2
    assert GLOBAL_VAR == 0 

    vk_func(np.zeros([4,5]))
    assert len(vk_func._jaxpr_interpreters)==3
    assert isinstance(GLOBAL_VAR, jax.core.Tracer)
    GLOBAL_VAR = 0