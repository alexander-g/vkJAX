


import numpy as np
import jax
import kp

import vkjax
from vkjax.buffers import Buffer, BufferPool



def test_lowlevel():
    mgr  = kp.Manager()
    pool = BufferPool(mgr, 1)
    
    v0 = jax.core.Var(0, '', jax.core.ShapedArray((10,10), 'float32') )
    pool.get_buffer(v0, increment_op_counter=True)
    
    assert pool.op_counter == 1
    v1 = jax.core.Var(1, '', jax.core.ShapedArray((10,10), 'float32') )
    pool.get_buffer(v1, increment_op_counter=True)
    
    assert pool.op_counter == 2
    pool.get_buffer(v0, increment_op_counter=True)

    #this one should re-use a previous tensor
    assert pool.op_counter == 3
    v3 = jax.core.Var(3, '', jax.core.ShapedArray((10,10), 'float32') )
    pool.get_buffer(v3, increment_op_counter=True)

    assert pool.buffers[v0].accesses == [0,2]
    assert pool.buffers[v1].accesses == [1]
    assert pool.buffers[v3].accesses == [3]

    pool.create_tensors()

    #only 2 tensors should have been created
    unique_tensors = set([b.tensor for b in pool.buffers.values()])
    assert len( unique_tensors ) == 2


def test_highlevel():
    
    def func(a,b):
        c = a+5
        d = c*b
        e = d-2
        f = e**3
        g = f-1
        h = g*5
        return h
    
    a0,b0 = 65, 5
    jaxpr0 = jax.make_jaxpr(func)(a0,b0)
    print(jaxpr0)

    vkfunc0 = vkjax.wrap(func, reuse_buffers=True)
    assert vkfunc0(a0,b0) == func(a0,b0)
    pool = vkfunc0._get_or_create_jaxpr_interpreter((a0,b0))[0].bufferpool

    unique_tensors = set([b.tensor for b in pool.buffers.values()])
    assert len( unique_tensors ) == 8  #11 buffers overall, -3 re-used

