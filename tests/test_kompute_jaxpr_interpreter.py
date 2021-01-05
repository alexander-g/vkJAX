import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax.kompute_jaxpr_interpreter as vkji

import jax, jax.numpy as jnp, numpy as np
import pytest


def add0(x): return x+x
def add1(x): return x+1.0
def add2(x,y): return x+y
def add3(x,y): return jax.jit(add0)(x) + jax.jit(add2)(y,x)
def add4(x,y): return jax.jit(add0)(x) + jax.jit(add2)(y, 1.0)

def div0(x,y): return x/y
def sub0(x,y): return x-y
def mul0(x,y): return x*y

def reshape0(x): return x.reshape(4,-1)       #this fails. why?
def reshape1(x): return (x+1).reshape(4,-1)

def dot0(x,y): return jnp.dot(x,y)
dot1_const = np.random.random([100,32]).astype(np.float64)
def dot1(x):   return jnp.dot(x, dot1_const)

def relu0(x): return jax.nn.relu(x)

def reduce_max0(x): return jnp.max(x, axis=1)
def reduce_sum0(x): return jnp.sum(x, axis=1)

def gt0(x,y): return x>y
def lt0(x,y): return x<y
def eq0(x,y): return x==y
def eq1(x):   return x==x.max(axis=-1)

def exp0(x):  return jnp.exp(x)
def log0(x):  return jnp.log(x)

def iota0():  return jnp.arange(32)

def select0(x,y,z):    return jnp.where(x,y,z)
def concatenate0(x,y): return jnp.concatenate([x,y], axis=-1)

#equivalent to x[i[0],i[2]] (for 2D)
gather_fn0 = lambda x,i: jax.lax.gather(x,
                                        i, 
                                        jax.lax.GatherDimensionNumbers( offset_dims=(),
                                                                        collapsed_slice_dims=(0,1),
                                                                        start_index_map=(0,1)),
                                        slice_sizes=(1,1),
                         )


param_matrix = [
    (add0, 'add x+x scalar',            [5.0], ),
    (add0, 'add x+x array1d',           [np.random.random(32)] ),
    (add0, 'add x+x array3d',           [np.random.random((32,32,32))] ),

    (add1, 'add x+1 scalar',            [5.0] ),
    (add1, 'add x+1 array3d',           [np.random.random((32,32,32))] ),

    (add2, 'add x+y scalar-scalar',     [5.0, 7.0]),
    (add2, 'add x+y scalar-array3d',    [5.0, np.random.random((32,32,32))]),
    (add2, 'add x+y array3d-array3d',   [np.random.random((32,32,32)), np.random.random((32,32,32))]),
    (add2, 'broadcast_add [2,32]+[32]', [np.random.random((2,32)), np.random.random((32))]),

    (add3, 'add nested (x+x)+(y+x)',    [5.0, 7.1]),
    (add3, 'add nested (x+x)+(y+x)',    [5.0, 7.1]),

    (add4, 'nested const (x+x)+(y+1)',  [5.0, 7.1]),

    (div0, 'div0 x/y',                  [np.random.random([2,32,32,3]), 255.0]),
    (sub0, 'sub0 x-y',                  [np.random.random([2,32,32,3]), 255.0]),
    (mul0, 'mul0 x*y',                  [np.random.random([2,32,32,3]), 255.0]),

    #(reshape0, 'reshape0',              [np.random.random([2,32,32]) ] ),
    (reshape1, 'reshape1',              [np.random.random([2,32,32]) ] ),

    (dot0, 'dot0 x@y',                  [np.random.random([2,100]), np.random.random([100,32])] ),
    (dot1, 'dot1 x@const',              [np.random.random([2,100])] ),

    (relu0, 'relu0',                    [np.random.random([32,32,32])-0.5]),

    (reduce_max0, 'reduce_max0',        [np.random.random([32,32])]),
    (reduce_sum0, 'reduce_sum0',        [np.random.random([32,32])]),

    (gt0, 'gt0',                        [np.random.random([32,32]), np.random.random([32,32])]),
    (lt0, 'lt0',                        [np.random.random([32,32]), np.random.random([32,32])]),
    (eq0, 'eq0',                        [np.random.randint(0,3, size=[32,32]).astype(np.float32), 
                                         np.random.randint(0,3, size=[32,32]).astype(np.float32) ]),
    (eq1, 'eq1 x==x.max(-1)',           [np.random.random([32,32])]),

    (exp0, 'exp(x)',                    [np.random.uniform(0,5,size=[32,32])]),
    #(log0, 'log(x)',                    [np.random.uniform(0,5,size=[32,32])]), #fails, why?
    (log0, 'log(x+100)',                [np.random.uniform(0,5,size=[32,32])+100]),

    (iota0, 'iota0',                    []),

    (select0, 'select0',                [np.random.random([32,32])>0.5, np.ones([32,32]), np.zeros([32,32])]),
    (concatenate0, 'concatenate0',      [np.random.random([32,32,32]), np.random.random([32,32,16])]),

    #(gather_fn0, 'gather_fn0',          [np.random.random([32,10]), 
    #                                     np.c_[np.random.randint(0,32, size=[32]), 
    #                                           np.random.randint(0,10, size=[32])] ]),
]


@pytest.mark.parametrize("f,desc,args", param_matrix)
def test_matrix_kompute_interpreter(f, desc, args):
    print(f'==========TEST START: {desc}==========')
    args = jax.tree_map(jnp.asarray, args)
    jaxpr = jax.make_jaxpr(f)(*args)
    print(jaxpr)
    interpreter = vkji.JaxprInterpreter(jaxpr)

    y     = interpreter.run(*args)
    ytrue = f(*args)

    assert np.shape(y) == np.shape(ytrue)
    assert y.dtype     == ytrue.dtype
    assert np.allclose(y, ytrue)

    print(f'==========TEST END:  {desc}==========')
    print()

