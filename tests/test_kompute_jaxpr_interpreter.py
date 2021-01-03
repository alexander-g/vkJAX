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

def reshape0(x): return x.reshape(4,-1)       #this fails. why?
def reshape1(x): return (x+1).reshape(4,-1)

def dot0(x,y): return jnp.dot(x,y)
dot1_const = np.random.random([100,32]).astype(np.float64)
def dot1(x):   return jnp.dot(x, dot1_const)

def relu0(x): return jax.nn.relu(x)


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

    #(reshape0, 'reshape0',              [np.random.random([2,32,32]) ] ),
    (reshape1, 'reshape1',              [np.random.random([2,32,32]) ] ),

    (dot0, 'dot0 x@y',                  [np.random.random([2,100]), np.random.random([100,32])] ),
    (dot1, 'dot1 x@const',              [np.random.random([2,100])] ),

    (relu0, 'relu0',                    [np.random.random([32,32,32])-0.5])
]


@pytest.mark.parametrize("f,desc,args", param_matrix)
def test_matrix_kompute_interpreter(f, desc, args):
    print(f'==========TEST START: {desc}==========')
    jaxpr = jax.make_jaxpr(f)(*args)
    print(jaxpr)
    interpreter = vkji.JaxprInterpreter(jaxpr)

    y     = interpreter.run(*args)
    ytrue = f(*args)

    assert np.shape(y) == np.shape(ytrue)
    assert np.allclose(y, ytrue)

    print(f'==========TEST END:  {desc}==========')
    print()

