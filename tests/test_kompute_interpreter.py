import vkjax.parser as parser
import vkjax.kompute_interpreter as vki

import jax, numpy as np
import pytest


def add0(x): return x+x
def add1(x): return x+1.0
def add2(x,y): return x+y
def add3(x,y): return jax.jit(add0)(x) + jax.jit(add2)(y,x)



param_matrix = [
    (add0, 'add x+x scalar',            [5.0], ),
    (add0, 'add x+x array1d',           [np.random.random(32)] ),
    (add0, 'add x+x array3d',           [np.random.random((32,32,32))] ),

    (add1, 'add x+1 scalar',            [5.0] ),
    (add1, 'add x+1 array3d',           [np.random.random((32,32,32))] ),

    (add2, 'add x+y scalar-scalar',     [5.0, 7.0]),
    (add2, 'add x+y scalar-array3d',    [5.0, np.random.random((32,32,32))]),
    (add2, 'add x+y array3d-array3d',   [np.random.random((32,32,32)), np.random.random((32,32,32))]),

    (add3, 'add nested (x+x)+(y+x)',    [5.0, 7.1]),
]


@pytest.mark.parametrize("f,desc,args", param_matrix)
def test_matrix_kompute_interpreter(f, desc, args):
    print(f'==========TEST START: {desc}==========')
    hlo = jax.xla_computation(f)(*args).as_hlo_text()
    functions = parser.parse_hlo(hlo)
    interpreter = vki.KomputeInterpreter(functions)

    y     = interpreter.run(*args)
    ytrue = f(*args)

    assert np.shape(y) == np.shape(ytrue)
    assert np.allclose(y, ytrue)

    print(f'==========TEST END:  {desc}==========')
    print()

