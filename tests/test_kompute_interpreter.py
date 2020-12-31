import vkjax.parser as parser
import vkjax.kompute_interpreter as vki

import jax, numpy as np
import pytest


def add0(x): return x+x
def add1(x): return x+1.0



param_matrix = [
    (add0, 'add x+x scalar',            [5.0], ),
    (add0, 'add x+x array1d',           [np.random.random(32)] ),
    (add0, 'add x+x array3d',           [np.random.random((32,32,32))] ),

    (add1, 'add x+1 scalar',            [5.0] ),
    (add1, 'add x+1 array3d',           [np.random.random((32,32,32))] ),
]


@pytest.mark.parametrize("f,desc,args", param_matrix)
def test_matrix_kompute_interpreter(f, desc, args):
    hlo = jax.xla_computation(f)(*args).as_hlo_text()
    functions = parser.parse_hlo(hlo)
    interpreter = vki.KomputeInterpreter(functions)

    y     = interpreter.run(*args)
    ytrue = f(*args)

    assert np.shape(y) == np.shape(ytrue)
    assert np.allclose(y, ytrue)

