import vkjax.parser as parser
import vkjax.kompute_interpreter as vki

import jax, numpy as np






def test_basic_scalar():
    f = lambda x: x+x
    hlo = jax.xla_computation(f)(5.0).as_hlo_text()
    functions = parser.parse_hlo(hlo)

    interpreter = vki.KomputeInterpreter(functions)
    y = interpreter.run(x=5.0)
    print(y)
    assert y==10.0
    

def test_basic_array1d():
    f = lambda x: x+x
    X = np.linspace(0,1,32)
    hlo = jax.xla_computation(f)(X).as_hlo_text()
    functions = parser.parse_hlo(hlo)

    interpreter = vki.KomputeInterpreter(functions)
    y = interpreter.run(x=X)
    print(y)
    assert y.shape == X.shape
    assert np.allclose(y, X+X)
    




