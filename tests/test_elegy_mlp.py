import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax.kompute_jaxpr_interpreter as vkji

import jax, jax.numpy as jnp, numpy as np
import elegy




class MLP(elegy.Module):
    """Standard LeNet-300-100 MLP network."""
    def __init__(self, n1: int = 300, n2: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.n1 = n1
        self.n2 = n2

    def call(self, image: jnp.ndarray):
        image = image.astype(jnp.float32) / 255.0

        mlp = elegy.nn.sequential(
            elegy.nn.Flatten(),
            elegy.nn.Linear(self.n1, with_bias=False),
            jax.nn.relu,
            elegy.nn.Linear(self.n2),
            jax.nn.relu,
            elegy.nn.Linear(10),
        )

        return mlp(image)



def test_basic_inference():
    x = np.random.random([2,32,32,3]).astype(np.float32)

    mlp_module = MLP()
    mlp_model  = elegy.Model(mlp_module)

    mlp_model.predict(x) #for initialization
    jaxpr       = jax.make_jaxpr(mlp_model.predict_fn)(x)
    print(jaxpr)
    interpreter = vkji.JaxprInterpreter(jaxpr)

    y     = interpreter.run(x)
    ytrue = mlp_model.predict(x)

    assert np.shape(y) == np.shape(ytrue)
    assert np.allclose(y, ytrue)
