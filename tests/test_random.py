import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax
import jax, jax.numpy as jnp, numpy as np




def test_normal0():
    for i in [0, np.random.randint(10000)]:
        key   = jax.random.PRNGKey(i)
        jaxpr = jax.make_jaxpr(jax.random.normal)(key)
        #print(jaxpr)

        vkfunc = vkjax.wrap(jax.random.normal)
        ypred  = vkfunc(key)
        ytrue = jax.random.normal(key)
        

        assert np.allclose(ytrue, ypred)
