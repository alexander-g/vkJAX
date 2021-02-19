import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax
import jax, jax.numpy as jnp, numpy as np



def test_normal0():
    vkfunc = vkjax.wrap(jax.random.normal)
    for i in [0] + list(np.random.randint(10000, size=50)):
        key   = jax.random.PRNGKey(i)
        jaxpr = jax.make_jaxpr(jax.random.normal)(key)
        #print(jaxpr)

        ypred  = vkfunc(key)
        ytrue  = jax.random.normal(key)
        
        print(ypred)
        print()
        print(ytrue)

        assert np.allclose(ytrue, ypred, rtol=1e-5, atol=2e-3) #high atol, as in erf_inv


def test_normal1():
    i = np.random.randint(10000)
    key   = jax.random.PRNGKey(i)

    fn    = lambda k: jax.random.normal(k, shape=[2,34])
    jaxpr = jax.make_jaxpr(fn)(key)
    #print(jaxpr)

    vkfunc = vkjax.wrap(fn)
    ypred  = vkfunc(key)
    ytrue  = fn(key)
    
    print(ypred)
    print()
    print(ytrue)

    assert np.allclose(ytrue, ypred, rtol=1e-5, atol=2e-3) #high atol, as in erf_inv


def test_randint0():
    fn     = lambda k: jax.random.randint(k, shape=[1], minval=0,maxval=9999)
    vkfunc = vkjax.wrap(fn)
    for i in [0] + list(np.random.randint(10000, size=50)):
        key   = jax.random.PRNGKey(i)
        jaxpr = jax.make_jaxpr(jax.random.normal)(key)
        #print(jaxpr)

        ypred  = vkfunc(key)
        ytrue  = fn(key)
        
        print(ypred)
        print()
        print(ytrue)

        assert np.all(ypred == ytrue)
