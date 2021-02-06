import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax

import jax, jax.numpy as jnp, numpy as np
import pytest


seed = np.random.randint(0, 1000000)
np.random.seed(seed)

#2d maxpooling
def reduce_window_max0(x): return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1,2,2,1), window_strides=(1,1,1,1), padding='VALID')
def reduce_window_max1(x): return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1,3,3,1), window_strides=(1,2,2,1), padding='SAME')


param_matrix = [
    (reduce_window_max0, '2x2 no-pad',               [np.random.random([11,100,111,5])] ),
    (reduce_window_max1, '3x3 +pad',                 [np.random.random([77,10,99,17])] ),
]




@pytest.mark.parametrize("f,desc,args", param_matrix)
def test_reduce_window_matrix(f, desc, args):
    print(f'==========TEST START: {desc}==========')
    print(f'**********RANDOM SEED: {seed}*********')
    args = jax.tree_map(jnp.asarray, args)
    jaxpr = jax.make_jaxpr(f)(*args)
    print(jaxpr)
    vkfunc = vkjax.Function(f)

    y     = vkfunc(*args)
    ytrue = f(*args)

    #print(args[0].reshape(4,5).round(5))
    print()


    print(y.squeeze())
    print()
    print(ytrue.squeeze())

    assert jax.tree_structure(y) == jax.tree_structure(ytrue)
    assert np.all(jax.tree_leaves(jax.tree_multimap(lambda x,y: np.shape(x)==np.shape(y), y,ytrue)))
    dtype = lambda x: np.asarray(x).dtype
    assert np.all(jax.tree_leaves(jax.tree_multimap(lambda x,y: dtype(x)==dtype(y),       y,ytrue)))
    assert np.all(jax.tree_leaves(jax.tree_multimap(lambda x,y: np.allclose(x,y),         y,ytrue)))
    #assert np.all(jax.tree_leaves(jax.tree_multimap(lambda x,y: np.all(x==y),         y,ytrue)))

    print(f'==========TEST END:  {desc}==========')
    print()
