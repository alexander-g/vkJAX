import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax, vkjax.elegy

import jax, jax.numpy as jnp, numpy as np
import elegy, optax
from common import eval_jaxpr, safe_allclose



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
            elegy.nn.Linear(self.n1),
            jax.nn.relu,
            elegy.nn.Linear(self.n2),
            jax.nn.relu,
            elegy.nn.Linear(10, b_init=elegy.initializers.RandomNormal()),
        )

        return mlp(image)



def test_basic_inference():
    x = np.random.random([2,32,32,3]).astype(np.float32)

    module = MLP()
    model  = elegy.Model(module)

    model.predict(x) #for initialization
    jaxpr       = jax.make_jaxpr(model.call_pred_step, static_argnums=[1,3])(x, 'train', model.states, False)
    print(jaxpr)
    
    vkmodel = vkjax.elegy.vkModel(module)
    vkmodel.states = model.states

    y     = vkmodel.predict(x)
    ytrue = model.predict(x)

    assert np.shape(y) == np.shape(ytrue)
    #XXX: need higher than default atol,
    assert np.allclose(y, ytrue, atol=1e-6)
    assert all( jax.tree_multimap(np.allclose, jax.tree_leaves(model.states), jax.tree_leaves(vkmodel.states)) )



def test_scce():
    scce  = lambda x,y: elegy.losses.sparse_categorical_crossentropy(y,x).mean()
    X     = np.random.random([8,10]), np.random.randint(0,10, size=8)
    jaxpr = jax.make_jaxpr(scce)(*X)
    print(jaxpr)
    vkfunc = vkjax.Function(scce)

    ypred = vkfunc(*X)
    ytrue = scce(*X)
    assert np.allclose(ypred, ytrue)

def test_scce_value_and_grad():
    scce    = lambda x,y: elegy.losses.sparse_categorical_crossentropy(y,x).mean()
    scce_vg = jax.value_and_grad(scce)
    X       = np.random.random([8,10]), np.random.randint(0,10, size=8)
    jaxpr   = jax.make_jaxpr(scce_vg)(*X)
    vkfunc = vkjax.Function(scce_vg)

    ypred = vkfunc(*X)
    ytrue = scce_vg(*X)

    assert all(jax.tree_multimap(np.allclose, ytrue, ypred))



def test_basic_training():
    B = 8
    x = np.random.random([B,32,32,3]).astype(np.float32)
    y = np.random.randint(0,10, size=B)

    module = MLP()
    model  = elegy.Model(module,
                         loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
                         optimizer=optax.sgd(0.1))
    model.maybe_initialize("train", x,y)

    
    jaxpr       = jax.make_jaxpr(model.call_train_step, static_argnums=[2,6])(x, y, 'train', None, None, model.states, False)
    print(jaxpr)
    
    vkmodel = vkjax.elegy.vkModel(module,
                                  loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
                                  optimizer=optax.sgd(0.1))
    #hacky
    vkmodel.run_eagerly = True
    vkmodel.maybe_initialize("train", x,y)
    vkmodel.run_eagerly = False
    vkmodel.states = model.states

    ypred = vkmodel.train_on_batch(x,y)
    ytrue = model.train_on_batch(x,y)

    assert jax.tree_structure(ypred) == jax.tree_structure(ytrue)
    assert all(jax.tree_leaves(jax.tree_multimap(lambda a,b: np.shape(a)==np.shape(b), jax.tree_leaves(ypred), jax.tree_leaves(ytrue))))
    #XXX:atol higher than default
    assert all(jax.tree_leaves(jax.tree_multimap(lambda a,b: np.allclose(a,b, atol=1e-7), jax.tree_leaves(ypred), jax.tree_leaves(ytrue) )))
    assert all(jax.tree_multimap(lambda a,b: np.allclose(a,b, atol=1e-7), jax.tree_leaves(model.states), jax.tree_leaves(vkmodel.states)) ) 

    #deep inspection of inner variables
    interpreter = list(vkmodel.call_train_step_jit._jaxpr_interpreters.values())[0]
    jaxpr       = interpreter.jaxpr
    _, envtrue  = eval_jaxpr(jaxpr.jaxpr, jaxpr.literals, x, y, None,None, model.states, return_env=True)
    _, envpred  = interpreter.run(x, y, 'train', None,None, model.states, False, return_all=True)
    #XXX:atol higher than default
    assert np.all([safe_allclose(envpred.get(k, None), vtrue, atol=1e-6)  for k,vtrue in envtrue.items()])




def test_basic_initialization():
    x = np.random.random([2,32,32,3]).astype(np.float32)

    module   = MLP()
    vkmodel  = vkjax.elegy.vkModel(module)

    vkmodel.predict(x)
    vkstates = vkmodel.states

    module = MLP()
    model  = elegy.Model(module)
    model.predict(x)
    states = model.states

    #NOTE: atol higher than default, limited by erf_inv
    assert np.all(jax.tree_multimap(lambda x,y: np.allclose(x,y, rtol=1e-5, atol=2e-3), jax.tree_leaves(states), jax.tree_leaves(vkstates) ))
