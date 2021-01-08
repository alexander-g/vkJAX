import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax.kompute_jaxpr_interpreter as vkji

import jax, jax.numpy as jnp, numpy as np
import elegy, optax




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



def test_scce():
    scce  = lambda x,y: elegy.losses.sparse_categorical_crossentropy(y,x).mean()
    X     = np.random.random([8,10]), np.random.randint(0,10, size=8)
    jaxpr = jax.make_jaxpr(scce)(*X)
    interpreter = vkji.JaxprInterpreter(jaxpr)

    ypred = interpreter.run(*X)
    ytrue = scce(*X)
    assert np.allclose(ypred, ytrue)

def test_scce_value_and_grad():
    scce    = lambda x,y: elegy.losses.sparse_categorical_crossentropy(y,x).mean()
    scce_vg = jax.value_and_grad(scce)
    X       = np.random.random([8,10]), np.random.randint(0,10, size=8)
    jaxpr   = jax.make_jaxpr(scce_vg)(*X)
    interpreter = vkji.JaxprInterpreter(jaxpr)

    ypred = interpreter.run(*X)
    ytrue = scce_vg(*X)

    assert all(jax.tree_multimap(np.allclose, ytrue, ypred))



def test_basic_training():
    x = np.random.random([2,32,32,3]).astype(np.float32)
    y = np.random.randint(0,10, size=2)
    #y = np.random.random([2,10]).astype(np.float32) #for mae

    mlp_module = MLP()
    mlp_model  = elegy.Model(mlp_module,
                             loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
                             #loss=elegy.losses.MeanAbsoluteError(),
                             optimizer=optax.sgd(0.1))
    mlp_model.maybe_initialize(elegy.model.model_base.Mode.train, x,y)

    #XXX: setting RNG to low number because uint32 not yet implemented
    #everything converted to floats, results in truncation
    elegy.module.set_rng(elegy.random.RNG(0))
    
    train_jit, _get_state_fn, _set_state_fn   = elegy.module.jit(mlp_model.train_fn,   modules=mlp_model, unwrapped=True)
    state = _get_state_fn()
    
    jaxpr, outshapes = jax.make_jaxpr(train_jit, static_argnums=(0,1), return_shape=True)(*state, x,y)
    _set_state_fn(state)

    interpreter = vkji.JaxprInterpreter(jaxpr, static_argnums=(0,1))

    #state = _get_state_fn()
    ypred     = interpreter.run(*state, x,y)
    ypred     = jax.tree_unflatten(jax.tree_structure(outshapes), ypred)
    #_set_state_fn(state)
    
    #state = _get_state_fn()
    ytrue = train_jit(*state, x,y)
    _set_state_fn(state)


    #import pickle
    #open('ypred.pkl', 'wb').write(pickle.dumps(ypred))
    #open('ytrue.pkl', 'wb').write(pickle.dumps(ytrue))

    assert jax.tree_structure(ypred) == jax.tree_structure(ytrue)

    assert all(jax.tree_leaves(jax.tree_multimap(lambda a,b: np.shape(a)==np.shape(b), jax.tree_leaves(ypred), jax.tree_leaves(ytrue))))
    
    #XXX: atol value is quite high
    assert all(jax.tree_leaves(jax.tree_multimap(lambda a,b: np.allclose(a,b, atol=1e-2), jax.tree_leaves(ypred), jax.tree_leaves(ytrue) )))
