import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax, vkjax.elegy
from vkjax.kompute_jaxpr_interpreter import JaxprInterpreter

import jax, jax.numpy as jnp, numpy as np
import elegy, optax
from common import eval_jaxpr, safe_allclose




class Module0(elegy.Module):
    def call(self, x: jnp.ndarray):
        x = elegy.nn.Conv2D(32, (3,3), stride=(2,2))(x)
        x = jax.nn.relu(x)
        x = elegy.nn.Conv2D(32, (3,3), stride=(2,2))(x)
        x = jax.nn.relu(x)
        x = elegy.nn.Flatten()(x)
        x = elegy.nn.Linear(10)(x)
        
        return x



def test_basic_inference0():
    x = np.random.random([5,32,32,3]).astype(np.float32)

    module = Module0()
    model  = elegy.Model(module)

    model.predict(x, initialize=True) #for initialization
    jaxpr       = jax.make_jaxpr(model.call_pred_step, static_argnums=[2,3])(x, model.states, False, False)
    print(jaxpr)
    
    vkmodel = vkjax.elegy.vkModel(module)
    vkmodel.states = model.states
    vkmodel.initialized=True

    y     = vkmodel.predict(x)
    ytrue = model.predict(x)

    assert np.shape(y) == np.shape(ytrue)
    #XXX: need higher than default atol,
    assert np.allclose(y, ytrue, atol=1e-6)
    assert all( jax.tree_multimap(np.allclose, jax.tree_leaves(model.states), jax.tree_leaves(vkmodel.states)) )



def test_basic_training():
    B = 8
    x = np.random.random([B,32,32,3]).astype(np.float32)
    y = np.random.randint(0,10, size=B)

    module = Module0()
    model  = elegy.Model(module,
                         loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
                         optimizer=optax.sgd(0.1))
    model.init(x,y)

    jaxpr       = jax.make_jaxpr(model.call_train_step, static_argnums=[5,6])(x, y, None, None, model.states, False, False)
    print(jaxpr)
    
    vkmodel = vkjax.elegy.vkModel(module,
                                  loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
                                  optimizer=optax.sgd(0.1))
    
    #hacky
    vkmodel.run_eagerly = True
    vkmodel.init(x,y)
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
    #re-instantiate interpreter without memory buffer re-use
    bufferpool  = vkjax.buffers.BufferPool(interpreter.mgr, interpreter.workgroup_size, reuse_tensors=False)
    interpreter = JaxprInterpreter(jaxpr, interpreter.mgr, bufferpool, static_argnums=interpreter.static_argnums)

    _, envtrue  = eval_jaxpr(jaxpr.jaxpr, jaxpr.literals, x, y, None,None, model.states, return_env=True)
    _, envpred  = interpreter.run(x, y, None,None, model.states, False, False, return_all=True)
    
    #XXX:atol higher than default
    assert np.all([safe_allclose(envpred.get(k, None), vtrue, atol=5e-5)  for k,vtrue in envtrue.items()])



