import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax.kompute_jaxpr_interpreter as vkji

import jax, jax.numpy as jnp, numpy as np
import elegy, optax




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

    model.predict(x) #for initialization
    jaxpr       = jax.make_jaxpr(model.predict_fn)(x)
    print(jaxpr)
    interpreter = vkji.JaxprInterpreter(jaxpr)

    y     = interpreter.run(x)
    ytrue = model.predict(x)

    assert np.shape(y) == np.shape(ytrue)
    #XXX: need higher than default atol,
    assert np.allclose(y, ytrue, atol=1e-6)




def test_basic_training():
    B = 8
    x = np.random.random([B,32,32,3]).astype(np.float32)
    y = np.random.randint(0,10, size=B)

    module = Module0()
    model  = elegy.Model(module,
                         loss=elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
                         optimizer=optax.sgd(0.1))
    model.maybe_initialize(elegy.model.model_base.Mode.train, x,y)

    #XXX: setting RNG to low number because uint32 not yet implemented
    #everything converted to floats, results in truncation
    elegy.module.set_rng(elegy.random.RNG(0))
    
    train_jit, _get_state_fn, _set_state_fn   = elegy.module.jit(model.train_fn,   modules=model, unwrapped=True)
    state = _get_state_fn()
    
    jaxpr, outshapes = jax.make_jaxpr(train_jit, static_argnums=(0,1), return_shape=True)(*state, x,y)
    print(jaxpr)
    _set_state_fn(state)

    interpreter = vkji.JaxprInterpreter(jaxpr, static_argnums=(0,1))

    #state = _get_state_fn()
    ypred     = interpreter.run(*state, x,y)
    ypred     = jax.tree_unflatten(jax.tree_structure(outshapes), ypred)
    #_set_state_fn(state)
    
    #state = _get_state_fn()
    ytrue = train_jit(*state, x,y)
    _set_state_fn(state)


    assert jax.tree_structure(ypred) == jax.tree_structure(ytrue)
    assert all(jax.tree_leaves(jax.tree_multimap(lambda a,b: np.shape(a)==np.shape(b), jax.tree_leaves(ypred), jax.tree_leaves(ytrue))))
    #XXX:atol higher than default
    assert all(jax.tree_leaves(jax.tree_multimap(lambda a,b: np.allclose(a,b, atol=1e-7), jax.tree_leaves(ypred), jax.tree_leaves(ytrue) )))

    #deep inspection of inner variables
    state = _get_state_fn()
    _, envtrue = eval_jaxpr(jaxpr.jaxpr, jaxpr.literals, *jax.tree_leaves(state[2:]), x, y, return_env=True)
    _, envpred = interpreter.run(*state, x,y, return_all=True)
    #XXX:atol higher than default
    assert np.all([safe_allclose(envpred.get(k, None), vtrue, atol=1e-6)  for k,vtrue in envtrue.items()])












from jax import core
from jax import lax
from jax.util import safe_map

#from https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html, modified
def eval_jaxpr(jaxpr, consts, *args, return_env=False):
  # Mapping from variable -> value
  env = {}

  def read(var):
    # Literals are values baked into the Jaxpr
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val

  # Bind args and consts to environment
  write(core.unitvar, core.unit)
  safe_map(write, jaxpr.invars, args)
  safe_map(write, jaxpr.constvars, consts)

  # Loop through equations and evaluate primitives using `bind`
  for eqn in jaxpr.eqns:
    # Read inputs to equation from environment
    invals = safe_map(read, eqn.invars)
    if eqn.primitive.name=='xla_call':
        outvals, innerenv = eval_jaxpr(eqn.params['call_jaxpr'], [], *invals, return_env=True)
        env.update(innerenv)
    else:
        # `bind` is how a primitive is called
        outvals = eqn.primitive.bind(*invals, **eqn.params)
    # Primitives may return multiple outputs or not
    if not eqn.primitive.multiple_results:
      outvals = [outvals]
    # Write the results of the primitive into the environment
    safe_map(write, eqn.outvars, outvals)
  # Read the final result of the Jaxpr from the environment
  if return_env:
        return safe_map(read, jaxpr.outvars), env
  else:
        return safe_map(read, jaxpr.outvars)

def safe_allclose(x,y, *args, **kwargs):
    if x is None and isinstance(y, jax.core.Unit):
        return True
    if np.any(np.isnan(y)):
        return True
    return np.allclose(x,y, *args, **kwargs)

def safe_abs_max_error(x,y):
    if x is None and isinstance(y, jax.core.Unit):
        return None
    if np.shape(x)==(0,) or np.shape(y)==(0,):
        return None
    return np.abs(x-y).max()