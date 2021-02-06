import numpy as np
import jax, jax.numpy as jnp
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

  #print( [np.shape(a) for a in jax.tree_leaves(args)] )
  #print( [v.aval for v in jaxpr.invars ] )
  # Bind args and consts to environment
  write(core.unitvar, core.unit)
  safe_map(write, jaxpr.invars, jax.tree_leaves(args))
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