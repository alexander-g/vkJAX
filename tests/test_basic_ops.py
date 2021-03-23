import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax

import jax, jax.numpy as jnp, numpy as np
import pytest


seed = np.random.randint(0, 1000000)
np.random.seed(seed)



def noop(x): return x

def add0(x): return x+x
def add1(x): return x+1.0
def add2(x,y): return x+y
def add3(x,y): return jax.jit(add0)(x) + jax.jit(add2)(y,x)
def add4(x,y): return jax.jit(add0)(x) + jax.jit(add2)(y, 1.0)
def add00():   return jax.lax.add(5,100)

def div0(x,y): return x/y
def sub0(x,y): return x-y
def mul0(x,y): return x*y

def reshape0(x): return x.reshape(4,-1)       #this fails. why?
def reshape1(x): return (x+1).reshape(4,-1)
def broadcast0(x): return jax.lax.broadcast_in_dim(x, shape=(4,), broadcast_dimensions=())
def broadcast1(x): return jax.lax.broadcast_in_dim(x+1, shape=(4,1,1), broadcast_dimensions=(0,))
def broadcast2(x): 
    return jax.lax.broadcast_in_dim(x, shape=x.shape+(32,), broadcast_dimensions=tuple(np.arange(len(x.shape))))
def broadcast3(x): 
    return jax.lax.broadcast_in_dim(x, shape=(32,)+x.shape, broadcast_dimensions=(1,))

def dot0(x,y): return jnp.dot(x,y)
dot1_const = np.random.random([100,32]).astype(np.float64)
def dot1(x):   return jnp.dot(x, dot1_const)
def dot_general0(x,y): return jax.lax.dot_general(x,y, (((0,), (0,)), ((), ())) )
def dot_general1(x,y): return jax.lax.dot_general(x,y, (((1,), (1,)), ((), ())) )
def dot_general2(x,y): return jax.lax.dot_general(x,y, (((0,), (1,)), ((), ())) )

def relu0(x): return jax.nn.relu(x)

def reduce_max0(x): return jnp.max(x, axis=0)
def reduce_min0(x): return jnp.min(x, axis=0)
def reduce_max1(x): return jnp.max(x, axis=1)
def reduce_max2(x): return jnp.max(x, axis=[1,2,5])
def reduce_sum0(x): return jnp.sum(x, axis=0)
def reduce_sum1(x): return jnp.sum(x, axis=1)
def reduce_sum2(x): return jnp.sum(x, axis=[0,1,2])
def no_reduce0(x):  return jnp.sum(x, axis=())  #fails, like reshape0
def no_reduce1(x):  return jnp.sum(x+1, axis=())
def reduce_prod0(x): return jnp.prod(x, axis=0)
def argmax0(x):      return jnp.argmax(x, axis=0)
def argmin0(x):      return jnp.argmin(x, axis=0)

def gt0(x,y): return x>y
def ge0(x,y): return x>=y
def lt0(x,y): return x<y
def le0(x,y): return x<=y
def eq0(x,y): return x==y
def eq1(x):   return x==x.max(axis=-1)
def ne0(x,y): return x!=y

def or0(x,y): return x|y
def and0(x,y):return x&y

def exp0(x):  return jnp.exp(x)
def log0(x):  return jnp.log(x)
def abs0(x):  return jnp.abs(x)
def rsqrt0(x): return jax.lax.rsqrt(x)

def iota0():  return jnp.arange(32)

def select0(x,y,z):    return jnp.where(x,y,z)
def concatenate0(x,y): return jnp.concatenate([x,y], axis=-1)

def gather0(x): return x[:,:,4:7,:]
def gather1a(x): return x[5,:]
def gather1b(x): return x[:,5]

#equivalent to x[i[0], i[2]] with x.shape=(B,N), i.shape=(B,1,2)
gather_fn0 = lambda x,i: jax.lax.gather(x,
                                        i, 
                                        jax.lax.GatherDimensionNumbers( offset_dims=(),
                                                                        collapsed_slice_dims=(0,1),
                                                                        start_index_map=(0,1)),
                                        slice_sizes=(1,1),
                         )
#not a xla primitive, uses gather_fn0
def take_along_axis0(x,i): return jnp.take_along_axis(x,i, axis=-1)
#gradient of take_along_axis
def take_along_axis0_g(x,i): return jax.grad(lambda *args: take_along_axis0(*args).ravel().mean())(x,i)

#equivalent to x[:,i] with x.shape=(B,N), i.shape=(1,), i range 0...N
gather_fn1 = lambda x,i: jax.lax.gather(x,
                                        i,
                                        jax.lax.GatherDimensionNumbers(offset_dims=(0,),
                                                                       collapsed_slice_dims=(1,),
                                                                       start_index_map=(1,)),
                                        slice_sizes=(len(x),1),
                         )

#equivalent to x[i[0], i[2]] += u[i[0]],  with x.shape=(B,N), i.shape=(B,1,2), u.shape=(B,1)
scatter_fn0 = lambda x,i,u: jax.lax.scatter_add(
    x, i, u, jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,1),
        scatter_dims_to_operand_dims=(0,1)),
)

#equivalent to x[:,i]+=u with x.shape=(B,N), i.shape=(1,), u.shape=(B,)
scatter_add_fn1 = lambda x,i,u: jax.lax.scatter_add(
    x, i, u, jax.lax.ScatterDimensionNumbers(
        update_window_dims=(0,),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,)),
)

#no idea what exactly the primitive "add_any" does but looks like a simple add
#XXX: the +(1,) is a workaround to make the interpreter return a tuple
def add_any0(p, ct): return jax.vjp(lambda x:x+x, p)[1](ct)+(np.float32(1),)

def transpose0(x): return x.T
def rev0(x): return jax.lax.rev(x, dimensions=[1,2])
def integer_pow0(x): return x**2
def integer_pow1(x): return x**5
def pow0(x,y):       return jax.lax.pow(x,y)

def slice0(x):       return jax.lax.slice(x, [2], [33])
def slice1(x):       return jax.lax.slice(x, [55,5], [101,10])
def slice2(x):       return jax.lax.slice(x, [55,5], [101,10], [2,3])
def squeeze0(x):     return jnp.squeeze(x)

def threefry0a():    return jax.random.threefry2x32_p.bind(*np.ones(4, 'uint32'))
def threefry0b():    return jax.random.threefry2x32_p.bind(*np.ones([4,10], 'uint32'))
def threefry1(x):    return jax.random.threefry2x32_p.bind(*x)

def convert_element_type0(x): return x.astype(np.int32)
def convert_element_type1(x): return x.astype(np.float32)
def bitcast_convert_type0(x): return jax.lax.bitcast_convert_type(x, 'float32')

shift_left             = jax.lax.shift_left
shift_right_logical    = jax.lax.shift_right_logical
shift_right_arithmetic = jax.lax.shift_right_arithmetic

def shift_right_logical_1_32():
    return jax.lax.shift_right_logical(1, 32)

erf     = jax.lax.erf
erf_inv = jax.lax.erf_inv
rem     = jax.lax.rem
min     = jax.lax.min
max     = jax.lax.max

nextafter = jax.lax.nextafter


param_matrix = [
    #(noop, 'noop',                      [5.0]),

    (add0, 'add x+x scalar',            [5.0], ),
    (add0, 'add x+x array1d',           [np.random.random(32)] ),
    (add0, 'add x+x array3d',           [np.random.random((32,32,32))] ),
    (add00,'5+100',                     []),

    (add1, 'add x+1 scalar',            [5.0] ),
    (add1, 'add x+1 array3d',           [np.random.random((32,32,32))] ),

    (add2, 'add x+y scalar-scalar',     [5.0, 7.0]),
    (add2, 'add x+y scalar-array3d',    [5.0, np.random.random((32,32,32))]),
    (add2, 'add x+y array3d-array3d',   [np.random.random((32,32,32)), np.random.random((32,32,32))]),
    (add2, 'broadcast_add [2,32]+[32]', [np.random.random((2,32)), np.random.random((32))]),

    (add2, 'add x+y int',               [np.random.randint(65,size=(32,32,32)), np.random.randint(77, size=(32,32,32))]),

    (add3, 'add nested (x+x)+(y+x)',    [5.0, 7.1]),
    (add3, 'add nested (x+x)+(y+x)',    [5.0, 7.1]),

    (add4, 'nested const (x+x)+(y+1)',  [5.0, 7.1]),

    (div0, 'div0 x/y',                  [np.random.random([2,32,32,3]), 255.0]),
    (sub0, 'sub0 x-y',                  [np.random.random([2,32,32,3]), 255.0]),
    (sub0, 'sub0 (2,10)-(2,1)',         [np.random.random([2,10]), np.random.random([2,1])]),
    (mul0, 'mul0 x*y',                  [np.random.random([2,32,32,3]), 255.0]),

    #(reshape0, 'reshape0',              [np.random.random([2,32,32]) ] ),
    (reshape1, 'reshape1',              [np.random.random([2,32,32]) ] ),
    (broadcast0, 'broadcast scalar',    [np.random.random()]),
    (broadcast1, 'broadcast 1D->3D',    [np.random.random(4)]),
    (broadcast2, 'broadcast append',    [np.random.random(4)]),
    (broadcast3, '(128)->(32,128)',     [np.random.random(128)]),

    (dot0, 'dot0 x@y',                  [np.random.random([2,100]), np.random.random([100,32])] ),
    #(dot0, 'dot0 x@y',                  [np.random.random([2,256])*256-128, np.random.random([256,32])*256-128] ),
    (dot1, 'dot1 x@const',              [np.random.random([2,100])] ),
    (dot_general0, 'dot axes=(0,0)',    [np.random.random([100,2]), np.random.random([100,32])] ),
    (dot_general1, 'dot axes=(1,1)',    [np.random.random([2,100]), np.random.random([32,100])] ),
    (dot_general2, 'dot axes=(0,1)',    [np.random.random([100,2]), np.random.random([32,100])] ),

    (relu0, 'relu0',                    [np.random.random([32,32,32])-0.5]),
    (max,   'max_float32',              [np.random.random([77,99,200]), np.random.random([77,99,200])]),
    (min,   'min_float32',              [np.random.random([77,99,200]), np.random.random([77,99,200])]),
    (max,   'max_int32',                [np.random.randint(-1000,1000,[77,99,200]), 
                                         np.random.randint(-1000,1000,[77,99,200])]),
    (min,   'min_int32',                [np.random.randint(-1000,1000,[77,99,200]), 
                                         np.random.randint(-1000,1000,[77,99,200])]),


    (reduce_max0, 'max(axis=0)',        [np.random.random([32,32])]),
    (reduce_min0, 'min(axis=0)',        [np.random.random([77,99])]),
    (reduce_max1, 'max(axis=1)',        [np.random.random([32,32])-1.0]),
    (reduce_max2, 'max(axis=125)',      [np.random.random([3,33,67,99,4,7])]),
    (reduce_sum0, 'sum(axis=0) 2D',     [np.random.random([32,32])]),
    (reduce_sum0, 'sum(axis=0) 1D',     [np.random.random([32])]),
    (reduce_sum1, 'sum(axis=1)',        [np.random.random([32,32])]),
    (reduce_sum2, 'sum(axis=012)',      [np.random.random([65,77,22,7])]),
    #(no_reduce0, 'no_reduce0(axis=())', [np.random.random([32])]),
    (no_reduce1, 'no_reduce1(axis=())', [np.random.random([32])]),
    (argmax0, 'argmax(axis=0)',         [np.random.random([99,77])]),
    (argmin0, 'argmin(axis=0)',         [np.random.random([99,77])]),
    
    (reduce_prod0,'prod(axis=0)',       [np.random.random([32,32])+0.5]),

    (gt0, 'gt0',                        [np.random.random([32,32]), np.random.random([32,32])]),
    (ge0, 'ge0',                        [np.random.random([32,32]), np.random.random([32,32])]),
    (ge0, 'ge0 float>=bool',            [np.random.random([32,32])+0.5, np.random.random([32,32])>0.5]),
    (ge0, 'ge0 bool>=float',            [np.random.random([32,32])>0.5, np.random.random([32,32])+0.5]),
    (lt0, 'lt0',                        [np.random.random([32,32]), np.random.random([32,32])]),
    (lt0, 'lt0 int[]<scalar',           [np.random.randint(999,size=[999]), 555]),
    (le0, 'le0',                        [np.random.random([77,99]), np.random.random([77,99])]),
    (eq0, 'eq0',                        [np.random.randint(0,3, size=[32,32]).astype(np.float32), 
                                         np.random.randint(0,3, size=[32,32]).astype(np.float32) ]),
    (eq1, 'eq1 x==x.max(-1)',           [np.random.random([32,32])]),
    (ne0, 'x!=y',                       [np.random.random([77,99]), np.random.random([77,99])]),
    
    (or0, 'or0 x|y',                    [np.random.random([77,32]).view('uint32'), np.random.random([77,32]).view('uint32')]),
    (and0,'and0 x|y',                   [np.random.random([77,32]).view('uint32'), np.random.random([77,32]).view('uint32')]),

    (exp0, 'exp(x)',                    [np.random.uniform(0,5,size=[32,32])]),
    #(log0, 'log(x)',                    [np.random.uniform(0,5,size=[32,32])]), #fails, why? numerical issues?
    (log0, 'log(x+100)',                [np.random.uniform(0,5,size=[32,32])+100]),
    (abs0, 'abs(x)',                    [np.random.random([32,32,32])]),
    (rsqrt0, 'rsqrt(x)',                [np.random.random([77,40,32])*10]),

    (iota0, 'iota0',                    []),

    (select0, 'select0',                [np.random.random([32,32])>0.5, np.ones([32,32]), np.zeros([32,32])]),
    (concatenate0, 'concatenate0',      [np.random.random([32,32,32]), np.random.random([32,32,16])]),

    (gather0, 'gather0',                [np.random.random([100,100,100,100])]),
    (gather1a, 'gather1a x[5,:]',       [np.random.random([8,8])]),
    (gather1b, 'gather1b x[:,5]',       [np.random.random([8,8])]),
    (gather_fn0, 'gather_fn0',          [np.random.random([32,10]), 
                                         np.c_[np.random.randint(0,32, size=[32]), 
                                               np.random.randint(0,10, size=[32])].reshape(32,1,2) ]),
    (scatter_fn0, 'scatter_fn0',        [np.random.random([32,10]), 
                                         np.c_[np.random.randint(0,32, size=[32]), 
                                               np.random.randint(0,10, size=[32])].reshape(32,1,2),
                                         np.random.random([32]).reshape(32,1) ]),
    (take_along_axis0, 'take_along0',   [np.random.random([32,10]), np.random.randint(0,10, size=32)[:,np.newaxis] ]),
    (take_along_axis0_g, 'take0_grad',  [np.random.random([32,10]), np.random.randint(0,10, size=32)[:,np.newaxis] ]),
    #(take_along_axis0_g, 'take0_grad',  [np.random.random([3,4]), np.random.randint(0,4, size=3)[:,np.newaxis] ]),

    (gather_fn1, 'gather_fn1',          [np.random.random([32,10]), np.random.randint(0,10, size=[1]) ]),
    (scatter_add_fn1, 'scatter_add1',   [np.random.random([32,10]), np.random.randint(0,10, size=[1]), np.random.random(32)]),

    (add_any0, 'add_any0',              [np.random.random([32,32]), np.random.random([32,32])]),

    (transpose0, 'random([N,N]).T',     [np.random.random([32,32])]),
    (transpose0, 'random([N,M]).T',     [np.random.random([32,65])]),
    (rev0,       'rev0 dims=1,2',       [np.random.random([33,77,88,11])]),

    (integer_pow0, 'x**2',              [np.random.random([77,9,35])]),
    (integer_pow1, 'x**5',              [np.random.random([77,9,35])*2-1]),
    (integer_pow1, 'int**5',            [np.random.randint(-1000, 1000, size=[77,9,35])]),
    (pow0, 'x**scalar',                 [np.random.random([77,9,35]), np.random.random()]),

    (slice0, '1-D slice(x, [2],[5])',   [np.random.random([99])]),
    (slice1, '2-D slice no strides',    [np.random.random([199,99])]),
    (slice2, '2-D slice + strides',     [np.random.random([199,99])]),
    (squeeze0, '5-D squeeze',           [np.random.random([199,99,1,1,5])]),

    (threefry0a,'all const size=1',     [] ),
    (threefry0b,'all const size=n',     [] ),
    (threefry1, 'all zero size=1',      [np.zeros(4).astype('uint32')] ),
    (threefry1, 'all ones size=1',      [np.ones(4).astype('uint32')] ),
    (threefry1, 'all random size=1',    [np.random.randint(0,10000000, size=4).astype('uint32')] ),
    (threefry1, 'all random size=65',   [np.random.randint(0,10000000, size=(4,65)).astype('uint32')] ),
    (threefry1, 'scalar_key_size=100',  [list(np.random.randint(0,10000000, size=(2,)).astype('uint32')) \
                                        +list(np.random.randint(0,10000000, size=(2,100)).astype('uint32')) ] ),

    (convert_element_type0, 'float2int',[np.random.random([77,101])*65]),
    (convert_element_type1, 'int2float',[np.random.randint(77,101, size=(99,99))]),
    (convert_element_type1,'bool2float',[np.random.random([77,101])>0.5]),
    (bitcast_convert_type0,'uint2float',[np.random.random([77,101]).view('uint32')]),

    (shift_left,             'x<<1',    [np.arange(-777,+777), 1]),
    (shift_left,             'x<<y',    [np.arange(-777,+777), np.random.randint(0,20, size=777*2)]),
    (shift_right_logical,    'x>>1',    [np.arange(-777,+777), 1]),
    (shift_right_logical_1_32,'1>>32',  []),
    (shift_right_logical, 'uint>>1',    [np.arange(-777,+777).view('uint32'), np.uint32(9)]),
    (shift_right_arithmetic, 'x>>1',    [np.arange(-777,+777), 1]),
    (shift_right_arithmetic, 'uint32',  [np.arange(-777,+777).view(np.uint32), np.uint32(1)]),

    (erf,     'erf(x)',                 [np.random.random([111,283])*10-5]),
    (erf_inv, 'erf_inv(x)',             [np.random.random([111,283])*2 -1]),
    (rem,     'rem(x,y)',               [np.random.randint(1000,10000, size=[77,99]), np.random.randint(1000)]),

    (nextafter, 'nextafter(X,inf)',     [np.random.random([77,101])*2-1, np.inf]),
    (nextafter, 'nextafter(X,-inf)',    [np.random.random([77,101])*2-1, -np.inf]),
    (nextafter, 'nextafter(X,0)',       [np.random.random([77,101])*2-1, 0.0]),
    (nextafter, 'nextafter(X,Y)',       [np.random.random([77,101])*2-1, np.random.random([77,101])]),
]

for fname in ['cos', 'sin', 'tan', 'cosh', 'sinh', 'tanh', 
              'acos', 'asin', 'atan', 'acosh', 'asinh', 'atanh',
              'ceil', 'floor', 'sign']:
    param_matrix += [ (getattr(jax.lax, fname), fname, [np.random.random([77,101])*2-1] ) ]


TOLERANCES = {
    'erf(x)':            (1e-5, 1e-6),
    'erf_inv(x)':        (1e-5, 2e-3),      #high atol
    'sum(axis=012)':     (1e-4, 1e-8),      #reduce_sum2 
    'sin':               (1e-5, 1e-6),
    'sinh':              (1e-5, 1e-6),
    'tan':               (1e-5, 1e-6),
}


@pytest.mark.parametrize("f,desc,args", param_matrix)
def test_matrix_kompute_interpreter(f, desc, args):
    print(f'==========TEST START: {f.__name__} {desc}==========')
    print(f'**********RANDOM SEED: {seed}*********')
    args = jax.tree_map(jnp.asarray, args)
    jaxpr = jax.make_jaxpr(f)(*args)
    print(jaxpr)
    vkfunc = vkjax.Function(f)

    y     = vkfunc(*args)
    ytrue = f(*args)

    print(y)
    print()
    print(ytrue)


    assert jax.tree_structure(y) == jax.tree_structure(ytrue)
    assert np.all(jax.tree_leaves(jax.tree_multimap(lambda x,y: np.shape(x)==np.shape(y), y,ytrue)))
    dtype = lambda x: np.asarray(x).dtype
    assert np.all(jax.tree_leaves(jax.tree_multimap(lambda x,y: dtype(x)==dtype(y),       y,ytrue)))
    
    tols = TOLERANCES.get(desc, [])
    assert np.all(jax.tree_leaves(jax.tree_multimap(lambda x,y: np.allclose(x,y, *tols, equal_nan=True),  y,ytrue)))

    print(f'==========TEST END:  {desc}==========')
    print()




def test_nextafter():
    x = np.random.random([77,101,5])

    vkfunc = vkjax.wrap(nextafter)
    ypred = vkfunc(x, np.inf)
    assert np.all(ypred > x)

    ypred = vkfunc(x, -np.inf)
    assert np.all(x > ypred)

    ypred = vkfunc(x, 0.0)
    assert np.all( np.sign(ypred-x) == np.sign(0-x) )

    x2 = np.random.random(x.shape)
    ypred = vkfunc(x, x2)
    assert np.all( np.sign(ypred-x) == np.sign(x2-x) )
