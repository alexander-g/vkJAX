import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import vkjax.kompute_jaxpr_interpreter as vkji

import jax, jax.numpy as jnp, numpy as np
import pytest


seed = np.random.randint(0, 1000000)
np.random.seed(seed)


kernel_1x1 = np.random.random([1,1,5,33])  #1x1 convolution kernel, 5 input channels, 2 outputchannels
#kernel_1x1 = np.random.random([1,1,2,1])  #1x1 convolution kernel, 5 input channels, 2 outputchannels
def conv0(x): 
    return jax.lax.conv_general_dilated(x, kernel_1x1, 
                                        window_strides=(1,1), padding='VALID', 
                                        dimension_numbers=jax.lax.ConvDimensionNumbers((0,3,1,2), (3,2,0,1), (0,3,1,2)) )

def conv1(x, k): 
    return jax.lax.conv_general_dilated(x, k, 
                                        window_strides=(1,1), padding='VALID', 
                                        dimension_numbers=jax.lax.ConvDimensionNumbers((0,3,1,2), (3,2,0,1), (0,3,1,2)) )
def conv1a(x, k): 
    return jax.lax.conv_general_dilated(x, k, 
                                        window_strides=(1,1), padding='VALID', 
                                        dimension_numbers=jax.lax.ConvDimensionNumbers((0,1,2,3), (0,1,2,3), (0,1,2,3)) )

def conv2(x, k): 
    return jax.lax.conv_general_dilated(x, k, 
                                        window_strides=(1,1), padding='SAME', 
                                        dimension_numbers=jax.lax.ConvDimensionNumbers((0,3,1,2), (3,2,0,1), (0,3,1,2)) )

def conv3(x, k):
    return jax.lax.conv_general_dilated(x, k, 
                                        window_strides=(1,1), padding=[(2,0),(0,3)], 
                                        dimension_numbers=jax.lax.ConvDimensionNumbers((0,3,1,2), (3,2,0,1), (0,3,1,2)) )

def conv4(x, k):
    return jax.lax.conv_general_dilated(x, k, 
                                        window_strides=(2,2), padding='SAME', 
                                        dimension_numbers=jax.lax.ConvDimensionNumbers((0,3,1,2), (3,2,0,1), (0,3,1,2)) )

def conv5(x, k):
    return jax.lax.conv_general_dilated(x, k, 
                                        #window_strides=(2,2), padding='SAME',
                                        window_strides=(2,2), padding='VALID',
                                        rhs_dilation=(2,2),
                                        dimension_numbers=jax.lax.ConvDimensionNumbers((0,3,1,2), (3,2,0,1), (0,3,1,2)) )

def conv6a(x, k):
    return jax.lax.conv_general_dilated(x, k, 
                                        window_strides=(1,1), padding=[(2,2), (3,3)],
                                        lhs_dilation=(2,2),
                                        dimension_numbers=jax.lax.ConvDimensionNumbers((0,3,1,2), (3,2,0,1), (0,3,1,2)) )

def conv6b(x, k):
    return jax.lax.conv_general_dilated(x, k, 
                                        #window_strides=(2,2), padding=[(5,5), (3,3)],
                                        window_strides=(2,2), padding=[(0,0), (0,0)],
                                        lhs_dilation=(2,2),
                                        dimension_numbers=jax.lax.ConvDimensionNumbers((0,3,1,2), (3,2,0,1), (0,3,1,2)) )


param_matrix = [
    (conv0, 'conv0 1x1 const kernel no pad',                 [np.random.random([11,100,100,5])] ),
    
    (conv1, 'conv1 1x1 var kernel no pad',                   [np.random.random([11,100,100,33]), np.random.random([1,1,33,11])] ),
    (conv1, 'conv1 3x3 var kernel no pad',                   [np.random.random([40,65,33,5]),    np.random.random([3,3,5,7])] ),
    (conv1a,'conv1a 3x3 convdims 0123',                      [np.random.random([40,8,65,35]),    np.random.random([39,8,3,3])] ),

    (conv2, 'conv2 1x1 var kernel +pad',                     [np.random.random([77,17,9,12]), np.random.random([1,1,12,11])] ),
    (conv2, 'conv2 3x3 var kernel +pad',                     [np.random.random([23,44,19,7]), np.random.random([3,3,7,38])] ),
    (conv2, 'conv2 7x7 var kernel +pad',                     [np.random.random([8,12,19,3]),  np.random.random([7,7,3,4])] ),
    (conv3, 'conv3 3x3 uneven pad',                          [np.random.random([15,67,42,11]), np.random.random([3,3,11,38])] ),

    (conv4, 'conv4 1x1 window strides=2',                    [np.random.random([2,67,42,3]), np.random.random([1,1,3,2])] ),
    (conv4, 'conv4 3x3 window strides=2',                    [np.random.random([15,67,42,11]), np.random.random([3,3,11,7])] ),

    (conv5, 'conv5 3x3 window strides=2 + rhs_dilate=2',     [np.random.random([15,67,42,11]), np.random.random([3,3,11,7])] ),

    (conv6a, 'conv6a 3x3 window strides=2 + lhs_dilate=2',   [np.random.random([15,67,42,11]), np.random.random([3,3,11,7])] ),
    (conv6b, 'conv6b 3x3 window strides=2 + lhs_dilate=2',   [np.random.random([15,67,42,11]), np.random.random([3,3,11,7])] ),
]




@pytest.mark.parametrize("f,desc,args", param_matrix)
def test_convmatrix_kompute_interpreter(f, desc, args):
    print(f'==========TEST START: {desc}==========')
    print(f'**********RANDOM SEED: {seed}*********')
    args = jax.tree_map(jnp.asarray, args)
    jaxpr = jax.make_jaxpr(f)(*args)
    print(jaxpr)
    interpreter = vkji.JaxprInterpreter(jaxpr)

    y     = interpreter.run(*args, profile=False)
    ytrue = f(*args)

    #print(args[0].reshape(3,3).round(5))
    #print(args[1].reshape(2,2))
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
