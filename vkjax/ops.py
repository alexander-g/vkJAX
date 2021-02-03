import typing as tp

from . import shaders

import kp
import numpy as np
import jax

class Buffer(tp.NamedTuple):
    tensor : kp.Tensor
    dtype  : np.dtype
    shape  : tp.Tuple

    def numpy(self):
        array = self.tensor.numpy()
        #the tensor might be larger than the shape it represents
        n     = int(np.prod(self.shape))
        array = array[:n].reshape(self.shape)
        return array


class Op(tp.NamedTuple):
    tensors:  tp.List[kp.Tensor]
    shader:   bytes
    equation : jax.core.JaxprEqn



def element_wise_binary_op(self, equation:jax.core.JaxprEqn):
    assert equation.params=={}
    assert len(equation.invars)==2
    assert len(equation.outvars)==1
    
    outvar      = equation.outvars[0]

    intensors   = []
    bcast_ops   = []
    for invar in equation.invars:
        buf = self.get_or_create_buffer(invar)
        if invar.aval.shape != outvar.aval.shape:
            buf,bcast_op = broadcast(self, buf, outvar)
            bcast_ops.append(bcast_op)
        intensors.append(buf.tensor)
    
    outbuf = self.get_or_create_buffer(outvar)
    shader_bytes = shaders.get_shader(equation.primitive.name)
    return bcast_ops+[Op(intensors+[outbuf.tensor], shader_bytes, equation)]

add = element_wise_binary_op
sub = element_wise_binary_op
mul = element_wise_binary_op
div = element_wise_binary_op
max = element_wise_binary_op
gt  = element_wise_binary_op
ge  = element_wise_binary_op
lt  = element_wise_binary_op
eq  = element_wise_binary_op
#not sure but seeems to be the same
add_any = add




def element_wise_unary_op(self, equation:jax.core.JaxprEqn):
    assert equation.params=={}
    assert len(equation.invars)==1
    assert len(equation.outvars)==1

    outvar      = equation.outvars[0]
    invar       = equation.invars[0]

    assert outvar.aval.shape == invar.aval.shape
    assert outvar.aval.dtype == invar.aval.dtype == np.float32

    inbuf  = self.get_or_create_buffer(invar)
    outbuf = self.get_or_create_buffer(outvar)
    shader_bytes = shaders.get_shader(equation.primitive.name)
    return [Op([outbuf.tensor, inbuf.tensor], shader_bytes, equation)]

exp = element_wise_unary_op
log = element_wise_unary_op
neg = element_wise_unary_op
abs = element_wise_unary_op
rsqrt = element_wise_unary_op




def broadcast(self, buf:Buffer, newvar:jax.core.Var):
    shape_in  = np.ones(len(newvar.aval.shape), dtype=int)
    for i,(olddim,newdim) in enumerate(zip(buf.shape[::-1], newvar.aval.shape[::-1])):
        if olddim!=newdim and olddim!=1:
            raise ValueError(f'Cannot broadcast from {buf.shape} to {newvar.aval}')
        shape_in[-i-1] = olddim
    
    #FIXME: this should not be a shader call
    outbuf = self.get_or_create_buffer(newvar)
    shape_in  = ','.join(map(str, shape_in))
    shape_out = ','.join(map(str, outbuf.shape))
    n         = len(outbuf.shape)
    shader_bytes = shaders.get_shader('broadcast_in_dim', N=n, SHAPE_IN=shape_in, SHAPE_OUT=shape_out)
    return outbuf, Op([outbuf.tensor, buf.tensor], shader_bytes, 'broadcast')


def broadcast_in_dim(self, equation:jax.core.JaxprEqn):
    #assert broadcast_dimensions are sorted
    assert np.all(np.diff(equation.params['broadcast_dimensions'])>0)
    invar  = equation.invars[0]
    outvar = equation.outvars[0]
    assert len(invar.aval.shape) in [1,0]

    inbuf    = self.get_or_create_buffer(invar)
    if np.prod(inbuf.shape) == np.prod(outvar.aval.shape):
        #same size, just create new buffer with same tensor but new shape
        outbuf   = Buffer(inbuf.tensor, inbuf.dtype, outvar.aval.shape)
        self.buffers[outvar] = outbuf
        return []
    else:
        #size changed, need a shader call (currently)
        #FIXME: shouldn't be a shader call
        outbuf    = self.get_or_create_buffer(outvar)
        shape_in  = np.ones(len(outbuf.shape), dtype=int)
        for i,bcastdim in enumerate(equation.params['broadcast_dimensions']):
            shape_in[bcastdim] = inbuf.shape[i]
        shape_in  = ','.join(map(str, shape_in))
        shape_out = ','.join(map(str, outbuf.shape))
        n         = len(outbuf.shape)
        shader_bytes = shaders.get_shader('broadcast_in_dim', N=n, SHAPE_IN=shape_in, SHAPE_OUT=shape_out)
        return [Op([outbuf.tensor, inbuf.tensor], shader_bytes, equation)]

def xla_call(self, equation:jax.core.JaxprEqn):
    assert equation.params['device'] == None
    assert equation.params['backend'] == None

    jaxpr = equation.params['call_jaxpr']
    assert len(equation.invars) == len(jaxpr.invars)
    assert len(equation.outvars) == len(jaxpr.outvars)
    for eq_var, jaxpr_var in zip(equation.invars, jaxpr.invars):
        if str(eq_var)=='*':
            #ignored
            continue
        #connect call parameters to the function-local variables
        self.buffers[jaxpr_var] = self.get_or_create_buffer(eq_var)
    
    all_ops = self.analyze_jaxpr(jaxpr)
    
    for eq_var, jaxpr_var in zip(equation.outvars, jaxpr.outvars):
        if str(eq_var)=='_':
            #ignored
            continue
        if str(jaxpr_var)=='*':
            #strange, but can happen that * gets assigned a variable
            self.buffers[eq_var] = None
            continue
        #connect the function-local variables ot the output variables
        self.buffers[eq_var] = self.get_or_create_buffer(jaxpr_var)
    return all_ops

def custom_jvp_call_jaxpr(self, equation:jax.core.JaxprEqn):
    assert equation.params['num_consts'] == 0
    
    jaxpr = equation.params['fun_jaxpr'].jaxpr
    #FIXME: code duplication with xla_call()
    assert len(equation.invars) == len(jaxpr.invars)
    assert len(equation.outvars) == len(jaxpr.outvars)
    for eq_var, jaxpr_var in zip(equation.invars, jaxpr.invars):
        self.buffers[jaxpr_var] = self.get_or_create_buffer(eq_var)
    
    all_ops = self.analyze_jaxpr(jaxpr)
    
    for eq_var, jaxpr_var in zip(equation.outvars, jaxpr.outvars):
        self.buffers[eq_var] = self.buffers[jaxpr_var]
    return all_ops


def reshape(self, equation:jax.core.JaxprEqn):
    assert equation.params['dimensions'] == None
    assert len(equation.outvars) == 1
    assert len(equation.invars) == 1
    invar  = equation.invars[0]
    outvar = equation.outvars[0]
    assert invar.aval.dtype == outvar.aval.dtype

    #no real operation needed, just replace the shape in the buffer
    buffer = self.get_or_create_buffer(invar)
    newbuffer = Buffer(buffer.tensor, buffer.dtype, outvar.aval.shape)
    self.buffers[outvar] = newbuffer
    return []

def dot_general(self, equation:jax.core.JaxprEqn):
    assert equation.params['precision'] == None
    dim_numbers = equation.params['dimension_numbers']
    assert dim_numbers[1] == ((),())
    assert dim_numbers[0][0] in [(0,), (1,)]
    assert dim_numbers[0][1] in [(0,), (1,)]
    assert len(equation.invars) == 2
    assert len(equation.outvars) == 1
    assert all([v.aval.dtype==np.float32 for v in equation.invars+equation.outvars])

    inbufs  = [self.get_or_create_buffer(v) for v in equation.invars]
    outbuf  = self.get_or_create_buffer(equation.outvars[0])
    N, M    = outbuf.shape[0],  outbuf.shape[1]
    C       = inbufs[0].shape[dim_numbers[0][0][0]]
    cdim_a  = dim_numbers[0][0][0]
    cdim_b  = dim_numbers[0][1][0]

    shader_bytes = shaders.get_shader(equation.primitive, N=N, C=C, M=M, CDIM_A=cdim_a, CDIM_B=cdim_b)
    return [Op([b.tensor for b in [outbuf]+inbufs], shader_bytes, equation)]

def iota(self, equation:jax.core.JaxprEqn):
    assert equation.params['dimension'] == 0
    outbuf = self.get_or_create_buffer(equation.outvars[0])
    shader_bytes = shaders.get_shader(equation.primitive)
    #self.sequence.record_algo_data([outbuf.tensor], shader_bytes)
    return [Op([outbuf.tensor], shader_bytes, equation)]

def reduce_op(self, equation:jax.core.JaxprEqn):
    axes  = equation.params['axes']
    if axes==():
        #strange but can happen -> noop
        return noop(self, equation)
    outvar = equation.outvars[0]
    invar  = equation.invars[0]
    if len(invar.aval.shape)==1:
        invar.aval.shape = invar.aval.shape+(1,)

    inbuf  = self.get_or_create_buffer(invar)
    outbuf = self.get_or_create_buffer(outvar)
    
    reduced_shape = [s if i not in axes else 1 for i,s  in enumerate(inbuf.shape)]
    reduce_dims   = [s if i     in axes else 1 for i,s  in enumerate(inbuf.shape)]

    shader_consts = dict()
    shader_consts['N']           = len(inbuf.shape)
    shader_consts['SHAPE_A']     = ','.join(map(str,inbuf.shape))
    shader_consts['SHAPE_OUT']   = ','.join(map(str,reduced_shape))
    shader_consts['REDUCE_DIMS'] = ','.join(map(str, reduce_dims))
    shader_consts['REDUCE_SIZE'] = np.prod(reduce_dims)

    shader_bytes = shaders.get_shader(equation.primitive.name, **shader_consts)
    return [Op([outbuf.tensor, inbuf.tensor], shader_bytes, equation)]

reduce_max  = reduce_op
reduce_sum  = reduce_op
reduce_prod = reduce_op


def select(self, equation:jax.core.JaxprEqn):
    assert equation.invars[0].aval.shape \
        == equation.invars[1].aval.shape \
        == equation.invars[2].aval.shape \
        == equation.outvars[0].aval.shape
    
    inbufs = [self.get_or_create_buffer(var) for var in equation.invars]
    outbuf = self.get_or_create_buffer(equation.outvars[0])

    shader_bytes = shaders.get_shader(equation.primitive.name)
    return [Op([b.tensor for b in [outbuf]+inbufs], shader_bytes, equation)]

def concatenate(self, equation:jax.core.JaxprEqn):
    #currently only support for concatenation of 2 parameters
    assert len(equation.invars)==2
    #currently only supporting the last axis
    ndims = len(equation.outvars[0].aval.shape)
    assert equation.params['dimension'] == ndims-1
    #all other dimensions must be equal
    assert equation.outvars[0].aval.shape[:-1] \
        == equation.invars[0].aval.shape[:-1]  \
        == equation.invars[1].aval.shape[:-1]

    inbufs = [self.get_or_create_buffer(var) for var in equation.invars]
    outbuf = self.get_or_create_buffer(equation.outvars[0])

    cols_a   = inbufs[0].shape[-1]
    cols_b   = inbufs[1].shape[-1]
    cols_out = outbuf.shape[-1]
    size_out = np.prod(outbuf.shape)

    shader_bytes = shaders.get_shader(equation.primitive.name, COLS_A=cols_a, COLS_B=cols_b, COLS_OUT=cols_out, SIZE_OUT=size_out)
    return [Op([b.tensor for b in [outbuf]+inbufs], shader_bytes, equation)]


def gather(self, equation:jax.core.JaxprEqn):
    d0 = jax.lax.GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0, 1), start_index_map=(0, 1))
    d1 = jax.lax.GatherDimensionNumbers(offset_dims=(0,), collapsed_slice_dims=(1,), start_index_map=(1,))
    if equation.params['dimension_numbers'] == d0 and equation.params['slice_sizes'] == (1,1):
        #equivalent to x[i[0], i[2]] with x.shape=(B,N), i.shape=(B,1,2)
        inbufs = [self.get_or_create_buffer(v) for v in equation.invars]
        outbuf = self.get_or_create_buffer(equation.outvars[0])

        assert len(inbufs[0].shape)==2
        assert inbufs[1].shape[1:] == (1,2)

        shader_bytes = shaders.get_shader('gather0', N=inbufs[0].shape[0], M=inbufs[0].shape[1])
        #self.sequence.record_algo_data([b.tensor for b in [outbuf]+inbufs], shader_bytes)
        return [Op([b.tensor for b in [outbuf]+inbufs], shader_bytes, equation)]
    elif equation.params['dimension_numbers'] == d1 and equation.params['slice_sizes'][1] == 1:
        #equivalent to x[:,i] with x.shape=(B,N), i.shape=(1,), i range 0...N
        inbufs = [self.get_or_create_buffer(v) for v in equation.invars]
        outbuf = self.get_or_create_buffer(equation.outvars[0])

        assert len(inbufs[0].shape)==2
        assert inbufs[1].shape==(1,)

        shader_bytes = shaders.get_shader('gather1', N=inbufs[0].shape[0], M=inbufs[0].shape[1])
        return [Op([b.tensor for b in [outbuf]+inbufs], shader_bytes, equation)]
    else:
        raise NotImplementedError(equation)

def scatter_add(self, equation:jax.core.JaxprEqn):
    d0 = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,1), scatter_dims_to_operand_dims=(0,1))
    d1 = jax.lax.ScatterDimensionNumbers(update_window_dims=(0,), inserted_window_dims=(1,), scatter_dims_to_operand_dims=(1,))
    if equation.params['dimension_numbers'] == d0:
        #equivalent to x[i[0], i[2]] += u[i[0]],  with x.shape=(B,N), i.shape=(B,1,2), u.shape=(B,)
        inbufs = [self.get_or_create_buffer(v) for v in equation.invars]
        outbuf = self.get_or_create_buffer(equation.outvars[0])

        assert len(inbufs[0].shape)==2
        assert inbufs[0].shape == outbuf.shape
        assert inbufs[1].shape[1:] == (1,2)
        assert inbufs[2].shape[0] == inbufs[0].shape[0]

        shader_bytes = shaders.get_shader('scatter0', N=inbufs[0].shape[0], M=inbufs[0].shape[1])
        return [Op([b.tensor for b in [outbuf]+inbufs], shader_bytes, equation)]
    elif equation.params['dimension_numbers'] == d1:
        #equivalent to x[:,i]+=u with x.shape=(B,N), i.shape=(1,), u.shape=(B,)
        inbufs = [self.get_or_create_buffer(v) for v in equation.invars]
        outbuf = self.get_or_create_buffer(equation.outvars[0])

        assert len(inbufs[0].shape)==2
        assert inbufs[0].shape == outbuf.shape
        assert inbufs[1].shape==(1,)

        shader_bytes = shaders.get_shader('scatter1', N=inbufs[0].shape[0], M=inbufs[0].shape[1])
        return [Op([b.tensor for b in [outbuf]+inbufs], shader_bytes, equation)]
    else:
        raise NotImplementedError(equation)

def transpose(self, equation:jax.core.JaxprEqn):
    assert equation.params['permutation'] == (1,0)

    inbuf  = self.get_or_create_buffer(equation.invars[0])
    outbuf = self.get_or_create_buffer(equation.outvars[0])

    shader_bytes = shaders.get_shader(equation.primitive.name, N=inbuf.shape[0], M=inbuf.shape[1])
    return [Op([outbuf.tensor, inbuf.tensor], shader_bytes, equation)]


def noop(self, equation:jax.core.JaxprEqn):
    #does not perform any operations
    #simply re-uses the input buffer
    assert len(equation.invars)==1
    inbuf = self.get_or_create_buffer(equation.invars[0])
    self.buffers[equation.outvars[0]] = inbuf
    return []

#currently using float32 for everything
convert_element_type = noop
#not relevant for us i think
stop_gradient        = noop




def conv_general_dilated(self, equation:jax.core.JaxprEqn):
    params = equation.params
    assert params['precision']           == None
    assert params['batch_group_count']   == 1
    assert params['feature_group_count'] == 1
    assert len(equation.outvars[0].aval.shape)    == 4  #2D conv

    shader_consts = dict()
    shader_consts['N']           = 4;   #number of dimensions
    shader_consts['SHAPE_A']     = ','.join(map(str,equation.invars[0].aval.shape))
    shader_consts['SHAPE_B']     = ','.join(map(str,equation.invars[1].aval.shape))
    shader_consts['SHAPE_OUT']   = ','.join(map(str,equation.outvars[0].aval.shape))
    dim_numbers = params['dimension_numbers']
    shader_consts['SPEC_LHS']    = ','.join(map(str, dim_numbers.lhs_spec))
    shader_consts['SPEC_RHS']    = ','.join(map(str, dim_numbers.rhs_spec))
    shader_consts['SPEC_OUT']    = ','.join(map(str, dim_numbers.out_spec))
    shader_consts['PADDING']     = f'{params["padding"][0][0]},{params["padding"][1][0]}'
    shader_consts['STRIDES']     = ','.join(map(str, params["window_strides"]))
    shader_consts['DILATE_RHS']  = ','.join(map(str, params["rhs_dilation"]))
    shader_consts['DILATE_LHS']  = ','.join(map(str, params["lhs_dilation"]))
    
    inbufs = [self.get_or_create_buffer(v) for v in equation.invars]
    outbuf = self.get_or_create_buffer(equation.outvars[0])

    shader_bytes = shaders.get_shader('conv2d', **shader_consts)
    return [Op([b.tensor for b in [outbuf]+inbufs], shader_bytes, equation)]



def rev(self, equation:jax.core.JaxprEqn):
    inbuf  = self.get_or_create_buffer(equation.invars[0])
    outbuf = self.get_or_create_buffer(equation.outvars[0])

    reversed_dims = [int(i in equation.params['dimensions']) for i in range(len(inbuf.shape))]

    shader_consts = dict()
    shader_consts['N']             = len(inbuf.shape)   #number of dimensions
    shader_consts['SHAPE']         = inbuf.shape
    shader_consts['REVERSED_DIMS'] = tuple(reversed_dims)

    shader_bytes = shaders.get_shader('rev', **shader_consts)
    return [Op([outbuf.tensor, inbuf.tensor], shader_bytes, equation)]


def reduce_window_max(self, equation:jax.core.JaxprEqn):
    #only 2D implemented
    assert len(equation.outvars[0].aval.shape) == 4, NotImplemented
    params = equation.params
    assert params['base_dilation']   == (1,1,1,1), NotImplemented
    assert params['window_dilation'] == (1,1,1,1), NotImplemented

    shader_consts = dict()
    shader_consts['N']           = 4;   #number of dimensions
    shader_consts['SHAPE_A']     = equation.invars[0].aval.shape
    shader_consts['SHAPE_OUT']   = equation.outvars[0].aval.shape
    shader_consts['PADDING']     = tuple(p[0] for p in params['padding'])
    shader_consts['STRIDES']     = params["window_strides"]
    shader_consts['WINDOW']      = params["window_dimensions"]


    inbuf  = self.get_or_create_buffer(equation.invars[0])
    outbuf = self.get_or_create_buffer(equation.outvars[0])

    shader_bytes = shaders.get_shader('reduce_window_max_2d', **shader_consts)
    return [Op([outbuf.tensor, inbuf.tensor], shader_bytes, equation)]
