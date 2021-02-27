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
        #array = array.view(self.dtype)
        array  = view_or_convert_from_32bit(array, self.dtype)
        #the tensor might be larger than the shape it represents
        n     = int(np.prod(self.shape))
        array = array[:n].reshape(self.shape)
        return array


def view_or_convert_from_32bit(x, dtype):
    if dtype == np.bool:
        #GLSL booleans are 32-bit, cannot simply .view()
        return x>0
    else:
        return x.view(dtype)



NP2GLSL_DTYPES_MAP = {
    np.bool_:   'bool',
    np.int32:   'int',
    np.uint32:  'uint',
    np.float32: 'float',
}

def np_to_glsl_dtypes(np_dtypes):
    glsl_dtypes = []
    for i,dt in enumerate(np_dtypes):
        if dt not in NP2GLSL_DTYPES_MAP:
            raise NotImplementedError(f'{dt} data types currently not supported')
        glsl_dtypes += [(f'DTYPE{i}', NP2GLSL_DTYPES_MAP[dt])]
    return dict(glsl_dtypes)


class Op(tp.NamedTuple):
    tensors:  tp.List[kp.Tensor]
    shader:   bytes
    equation: jax.core.JaxprEqn
    workgroup:tp.Tuple[int] = None

    @classmethod
    def construct(cls, buffers:tp.List[Buffer], shader_name:str, equation:jax.core.JaxprEqn, **consts):
        dtype_consts = np_to_glsl_dtypes([b.dtype.type for b in buffers])
        shader_bytes = shaders.get_shader(shader_name, **consts, **dtype_consts)
        tensors      = [b.tensor for b in buffers]
        workgroup    = None
        return Op(tensors, shader_bytes, equation, workgroup)


def to_shape_const_str(shape:tp.Iterable):
    '''Converts a shape to a string for use in GLSL shaders'''
    return str(tuple(shape)).replace(',)',')')





###############################################################




def element_wise_binary_op(self, equation:jax.core.JaxprEqn):
    assert equation.params=={}
    assert len(equation.invars)==2
    assert len(equation.outvars)==1
    
    outvar      = equation.outvars[0]

    inbufs      = []
    bcast_ops   = []
    for invar in equation.invars:
        buf = self.get_or_create_buffer(invar)
        if invar.aval.shape != outvar.aval.shape:
            buf,bcast_op = broadcast(self, buf, outvar, invar.aval.dtype)
            bcast_ops.append(bcast_op)
        inbufs.append(buf)
    
    outbuf = self.get_or_create_buffer(outvar)
    return bcast_ops+[Op.construct(inbufs+[outbuf], equation.primitive.name, equation)]

add = element_wise_binary_op
sub = element_wise_binary_op
mul = element_wise_binary_op
div = element_wise_binary_op
max = element_wise_binary_op
min = element_wise_binary_op
gt  = element_wise_binary_op
ge  = element_wise_binary_op
lt  = element_wise_binary_op
le  = element_wise_binary_op
eq  = element_wise_binary_op
ne  = element_wise_binary_op
#not sure but seeems to be the same
add_any = add
pow = element_wise_binary_op
shift_left             = element_wise_binary_op
shift_right_logical    = element_wise_binary_op
shift_right_arithmetic = element_wise_binary_op
rem                    = element_wise_binary_op
nextafter              = element_wise_binary_op

locals()['or']  = element_wise_binary_op
locals()['and'] = element_wise_binary_op




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
    return [Op.construct([outbuf, inbuf], equation.primitive.name, equation)]

exp = element_wise_unary_op
log = element_wise_unary_op
neg = element_wise_unary_op
abs = element_wise_unary_op
rsqrt   = element_wise_unary_op
erf     = element_wise_unary_op
erf_inv = element_wise_unary_op


def templated_unary_op(self, equation:jax.core.JaxprEqn):
    inbuf  = self.get_or_create_buffer(equation.invars[0])
    outbuf = self.get_or_create_buffer(equation.outvars[0])
    assert outbuf.shape == inbuf.shape
    assert outbuf.dtype == inbuf.dtype == np.float32

    shader_consts = dict(FUNCTION=equation.primitive.name)
    return [Op.construct([outbuf, inbuf], 'unary_op', equation, **shader_consts)]

for fname in ['cos', 'sin', 'tan', 'cosh', 'sinh', 'tanh', 'acos', 'asin', 'atan', 'acosh', 'asinh', 'atanh']:
    locals()[fname] = templated_unary_op




def broadcast(self, buf:Buffer, newvar:jax.core.Var, dtype:np.dtype):
    #newvar has the correct shape to broadcast to, but maybe a wrong dtype
    aval   = jax.core.ShapedArray(newvar.aval.shape, dtype)
    newvar = jax.core.Var(newvar.count, suffix='_broadcast', aval=aval)

    shape_in  = np.ones(len(newvar.aval.shape), dtype=int)
    for i,(olddim,newdim) in enumerate(zip(buf.shape[::-1], newvar.aval.shape[::-1])):
        if olddim!=newdim and olddim!=1:
            raise ValueError(f'Cannot broadcast from {buf.shape} to {newvar.aval}')
        shape_in[-i-1] = olddim
    
    #FIXME: this should not be a shader call
    outbuf = self.get_or_create_buffer(newvar)
    inbuf  = buf

    shader_consts = dict()
    inshape  = (1,)+inbuf.shape
    outshape = (1,)+outbuf.shape
    shader_consts['N_A']       = len(inshape)
    shader_consts['N_OUT']     = len(outshape)
    shader_consts['SHAPE_A']   = to_shape_const_str(inshape)
    shader_consts['SHAPE_OUT'] = to_shape_const_str(outshape)
    shader_consts['BCAST_DIM'] = to_shape_const_str( np.arange(len(inshape)) )

    return outbuf, Op.construct([outbuf, inbuf], 'broadcast_in_dim', 'broadcast', **shader_consts)


def broadcast_in_dim(self, equation:jax.core.JaxprEqn):
    #assert broadcast_dimensions are sorted
    assert np.all(np.diff(equation.params['broadcast_dimensions'])>0)
    invar  = equation.invars[0]
    outvar = equation.outvars[0]

    inbuf    = self.get_or_create_buffer(invar)
    if np.prod(inbuf.shape) == np.prod(outvar.aval.shape):
        #same size, just create new buffer with same tensor but new shape
        outbuf   = Buffer(inbuf.tensor, inbuf.dtype, outvar.aval.shape)
        self.buffers[outvar] = outbuf
        return []
    else:
        #size changed, need a shader call (currently)
        #FIXME: shouldn't be a shader call
        #FIXME: code duplication with broadcast()

        outbuf    = self.get_or_create_buffer(outvar)

        shader_consts = dict()
        inshape  = (1,)+inbuf.shape
        outshape = (1,)+outbuf.shape
        shader_consts['N_A']       = len(inshape)
        shader_consts['N_OUT']     = len(outshape)
        shader_consts['SHAPE_A']   = to_shape_const_str(inshape)
        shader_consts['SHAPE_OUT'] = to_shape_const_str(outshape)
        shader_consts['BCAST_DIM'] = to_shape_const_str( [0]+ [i+1 for i in equation.params['broadcast_dimensions']] )

        return [Op.construct([outbuf, inbuf], 'broadcast_in_dim', equation, **shader_consts)]


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
    return [Op.construct([outbuf], 'iota', equation)]

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
reduce_min  = reduce_op
reduce_sum  = reduce_op
reduce_prod = reduce_op
argmin      = reduce_op
argmax      = reduce_op


def select(self, equation:jax.core.JaxprEqn):
    assert equation.invars[0].aval.shape \
        == equation.invars[1].aval.shape \
        == equation.invars[2].aval.shape \
        == equation.outvars[0].aval.shape
    
    inbufs = [self.get_or_create_buffer(var) for var in equation.invars]
    outbuf = self.get_or_create_buffer(equation.outvars[0])

    return [Op.construct([outbuf]+inbufs, 'select', equation)]

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

    shader_consts = dict()
    shader_consts['COLS_A']   = inbufs[0].shape[-1]
    shader_consts['COLS_B']   = inbufs[1].shape[-1]
    shader_consts['COLS_OUT'] = outbuf.shape[-1]
    shader_consts['SIZE_OUT'] = np.prod(outbuf.shape)

    return [Op.construct([outbuf]+inbufs, 'concatenate', equation, **shader_consts)]


def gather(self, equation:jax.core.JaxprEqn):
    params = equation.params
    inbufs = [self.get_or_create_buffer(v) for v in equation.invars]
    outbuf = self.get_or_create_buffer(equation.outvars[0])

    shader_consts = dict()
    shader_consts['N_A']             = len(inbufs[0].shape)
    shader_consts['N_B']             = len(inbufs[1].shape)
    shader_consts['N_OUT']           = len(outbuf.shape)

    shader_consts['SHAPE_A']         = to_shape_const_str(inbufs[0].shape)
    shader_consts['SHAPE_B']         = to_shape_const_str(inbufs[1].shape)
    shader_consts['SHAPE_OUT']       = to_shape_const_str(outbuf.shape)

    shader_consts['START_INDEX_MAP'] = to_shape_const_str(params['dimension_numbers'].start_index_map)
    shader_consts['SLICE_SIZES']     = to_shape_const_str(params['slice_sizes'])

    offset_dims  = tuple(params['dimension_numbers'].offset_dims) + (-1,)
    noffset_dims = tuple([i for i in range(len(inbufs[1].shape)-1) if i not in offset_dims]) + (-1,)
    shader_consts['OFFSET_DIMS']     = to_shape_const_str( offset_dims)
    shader_consts['NOFFSET_DIMS']    = to_shape_const_str(noffset_dims)
    shader_consts['N_OFF']           = len( offset_dims)
    shader_consts['N_NOFF']          = len(noffset_dims)
    collapsed_dims  = params['dimension_numbers'].collapsed_slice_dims
    
    ncollapsed_dims = [i for i in range(len(inbufs[0].shape)) if i not in collapsed_dims] + [-1]
    shader_consts['NCOLLAPSED_DIMS'] = to_shape_const_str(ncollapsed_dims)
    shader_consts['N_NCOLL']         = len(ncollapsed_dims)

    return [Op.construct([outbuf]+inbufs, 'gather', equation, **shader_consts)]


def scatter_add(self, equation:jax.core.JaxprEqn):
    params = equation.params
    d0 = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,1), scatter_dims_to_operand_dims=(0,1))
    d1 = jax.lax.ScatterDimensionNumbers(update_window_dims=(0,), inserted_window_dims=(1,), scatter_dims_to_operand_dims=(1,))
    if params['dimension_numbers'] == d0:
        #equivalent to x[i[0], i[2]] += u[i[0]],  with x.shape=(B,N), i.shape=(B,1,2), u.shape=(B,)
        inbufs = [self.get_or_create_buffer(v) for v in equation.invars]
        outbuf = self.get_or_create_buffer(equation.outvars[0])

        assert len(inbufs[0].shape)==2
        assert inbufs[0].shape == outbuf.shape
        assert inbufs[1].shape[1:] == (1,2)
        assert inbufs[2].shape[0] == inbufs[0].shape[0]

        shader_bytes = shaders.get_shader('scatter0', N=inbufs[0].shape[0], M=inbufs[0].shape[1])
        return [Op([b.tensor for b in [outbuf]+inbufs], shader_bytes, equation)]
    elif params['dimension_numbers'] == d1:
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
    assert len(equation.invars)==len(equation.outvars)==1
    inbuf  = self.get_or_create_buffer(equation.invars[0])
    outvar = equation.outvars[0]
    outbuf = Buffer(inbuf.tensor, outvar.aval.dtype, outvar.aval.shape)
    self.buffers[equation.outvars[0]] = outbuf
    return []

#not relevant for us i think
stop_gradient        = noop
squeeze              = noop
bitcast_convert_type = noop




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


def integer_pow(self, equation:jax.core.JaxprEqn):
    inbuf  = self.get_or_create_buffer(equation.invars[0])
    outbuf = self.get_or_create_buffer(equation.outvars[0])

    shader_bytes = shaders.get_shader('integer_pow', Y=equation.params['y'])
    return [Op([outbuf.tensor, inbuf.tensor], shader_bytes, equation)]


def slice(self, equation:jax.core.JaxprEqn):
    inbuf  = self.get_or_create_buffer(equation.invars[0])
    outbuf = self.get_or_create_buffer(equation.outvars[0])

    shader_consts = dict()
    N                          = len(inbuf.shape)
    strides                    = equation.params['strides'] or (1,)*N
    shader_consts['N']         = N
    shader_consts['START']     = to_shape_const_str(equation.params['start_indices'])
    shader_consts['STRIDES']   = to_shape_const_str(strides)
    shader_consts['SHAPE_A']   = to_shape_const_str(inbuf.shape)
    shader_consts['SHAPE_OUT'] = to_shape_const_str(outbuf.shape)

    shader_bytes = shaders.get_shader('slice', **shader_consts)
    return [Op([outbuf.tensor, inbuf.tensor], shader_bytes, equation)]


def threefry2x32(self, equation:jax.core.JaxprEqn):
    inbufs  = [self.get_or_create_buffer(v) for v in equation.invars]
    outbufs = [self.get_or_create_buffer(v) for v in equation.outvars]

    assert inbufs[0].shape  == inbufs[1].shape
    assert inbufs[2].shape  == inbufs[3].shape
    assert outbufs[0].shape == outbufs[1].shape

    shader_consts = dict(KEY_IS_SCALAR = int(inbufs[0].shape in [(), (1,)]) )

    return [Op.construct(outbufs+inbufs, 'threefry2x32', equation, **shader_consts)]


def convert_element_type(self, equation:jax.core.JaxprEqn):
    inbuf  = self.get_or_create_buffer(equation.invars[0])
    outbuf = self.get_or_create_buffer(equation.outvars[0])
    return [Op.construct([outbuf,inbuf], 'convert_element_type', equation)]
