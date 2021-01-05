import typing as tp

from . import shaders

import kp
import numpy as np
import jax


class Buffer(tp.NamedTuple):
    tensor : kp.Tensor
    dtype  : np.dtype
    shape  : tp.Tuple


class JaxprInterpreter:
    def __init__(self, jaxpr:jax.core.ClosedJaxpr):
        self.jaxpr     = jaxpr
        #dict mapping from jax.core.Var or int (for jax.core.Literal) to Buffer
        self.buffers   = {}
        self.mgr       = kp.Manager()

        self.analyze_closed_jaxpr(jaxpr)
    
    def analyze_closed_jaxpr(self, jaxpr:jax.core.ClosedJaxpr):
        '''Starts the analysis of the top level jaxpr.
           Records operations into a kp.Sequence and creates required buffers.'''
        self.sequence    = self.mgr.create_sequence("")
        self.sequence.begin()
        
        assert len(jaxpr.consts) == len(jaxpr.jaxpr.constvars)
        for constvar, constval in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
            self.get_or_create_buffer(constvar, initial_value=constval)

        self.analyze_jaxpr(jaxpr.jaxpr)
    
        output_tensors = [self.buffers[var].tensor for var in jaxpr.jaxpr.outvars]
        self.sequence.record_tensor_sync_local(output_tensors)
        self.sequence.end()
    
    def analyze_jaxpr(self, jaxpr:jax.core.Jaxpr):
        '''Analyzes (possibly inner) jaxprs'''
        for eq in jaxpr.eqns:
            if all([str(v)=='_' for v in eq.outvars]):
                #seems like a redundant operation
                continue
            method = getattr(self, eq.primitive.name, None)
            if method is None:
                raise NotImplementedError(eq.primitive.name)
            method(eq)

    def run(self, *X):
        '''Executes a previously recorded sequence with actual data'''
        input_tensors = [self.buffers[var].tensor for var in self.jaxpr.jaxpr.invars]
        assert len(input_tensors) == len(X)
        for input_tensor,x in zip(input_tensors,X):
            input_tensor.set_data(np.ravel(x))

        self.sequence.eval()
        
        output_bufs    = [self.buffers[var] for var in self.jaxpr.jaxpr.outvars]
        output_values  = [var.tensor.numpy().reshape(var.shape) for var in output_bufs]
        output_values  = [np.asarray(x, dtype=var.aval.dtype) for x,var in zip(output_values, self.jaxpr.jaxpr.outvars)]

        if len(output_values)==1:
            output_values = output_values[0]
        return output_values



    def element_wise_binary_op(self, equation:jax.core.JaxprEqn):
        assert equation.params=={}
        assert len(equation.invars)==2
        assert len(equation.outvars)==1
        
        outvar      = equation.outvars[0]

        intensors   = []
        for invar in equation.invars:
            buf = self.get_or_create_buffer(invar)
            if invar.aval.shape != outvar.aval.shape:
                buf = self.broadcast(buf, outvar)
            intensors.append(buf.tensor)
        
        outbuf = self.get_or_create_buffer(outvar)
        shader_bytes = shaders.get_shader(equation.primitive.name)
        self.sequence.record_algo_data(intensors+[outbuf.tensor], shader_bytes)
    
    add = element_wise_binary_op
    sub = element_wise_binary_op
    mul = element_wise_binary_op
    div = element_wise_binary_op
    max = element_wise_binary_op
    gt  = element_wise_binary_op
    lt  = element_wise_binary_op
    eq  = element_wise_binary_op


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
        self.sequence.record_algo_data([outbuf.tensor, inbuf.tensor], shader_bytes)

    exp = element_wise_unary_op
    log = element_wise_unary_op


    
    def broadcast(self, buf:Buffer, newvar:jax.core.Var):
        #NOTE currently only supporting broadcasting where the last dimensions are equal
        #e.g:  (1,1,8,16) -> (4,5,8,16) is ok, so is () -> (4,5,8,16)
        #but not (1,1,8,1) -> (4,5,8,16)
        stripped_dims = len(buf.shape)-np.searchsorted(buf.shape, 2)
        newdims = len(newvar.aval.shape)
        assert (buf.shape[len(buf.shape)-stripped_dims:]==newvar.aval.shape[(newdims-stripped_dims):])

        #FIXME: this should not be a shader call
        outbuf = self.get_or_create_buffer(newvar)
        N = int(np.prod(buf.shape))
        shader_bytes = shaders.get_shader('broadcast', N=N)
        self.sequence.record_algo_data([outbuf.tensor, buf.tensor], shader_bytes)
        return outbuf
    
    def broadcast_in_dim(self, equation:jax.core.JaxprEqn):
        #assert broadcast_dimensions are sorted
        assert np.all(np.diff(equation.params['broadcast_dimensions'])>0)
        invar  = equation.invars[0]
        outvar = equation.outvars[0]
        assert len(invar.aval.shape)==1
        

        inbuf  = self.get_or_create_buffer(invar)
        outbuf = Buffer(inbuf.tensor, inbuf.dtype, outvar.aval.shape)
        if outbuf.shape != outvar.aval.shape:
            outbuf = self.broadcast(inbuf, outvar)
        self.buffers[outvar] = outbuf

    
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
        
        self.analyze_jaxpr(jaxpr)
        
        for eq_var, jaxpr_var in zip(equation.outvars, jaxpr.outvars):
            if str(eq_var)=='_':
                #ignored
                continue
            #connect the function-local variables ot the output variables
            self.buffers[eq_var] = self.buffers[jaxpr_var]

    def custom_jvp_call_jaxpr(self, equation:jax.core.JaxprEqn):
        assert equation.params['num_consts'] == 0
        
        jaxpr = equation.params['fun_jaxpr'].jaxpr
        #FIXME: code duplication with xla_call()
        assert len(equation.invars) == len(jaxpr.invars)
        assert len(equation.outvars) == len(jaxpr.outvars)
        for eq_var, jaxpr_var in zip(equation.invars, jaxpr.invars):
            self.buffers[jaxpr_var] = self.get_or_create_buffer(eq_var)
        
        self.analyze_jaxpr(jaxpr)
        
        for eq_var, jaxpr_var in zip(equation.outvars, jaxpr.outvars):
            self.buffers[eq_var] = self.buffers[jaxpr_var]
    

    def get_or_create_buffer(self, var:tp.Union[jax.core.Var, jax.core.Literal], initial_value:tp.Optional[np.ndarray] = None):
        if isinstance(var, jax.core.Literal):
            #literals are for some reason not hashable but have a hash
            varhash = var.hash
        else:
            varhash = var
        
        if varhash not in self.buffers:
            #kompute always uses f32
            dtype = np.float32

            #create new tensor
            if hasattr(var, 'val'):
                #literals
                initial_value = var.val
            elif initial_value is None:
                initial_value = np.empty(var.aval.shape, dtype)
            else:
                assert initial_value.shape == var.aval.shape
                if initial_value.shape == (0,):
                    #zero sized array, would cause kompute to segfault
                    #expand it, but dont update the buffer shape
                    initial_value = np.empty((1,), dtype)
                if initial_value.dtype != dtype:
                    initial_value = initial_value.astype(dtype)
            tensor = kp.Tensor( np.ravel(initial_value) )
            self.mgr.eval_tensor_create_def([tensor])
            self.sequence.record_tensor_sync_device([tensor])
            self.buffers[varhash] = Buffer(tensor, dtype, var.aval.shape)
        return self.buffers[varhash]
    

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

    def dot_general(self, equation:jax.core.JaxprEqn):
        assert equation.params['precision'] == None
        assert equation.params['dimension_numbers'] == (((1,), (0,)), ((), ()))
        assert len(equation.invars) == 2
        assert len(equation.outvars) == 1
        assert all([v.aval.dtype==np.float32 for v in equation.invars+equation.outvars])

        inbufs = [self.get_or_create_buffer(v) for v in equation.invars]
        outbuf = self.get_or_create_buffer(equation.outvars[0])
        N, C, M = outbuf.shape[0], inbufs[0].shape[1], outbuf.shape[1]

        shader_bytes = shaders.get_shader(equation.primitive, N=N, C=C, M=M)
        self.sequence.record_algo_data([b.tensor for b in [outbuf]+inbufs], shader_bytes)
    
    def iota(self, equation:jax.core.JaxprEqn):
        assert equation.params['dimension'] == 0
        outbuf = self.get_or_create_buffer(equation.outvars[0])
        shader_bytes = shaders.get_shader(equation.primitive)
        self.sequence.record_algo_data([outbuf.tensor], shader_bytes)



    def reduce_op(self, equation:jax.core.JaxprEqn):
        assert equation.params['axes']==(1,)
        outvar = equation.outvars[0]
        invar  = equation.invars[0]
        assert len(outvar.aval.shape)==1
        assert len(invar.aval.shape)==2

        inbuf  = self.get_or_create_buffer(invar)
        outbuf = self.get_or_create_buffer(outvar)

        shader_bytes = shaders.get_shader(equation.primitive.name, N=invar.aval.shape[1])
        self.sequence.record_algo_data([outbuf.tensor, inbuf.tensor], shader_bytes)

    reduce_max = reduce_op
    reduce_sum = reduce_op


    def select(self, equation:jax.core.JaxprEqn):
        assert equation.invars[0].aval.shape \
            == equation.invars[1].aval.shape \
            == equation.invars[2].aval.shape \
            == equation.outvars[0].aval.shape
        
        inbufs = [self.get_or_create_buffer(var) for var in equation.invars]
        outbuf = self.get_or_create_buffer(equation.outvars[0])

        shader_bytes = shaders.get_shader(equation.primitive.name)
        self.sequence.record_algo_data([b.tensor for b in [outbuf]+inbufs], shader_bytes)

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

        shader_bytes = shaders.get_shader(equation.primitive.name, COLS_A=cols_a, COLS_B=cols_b, COLS_OUT=cols_out)
        self.sequence.record_algo_data([b.tensor for b in [outbuf]+inbufs], shader_bytes)


    def noop(self, equation:jax.core.JaxprEqn):
        #does not perform any operations
        #simply re-uses the input buffer
        inbuf = self.get_or_create_buffer(equation.invars[0])
        self.buffers[equation.outvars[0]] = inbuf

    #currently using float32 for everything
    convert_element_type = noop
    #not relevant for us i think
    stop_gradient        = noop

