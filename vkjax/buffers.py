import typing as tp
import numpy as np

import kp
import jax


class TensorPlaceholder:
    def __init__(self, t=None):
        self.t = t


class Buffer:
    def __init__(self, tensor:kp.Tensor, dtype:np.dtype, shape:tp.Tuple[int]):
        if isinstance(tensor, TensorPlaceholder):
            super().__setattr__('tensor', tensor)
        else:
            super().__setattr__('tensor', TensorPlaceholder(tensor))
        self.dtype  = dtype
        self.shape  = shape
        self.accesses:tp.List[int] = []

    def numpy(self):
        array = self.tensor.data()
        array  = view_or_convert_from_32bit(array, self.dtype)
        #the tensor might be larger than the shape it represents
        n     = int(np.prod(self.shape))
        array = array[:n].reshape(self.shape)
        return array
    
    def nbytes(self):
        assert self.dtype.type in {np.bool_, np.float32, np.int32, np.uint32}, NotImplementedError(self.dtype)
        #NOTE: booleans are also 32-bit in glsl
        return int(np.prod(self.shape))*4
    
    def __getattribute__(self, name):
        if name=='tensor':
            return super().__getattribute__(name).t
        else:
            return super().__getattribute__(name)
    
    def __setattr__(self, name, value):
        if name=='tensor':
            super().__getattribute__(name).t = value
        else:
            super().__setattr__(name, value)

    def view(self, new_dtype, new_shape):
        new_b          = Buffer(super().__getattribute__('tensor'), new_dtype, new_shape)
        new_b.accesses = self.accesses
        return new_b


def view_or_convert_from_32bit(x, dtype):
    if dtype == np.bool:
        #GLSL booleans are 32-bit, cannot simply .view()
        return x>0
    else:
        return x.view(dtype)

def maybe_pad(x, pad_to=1):
    x          = np.ravel(x)
    remainder  = x.size % pad_to
    if remainder != 0 or x.size==0:
        x = np.pad(x, (0,pad_to-remainder))
    return x

def view_as_float32(x):
    assert x.dtype.type in {np.bool_, np.float32, np.int32, np.int64, np.uint32, np.float64}, NotImplementedError(x.dtype)
    if x.dtype==np.bool:
        #glsl booleans are 32-bit
        x = x.astype('uint32')
    if x.dtype.type in {np.float64}:
        x = x.astype('float32')
    if x.dtype.type in {np.int64}:
        x = x.astype('int32')
    return x.view('float32')


def hashable(var:tp.Union[jax.core.Var, jax.core.Literal]):
    '''Returns a value that can be hashed. Specifically for jax.core.Literal'''
    if isinstance(var, jax.core.Literal):
        #literals are for some reason not hashable but have a hash
        return var.hash
    else:
        return var



class BufferPool:
    def __init__(self, mgr:kp.Manager, workgroup_size:int, reuse_tensors=True):
        self.mgr = mgr
        self.workgroup_size = workgroup_size
        #dict mapping from jax.core.Var or int (for jax.core.Literal) to Buffer
        self.buffers = dict()
        self.op_counter = 0
        self.reuse_tensors = reuse_tensors

    def get_buffer(self, var:jax.core.Var, increment_op_counter:bool=False):
        varhash = hashable(var)
        if varhash not in self.buffers:
            if isinstance(var.aval, jax.core.AbstractUnit):
                return None
            #create buffer without tensor (except for literals)
            #tensor will be created after all buffers are collected
            self.buffers[varhash] = Buffer(None, var.aval.dtype, var.aval.shape)   
        b = self.buffers[varhash]
        
        #remember that this buffer gets accessed in this op
        b.accesses.append(self.op_counter)

        if hasattr(var, 'val') and b.tensor is None:
            #literals
            self.mark_buffer_as_constant(b, var, var.val)

        if increment_op_counter:
            self.op_counter += 1
        return b
    
    def mark_buffer_as_constant(self, b:Buffer, var:jax.core.Var, value:np.ndarray):
        if b.tensor is None:
            #new constant buffer, create tensor and set value
            b.tensor = self._create_tensor(var, initial_value=value)
        varhash  = hashable(var)
        #the tensor of a constant must remain unchanged
        #marking it as being used from start to end
        b.accesses += [0, np.inf]

    def set_buffer(self, var:jax.core.Var, b:Buffer):
        if b is None:
            return
        varhash = hashable(var)
        self.buffers[varhash] = b

        #remember that this buffer gets accessed in this op
        b.accesses.append(self.op_counter)

    def create_tensors(self):
        tensor_access_map:tp.Dict[kp.Tensor, tp.List[int]] = dict()

        buffers = list(self.buffers.values())
        jaxvals = list(self.buffers.keys())

        #sort by size
        indices = sorted(range(len(buffers)), key=lambda i: buffers[i].nbytes())
        #start with the largest (easiest to be re-used)
        for i in reversed(indices):
            b = buffers[i]
            v = jaxvals[i]
            b_accesses = b.accesses

            if b.tensor is not None and max(b_accesses)==np.inf:
                #constant, skip
                continue

            t = None
            if self.reuse_tensors:
                #check if there are already created tensors that can be re-used
                t = self._search_for_free_tensor(b, tensor_access_map)
            
            if t is None:
                #no match, create new tensor
                t = self._create_tensor(v)
                tensor_access_map[t] = b_accesses

            self.buffers[v].tensor = t
    
    def _search_for_free_tensor(self, b:Buffer, tensor_access_map:tp.Dict[kp.Tensor, tp.List[int]]):
        for other_v, other_b in self.buffers.items():
            t = other_b.tensor
            if t is None:
                continue

            if isinstance(other_v, int) or max(other_b.accesses)==np.inf:
                #literal
                continue
            
             #must be at least as large
            if t.data().nbytes < b.nbytes():
                continue
            
            #must not be accessed inbetween this buffer's accesses
            t_accesses = tensor_access_map[t]
            if max(b.accesses) < min(t_accesses) or min(b.accesses) > max(t_accesses):
                #re-use tensor
                tensor_access_map[t] += b.accesses
                return t

    def _create_tensor(self, var:jax.core.Var, initial_value:np.ndarray = None):
        if hasattr(var, 'val'):
            #literals
            initial_value = var.val
        elif initial_value is None:
            initial_value = np.zeros(var.aval.shape, var.aval.dtype)
        else:
            assert initial_value.size == np.prod(var.aval.shape), (initial_value.shape, var.aval.shape)

        initial_value = maybe_pad(initial_value, pad_to=self.workgroup_size)
        #kompute currently only supports float32 tensors
        initial_value = view_as_float32(initial_value)

        t = self.mgr.tensor(initial_value)
        self.mgr.sequence().eval(kp.OpTensorSyncDevice([t]))
        return t



