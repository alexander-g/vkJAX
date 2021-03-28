import typing as tp
import time
import os

from . import ops
from . import shaders
from .buffers import BufferPool, maybe_pad, view_as_float32

import kp
import numpy as np
import jax


class JaxprInterpreter:
    def __init__(self, jaxpr:jax.core.ClosedJaxpr, static_argnums:tp.Tuple[int]=(), profiling:bool=False, reuse_buffers:bool=True):
        self.jaxpr          = jaxpr
        self.static_argnums = static_argnums
        self.profiling      = profiling
        device              = int(os.environ.get('VKJAX_DEVICE', 0))
        self.mgr            = kp.Manager(device)
        self.workgroup_size = get_maximum_workgroup_size(self.mgr)
        self.bufferpool     = BufferPool(self.mgr, self.workgroup_size, reuse_buffers)
        shaders.DEFAULTS['WORKGROUP_X'] = self.workgroup_size

        self.analyze_closed_jaxpr(jaxpr)
    
    def analyze_closed_jaxpr(self, jaxpr:jax.core.ClosedJaxpr):
        '''Starts the analysis of the top level jaxpr.
           Records operations into a kp.Sequence and creates required buffers.'''
        
        assert len(jaxpr.consts) == len(jaxpr.jaxpr.constvars)
        for constvar, constval in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
            b = self.bufferpool.get_buffer(constvar)
            self.bufferpool.mark_buffer_as_constant(b, constvar, value=constval)
        
        for v in jaxpr.jaxpr.invars + jaxpr.jaxpr.outvars:
            b = self.bufferpool.get_buffer(v)
            #by setting I/O buffers constant, they are guaranteed to get an own tensor
            #thus avoiding possibly larger than needed data<->device transfers
            #(currently a limitation by kompute)
            self.bufferpool.mark_buffer_as_constant(b, v, None)
        
        input_buffers  = [self.bufferpool.get_buffer(v) for v in jaxpr.jaxpr.invars]
        self.all_ops   = ops.analyze_jaxpr(self.bufferpool, jaxpr.jaxpr)
        output_buffers = [self.bufferpool.get_buffer(v) for v in jaxpr.jaxpr.outvars]
        self.bufferpool.create_tensors()

        n_timestamps  = len(self.all_ops)+2 if self.profiling else 0
        self.sequence = self.mgr.sequence(total_timestamps=n_timestamps)

        if len(input_buffers) > 0:
            input_tensors = [b.tensor for b in input_buffers]
            self.sequence.record(kp.OpTensorSyncDevice(input_tensors))

        for op in self.all_ops:
            tensors   = [b.tensor for b in op.buffers]
            workgroup = op.workgroup or (np.prod(op.buffers[0].shape),1,1)
            workgroup = (np.ceil(workgroup[0]/self.workgroup_size).astype(int),)+workgroup[1:]
            if np.prod(workgroup)==0:
                #zero-sized buffers and thus workgroups can happen for some reason, skip
                continue
            algo      = self.mgr.algorithm(tensors, op.shader, workgroup)
            self.sequence.record(kp.OpAlgoDispatch(algo))
    
        output_tensors = [b.tensor for b in output_buffers]
        self.sequence.record(kp.OpTensorSyncLocal(output_tensors))


    def run(self, *X, return_all=False):
        '''Executes a previously recorded sequence with actual data'''
        input_tensors = [self.bufferpool.get_buffer(var).tensor for var in self.jaxpr.jaxpr.invars]
        X             = jax.tree_leaves([x for i,x in enumerate(X) if i not in self.static_argnums])
        if len(input_tensors) != len(X):
            raise TypeError(f'Expected {len(input_tensors)} input arguments, received {len(X)}')
        for input_tensor, x, var in zip(input_tensors, X, self.jaxpr.jaxpr.invars):
            #kompute always uses float32
            x = view_as_float32(np.asarray(x, dtype=var.aval.dtype))
            input_tensor.data()[:] = maybe_pad(x, pad_to=len(input_tensor))

        self.sequence.eval()
        
        output_bufs    = [self.bufferpool.get_buffer(var) for var in self.jaxpr.jaxpr.outvars]
        output_values  = [buf.numpy() for buf in output_bufs]
        output_values  = tuple(output_values)
        
        if not return_all:
            return output_values
        else:
            all_tensors = [b.tensor for b in self.bufferpool.buffers.values() if b is not None]
            self.mgr.sequence().eval(kp.OpTensorSyncLocal(all_tensors))
            all_arrays = [(var, buf.numpy() if buf is not None else None) for var,buf in self.bufferpool.buffers.items()]
            return output_values, dict(all_arrays)

    def get_profiling_info(self):
        timestamps = self.sequence.get_timestamps()
        deltas     = np.diff(timestamps)
        labels     = ['data2device']+[op.equation for op in self.all_ops]+['data2host']
        return list(zip(labels, deltas))




def get_maximum_workgroup_size(mgr:kp.Manager):
    devprops = mgr.get_device_properties()
    return min(devprops['max_work_group_invocations'], devprops['max_work_group_size'][0])
