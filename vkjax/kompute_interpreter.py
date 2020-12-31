import typing as tp
from collections import namedtuple

from .parser import HLO_Statement

import kp
import numpy as np



Variable = namedtuple('Variable', ['tensor', 'dtype', 'shape'])



class KomputeInterpreter:
    def __init__(self, grouped_statements:tp.Dict[str, tp.List[HLO_Statement]]):
        self.mgr = kp.Manager()
        self.variables:tp.Dict[str, Variable] = {}
        self.functions = grouped_statements

        self.sequence    = self.mgr.create_sequence("")
        self.sequence.begin()
        entry_statements = get_entry_function(self.functions)
        output_vars      = self.call(entry_statements)
        output_tensors   = [var.tensor for var in output_vars]
        self.sequence.record_tensor_sync_local(output_tensors)
        self.sequence.end()

    def run(self, x):
        param_tensors    = [var.tensor for varname, var in self.variables.items() if varname.startswith('parameter')]
        assert len(param_tensors)==1
        x                = [x] if np.shape(x)==()  else x
        param_tensors[0].set_data(np.ravel(x))

        self.sequence.eval()
        
        root_stmt        = get_root_statement(get_entry_function(self.functions))
        outputs:tp.Tuple = self.variables[root_stmt.varname]
        outputs          = [o.tensor.numpy().reshape(o.shape) for o in outputs]
        if len(outputs)==1:
            outputs = outputs[0]
        return outputs






    def call(self, statements:tp.List[HLO_Statement]):
        for stmt in statements:
            result = self.interpret_statement(stmt)
            self.variables[stmt.varname] = result
        #last statement should be always the return value
        assert stmt.varname.startswith('ROOT')
        return result
    
    def interpret_statement(self, stmt:HLO_Statement):
        method = getattr(self, stmt.call_func, None)
        if method is None:
            raise NotImplementedError(stmt)
        return method(stmt)

    def parameter(self, stmt:HLO_Statement):
        var = self.variables.get(stmt.varname, None)
        if var is None:
            dtype, shape = stmt.vartypeshape
            print('creating Tensor', stmt.varname, dtype, shape)
            tensor = kp.Tensor(np.ones(shape, dtype).ravel())
            self.mgr.eval_tensor_create_def([tensor])
            self.sequence.record_tensor_sync_device([tensor])
            return Variable(tensor, dtype, shape)
        return var
    
    def constant(self, stmt:HLO_Statement):
        if stmt.call_params==['false']:
            result = False
        elif stmt.call_params==['true']:
            result = True
        else:
            raise NotImplementedError(stmt)
        return result
    
    def add(self, stmt:HLO_Statement):
        dtype, shape = stmt.vartypeshape
        assert stmt.call_static_params == ''
        result = kp.Tensor( np.ones( shape, dtype ).ravel() )
        self.mgr.eval_tensor_create_def([result])
        params = [ self.variables[param_name].tensor for param_name in stmt.call_params ]
        shader_bytes = get_shader('add')
        self.sequence.record_algo_data(params+[result], shader_bytes)
        return Variable(result, dtype, shape)

    def tuple(self, stmt:HLO_Statement):
        assert stmt.call_static_params == ''
        params = [ self.variables[param_name] for param_name in stmt.call_params ]
        result = tuple(params)
        return result





def get_entry_function(functions: tp.Dict[str, tp.List[HLO_Statement]]):
    entry_functions = [f for name,f in functions.items() if name.startswith('ENTRY')]
    assert len(entry_functions)==1
    return entry_functions[0]

def get_root_statement(statements: tp.List[HLO_Statement]):
    root_stmt = [stmt for stmt in statements if stmt.varname.startswith('ROOT')]
    assert len(root_stmt)==1
    return root_stmt[0]




shader_add = '''
#version 450

layout (local_size_x = 1) in;

layout(set = 0, binding = 0) buffer bina { float in_a[]; };
layout(set = 0, binding = 1) buffer binb { float in_b[]; };
layout(set = 0, binding = 2) buffer bout { float result[]; };

void main() {
    const uint index = gl_GlobalInvocationID.x;
    result[index] = in_a[index] + in_b[index];
}
'''

def get_shader(name:str):
    if name=='add':
        shader_str = shader_add
    else:
        raise NotImplementedError(name)

    import tempfile, subprocess, os
    tmpdir = tempfile.TemporaryDirectory(prefix='delete_me_')
    fname  = os.path.join(tmpdir.name,'shader.comp')
    open(fname,'w').write(shader_str)


    cmd = './glslangValidator -V '+fname
    if subprocess.Popen(cmd, shell=True).wait() != 0:
        raise RuntimeError('GLSL compilation failed for shader '+name)
    sprivname = fname.replace('shader.comp', 'comp.spv')
    sprivname = 'comp.spv'
    spirv     = open(sprivname, 'rb').read()
    return spirv
