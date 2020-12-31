import typing as tp

from .parser import HLO_Statement

import kp
import numpy as np



class KomputeInterpreter:
    def __init__(self, grouped_statements:tp.Dict[str, tp.List[HLO_Statement]]):
        self.functions = grouped_statements
        self.variables:tp.Dict[str, tp.Any] = {}

        self.mgr = kp.Manager()

    def run(self, x):
        entry_name       = get_entry_function(self.functions)
        entry_statements = self.functions[entry_name]
        
        self.sequence    = self.mgr.create_sequence("")
        self.sequence.begin()
        outputs          = self.call(entry_statements)
        self.sequence.record_tensor_sync_local(list(outputs))
        self.sequence.end()
        
        param_tensors    = [tensor for varname, tensor in self.variables.items() if varname.startswith('parameter')]
        assert len(param_tensors)==1
        x = [x] if np.shape(x)==()  else x
        param_tensors[0].set_data(x)
        param_tensors[0].map_data_from_host()
        self.mgr.eval_tensor_sync_device_def(param_tensors)

        self.sequence.eval()
        
        outputs = [o.numpy() for o in outputs]
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
            print('creating Tensor', stmt.varname, stmt.vartypeshape)
            tensor = kp.Tensor(np.ones(stmt.vartypeshape.shape, stmt.vartypeshape.dtype).ravel())
            self.mgr.eval_tensor_create_def([tensor])
            #self.sequence.record_tensor_sync_device([tensor]) #TODO?: record tensor creation to sequence
            return tensor
        return var
    
    def constant(self, stmt):
        if stmt.call_params==['false']:
            result = False
        elif stmt.call_params==['true']:
            result = True
        else:
            raise NotImplementedError(stmt)
        return result
    
    def add(self, stmt):
        dtype, shape = stmt.vartypeshape
        assert stmt.call_static_params == ''
        result = kp.Tensor( 65*np.ones( shape, dtype ).ravel() )
        self.mgr.eval_tensor_create_def([result])
        params = [ self.variables[param_name] for param_name in stmt.call_params ]
        shader_bytes = get_shader('add')
        self.sequence.record_algo_data(params+[result], shader_bytes)
        return result

    def tuple(self, stmt):
        assert stmt.call_static_params == ''
        params = [ self.variables[param_name] for param_name in stmt.call_params ]
        result = tuple(params)
        return result





def get_entry_function(functions: tp.Dict[str, tp.List[str]]):
    function_names  = functions.keys()
    entry_functions = [name for name in function_names if name.startswith('ENTRY')]
    assert len(entry_functions)==1
    return entry_functions[0]



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

def get_shader(name):
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
