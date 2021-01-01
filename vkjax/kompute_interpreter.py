import typing as tp
from collections import namedtuple

from .parser import HLO_Statement
from .       import shaders

import kp
import numpy as np



Variable = namedtuple('Variable', ['tensor', 'dtype', 'shape'])



class KomputeInterpreter:
    def __init__(self, grouped_statements:tp.Dict[str, tp.List[HLO_Statement]]):
        self.mgr         = kp.Manager()
        self.functions   = grouped_statements
        self.variables:tp.Dict[str, Variable] = {}
        self._call_stack = []

        self.sequence    = self.mgr.create_sequence("")
        self.sequence.begin()
        entry_function   = get_entry_function(self.functions)
        output_vars      = self.call(entry_function)
        output_tensors   = [var.tensor for var in output_vars]
        self.sequence.record_tensor_sync_local(output_tensors)
        self.sequence.end()

    def run(self, *X):
        entry_func       =  get_entry_function(self.functions)
        input_vars       = [self.variables[stmt.varname] for stmt in entry_func if stmt.call_func=='parameter']
        assert len(input_vars) == len(X)
        for input_var,x in zip(input_vars,X):
            input_var.tensor.set_data(np.ravel(x))

        self.sequence.eval()
        
        root_stmt        = get_root_statement(entry_func)
        outputs:tp.Tuple = self.variables[root_stmt.varname]
        outputs          = [o.tensor.numpy().reshape(o.shape) for o in outputs]
        if len(outputs)==1:
            outputs = outputs[0]
        return outputs






    def call(self, func:tp.Union[HLO_Statement, tp.List[HLO_Statement]]):
        if isinstance(func, HLO_Statement):
            assert func.call_static_params.startswith('to_apply=')
            func_name  = func.call_static_params.replace('to_apply=','')
            self._call_stack.append(func.call_params)
            func       = self.functions[func_name]
        else:
            self._call_stack.append([])
        for stmt in func:
            result = self.interpret_statement(stmt)
            self.variables[stmt.varname] = result
        self._call_stack.pop(-1)
        #last statement should be always be the return value
        assert stmt.varname.startswith('ROOT')
        return result
    
    def interpret_statement(self, stmt:HLO_Statement):
        func_name = stmt.call_func.replace('-','_')
        method = getattr(self, func_name, None)
        if method is None:
            raise NotImplementedError(stmt)
        return method(stmt)

    def parameter(self, stmt:HLO_Statement):
        if len(self._call_stack[-1])==0:
            #no parameters provided: top level call
            dtype, shape = stmt.vartypeshape
            tensor = kp.Tensor(np.ones(shape, dtype).ravel())
            self.mgr.eval_tensor_create_def([tensor])
            self.sequence.record_tensor_sync_device([tensor])
            return Variable(tensor, dtype, shape)
        else:
            #nested call
            assert len(stmt.call_params)==1
            index   = int(stmt.call_params[0])
            varname = self._call_stack[-1][index]
            return self.variables[varname]
        
    
    def constant(self, stmt:HLO_Statement):
        assert stmt.call_static_params == ''
        if stmt.call_params==['false']:
            result = False
        elif stmt.call_params==['true']:
            result = True
        elif stmt.vartypeshape.shape==():
            dtype, shape = stmt.vartypeshape
            constvalue   = np.array(stmt.call_params[0], dtype)
            tensor       = kp.Tensor(constvalue.ravel())
            self.mgr.eval_tensor_create_def([tensor])
            return Variable(tensor, dtype, shape)
        else:
            raise NotImplementedError(stmt)
        return result
    
    def add(self, stmt:HLO_Statement):
        assert stmt.call_static_params == ''
        dtype, shape = stmt.vartypeshape
        result = kp.Tensor( np.ones( shape, dtype ).ravel() )
        self.mgr.eval_tensor_create_def([result])
        params = [ self.variables[param_name].tensor for param_name in stmt.call_params ]
        shader_bytes = shaders.get_shader('add')
        self.sequence.record_algo_data(params+[result], shader_bytes)
        return Variable(result, dtype, shape)

    def tuple(self, stmt:HLO_Statement):
        assert stmt.call_static_params == ''
        params = [ self.variables[param_name] for param_name in stmt.call_params ]
        result = tuple(params)
        return result

    def broadcast(self, stmt:HLO_Statement):
        assert stmt.call_static_params == 'dimensions={}'
        params = [ self.variables[param_name].tensor for param_name in stmt.call_params ]
        assert len(params) == 1
        #FIXME: this should not be a shader call
        dtype, shape = stmt.vartypeshape
        result       = kp.Tensor( np.ones( shape, dtype ).ravel() )
        self.mgr.eval_tensor_create_def([result])
        shader_bytes = shaders.get_shader('broadcast')
        self.sequence.record_algo_data([result]+params, shader_bytes)
        return Variable(result, dtype, shape)
    
    def get_tuple_element(self, stmt:HLO_Statement):
        assert stmt.call_static_params.startswith('index=')
        index  = int(stmt.call_static_params.replace('index=',''))
        params = [ self.variables[param_name] for param_name in stmt.call_params ]
        assert len(params)==1
        tuple  = params[0]
        return tuple[index]






def get_entry_function(functions: tp.Dict[str, tp.List[HLO_Statement]]):
    entry_functions = [f for name,f in functions.items() if name.startswith('ENTRY')]
    assert len(entry_functions)==1
    return entry_functions[0]

def get_root_statement(statements: tp.List[HLO_Statement]):
    root_stmt = [stmt for stmt in statements if stmt.varname.startswith('ROOT')]
    assert len(root_stmt)==1
    return root_stmt[0]



