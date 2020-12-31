import typing as tp
from collections import namedtuple
#from .sequence import CommandSequence



def parse_hlo(hlo:str):
    lines     = hlo.splitlines()
    functions = group_lines_to_functions(lines)
    functions = dict([(name,process_function(function_lines)) for name,function_lines in functions.items()])
    return functions


def group_lines_to_functions(hlo_lines:tp.List[str]) -> tp.Dict[str, tp.List[str]]:
    functions = dict()
    while len(hlo_lines):
        line = hlo_lines.pop(0)
        if line.endswith('{'):
            #start of function
            funcname    = line.split(' ')[0]
            end_of_func = hlo_lines.index('}')
            funclines   = hlo_lines[:end_of_func]
            functions[funcname] = funclines
    return functions




HLO_Statement = namedtuple('HLO_Statement', ['varname', 'vartypeshape', 'call_func', 'call_params', 'call_static_params'])
TensorType    = namedtuple('TensorType', ['dtype', 'shape'])


def parse_type(type_str):
    if type_str.startswith('('):
        #tuple
        return [parse_type(x) for x in type_str[1:-1].split(', ')]
    assert not type_str.startswith('('), 'tuples not yet implemented'
    sep0 = type_str.find('[')
    dtype = type_str[:sep0]
    dtype = {'f32':'float32', 'u32':'uint32'}.get(dtype, None)
    if dtype is None and not type_str.startswith('pred'):
        raise NotImplementedError(f'TODO:{dtype}')
    sep1  = type_str.find(']')
    shape = type_str[sep0+1:sep1].split(',')
    if shape == ['']:
        #scalar
        shape = ()
    else:
        shape = tuple([int(x) for x in shape])
        #after the shape a tuple {2,1,0} is appended
        #not sure what it means, so assert it is there to stumble upon it if it changes
        assert type_str[sep1+1:] == '{'+','.join(map(str,reversed(range(len(shape)))))+'}'
    return TensorType(dtype, shape)


def parse_line(line: str):
    varname, expr = line.strip().split(' = ')
    type_is_tuple = expr.startswith('(')
    #type separator, either a space for single values or ')' for tuples
    typesep       = expr.find(' ') if not type_is_tuple else expr.find(')')+1
    vartypeshape  = expr[:typesep]
    vartypeshape  = parse_type(vartypeshape)
    newval        = expr[typesep+1:]
    call          = newval[:newval.find(')')+1]
    callfunc      = call[:call.find('(')]
    call_params   = call[len(callfunc)+1:-1].split(', ')
    call_static_params = newval[len(call)+2:]
    return HLO_Statement(varname, vartypeshape, callfunc, call_params, call_static_params)


def process_function(function_lines: tp.List[str]):
    statements = []
    for line in function_lines:
        hlo_statement = parse_line(line)
        statements.append(hlo_statement)
    return statements
