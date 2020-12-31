import vkjax.parser as parser
import jax




def test_parse_hlo_basic():
    f = lambda x: x+x
    hlo = jax.xla_computation(f)(0.0).as_hlo_text()
    functions = parser.parse_hlo(hlo)
    assert len(functions)==1

def test_parse_type():
    token  = 'f32[]'
    result  = parser.parse_type(token)
    assert result.dtype == 'float32'
    assert result.shape == ()

    token  = 'u32[10]{0}'
    result  = parser.parse_type(token)
    assert result.dtype == 'uint32'
    assert result.shape == (10,)

    token  = 'f32[32,10]{1,0}'
    result  = parser.parse_type(token)
    assert result.dtype == 'float32'
    assert result.shape == (32,10)

    token  = 'pred[]'
    result  = parser.parse_type(token)
    assert result.dtype == None
    assert result.shape == ()



def test_parse_line():
    line = '  parameter.36 = f32[128,32]{1,0} parameter(3)'
    stmt = parser.parse_line(line)
    assert stmt.varname            == 'parameter.36'
    assert stmt.vartypeshape.dtype == 'float32'
    assert stmt.vartypeshape.shape == (128,32)
    assert stmt.call_func          == 'parameter'
    assert stmt.call_params        == ['3']
    assert stmt.call_static_params == ''

    line = '  ROOT tuple.49 = (f32[128,10]{1,0}, u32[2]{0}, f32[10]{0}, f32[32,10]{1,0}) tuple(get-tuple-element.45, get-tuple-element.46, get-tuple-element.47, get-tuple-element.48)'
    stmt = parser.parse_line(line)
    assert stmt.varname            == 'ROOT tuple.49'
    assert len(stmt.vartypeshape)  == 4
    assert stmt.call_func          == 'tuple'
    assert stmt.call_params        == ['get-tuple-element.45', 'get-tuple-element.46', 'get-tuple-element.47', 'get-tuple-element.48']
    assert stmt.call_static_params == ''

    line = '  dot.11 = f32[128,10]{1,0} dot(parameter.7, parameter.8), lhs_contracting_dims={1}, rhs_contracting_dims={0}'
    stmt = parser.parse_line(line)
    assert stmt.varname            == 'dot.11'
    assert stmt.call_func          == 'dot'
    assert stmt.call_params        == ['parameter.7', 'parameter.8']
    assert stmt.call_static_params == 'lhs_contracting_dims={1}, rhs_contracting_dims={0}'



