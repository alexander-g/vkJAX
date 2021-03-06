import os

import vkjax
import pyshaderc

import subprocess

VKJAX_ROOT  = os.path.dirname(vkjax.__file__)
SHADERS_DIR = os.path.join(VKJAX_ROOT, 'shaders')

DEFAULTS = dict(
    COMMON_GLSL = open(os.path.join(VKJAX_ROOT, 'shaders', 'common.glsl')).read(),
    WORKGROUP_X = 32,
)

def get_shader(name:str, **constants):
    shader_file = os.path.join(SHADERS_DIR, f'{name}.comp')
    if not os.path.exists(shader_file):
        raise NotImplementedError(name)
    shader_str = open(shader_file).read()
    shader_str = preformat(shader_str).format(**DEFAULTS, **constants) 

    glsl_bytes = bytes(shader_str, encoding='utf8')
    try:
        spirv      = pyshaderc.compile_into_spirv(glsl_bytes, 'comp', filepath='', optimization='size')
    except:
        print_with_line_numbers(shader_str)
        raise
    return spirv



def preformat(msg):
    """ allow {{key}} to be used for formatting in text
    that already uses curly braces.  First switch this into
    something else, replace curlies with double curlies, and then
    switch back to regular braces
    """
    msg = msg.replace('{{', '<<<').replace('}}', '>>>')
    msg = msg.replace('{', '{{').replace('}', '}}')
    msg = msg.replace('<<<', '{').replace('>>>', '}')
    return msg


def print_with_line_numbers(x):
    lines = x.split('\n')
    lines = [f'{i+1:4d}|{line}' for i,line in enumerate(lines)]
    print('\n'.join(lines))
