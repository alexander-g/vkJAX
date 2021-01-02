import os, tempfile, subprocess

import vkjax

VKJAX_ROOT  = os.path.dirname(vkjax.__file__)
SHADERS_DIR = os.path.join(VKJAX_ROOT, 'shaders')
print(SHADERS_DIR)


def get_shader(name:str):
    shader_file = os.path.join(SHADERS_DIR, f'{name}.comp')
    if not os.path.exists(shader_file):
        raise NotImplementedError(name)
    shader_str  = open(shader_file).read()

    tmpdir = tempfile.TemporaryDirectory(prefix='delete_me_')
    fname  = os.path.join(tmpdir.name,'shader.comp')
    open(fname,'w').write(shader_str)


    sprivname = fname.replace('shader.comp', 'comp.spv')
    cmd = f'./glslangValidator -V {fname} -o {sprivname}'
    if subprocess.Popen(cmd, shell=True).wait() != 0:
        raise RuntimeError('GLSL compilation failed for shader '+name)
    spirv     = open(sprivname, 'rb').read()
    return spirv



