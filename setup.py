from setuptools import setup


requirements = open('requirements.txt').read().split()


setup(
    name='vkJAX',
    version='0.0.1',    
    description='JAX interpreter based on Vulkan',
    url='https://github.com/alexander-g/vkJAX',
    author='https://github.com/alexander-g',
    license='Unlicense',
    packages=['vkjax'],
    install_requires=requirements,

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: GPU',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: The Unlicense (Unlicense)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

    package_data={
        'vkjax': ['shaders/*']
    },
)