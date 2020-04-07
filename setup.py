#!/usr/bin/env python

import numpy as np
import setuptools
import sys
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension('jsonvectorizer.jsontype', ['jsonvectorizer/jsontype.pyx']),
    Extension('jsonvectorizer.lil', ['jsonvectorizer/lil.pyx']),
    Extension('jsonvectorizer.schema', ['jsonvectorizer/schema.pyx']),
    Extension(
        'jsonvectorizer.jsonvectorizer', ['jsonvectorizer/jsonvectorizer.pyx']
    )
]
for extension in extensions:
    extension.cython_directives = {'embedsignature': True}

compiler_directives={'language_level' : sys.version_info[0]}

with open('requirements.txt') as f:
    install_requires = f.read()

setuptools.setup(
    name='jsonvectorizer',
    version='0.1.2',
    packages=setuptools.find_packages(),
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
    include_dirs=[np.get_include()],
    package_data={'jsonvectorizer': ['*.pxd']},
    install_requires=install_requires
)
