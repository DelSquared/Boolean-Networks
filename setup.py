from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('booleannetworks.pyx'))

#run this in cmd:
#python setup.py build_ext --inplace
