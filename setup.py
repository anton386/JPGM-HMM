from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import cython_gsl
import numpy

ext_modules = [
    Extension("model_optimized",
              ["model_optimized.pyx"],
              extra_compile_args=["-fopenmp"],
              extra_link_args=["-fopenmp"],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=[cython_gsl.get_cython_include_dir()]),
    Extension("hmm_optimized",
              ["hmm_optimized.pyx"],
              extra_compile_args=["-fopenmp"],
              extra_link_args=["-fopenmp"],
              include_dirs=[numpy.get_include()])
]

setup(
    include_dirs = [cython_gsl.get_include()],
    ext_modules=cythonize(ext_modules)
)
