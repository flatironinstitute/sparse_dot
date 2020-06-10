# sparse_dot_mkl
[![Build Status](https://travis-ci.org/flatironinstitute/sparse_dot.svg?branch=master)](https://travis-ci.org/flatironinstitute/sparse_dot)
[![codecov](https://codecov.io/gh/flatironinstitute/sparse_dot/branch/master/graph/badge.svg)](https://codecov.io/gh/flatironinstitute/sparse_dot)
[![PyPI version](https://badge.fury.io/py/sparse-dot-mkl.svg)](https://badge.fury.io/py/sparse-dot-mkl)
[![Conda version](https://anaconda.org/conda-forge/sparse_dot_mkl/badges/version.svg)](https://anaconda.org/conda-forge/sparse_dot_mkl)

This is a wrapper for the sparse matrix multiplication in the intel MKL library.
It is implemented entirely in native python using `ctypes`.
The main advantage to MKL (which motivated this) is multithreaded sparse matrix multiplication. 
The scipy sparse implementation is single-threaded at the time of writing (2020-01-03).
A secondary advantage is the direct multiplication of a sparse and a dense matrix without requiring any
intermediate conversion (also multithreaded). 

Two functions are explicitly available - `dot_product_mkl` and `sparse_qr_solve_mkl`: 

#### dot_product_mkl
`dot_product_mkl(matrix_a, matrix_b, cast=False, copy=True, reorder_output=False, dense=False, debug=False)`

`matrix_a` and `matrix_b` are either numpy arrays (1d or 2d) or scipy sparse matrices (CSR or CSC).
Sparse COO or BSR matrices are not supported. 
Numpy arrays must be contiguous.

This package only works with float data.
`cast=True` will convert data to double-precision floats by making an internal copy if necessary.
If A and B are both single-precision floats they will be used as is.
`cast=False` will raise a ValueError if the input arrays are not both double-precision or both single-precision.

The output will be a dense array, unless both inputs are sparse, in which case the output will be a sparse matrix.
The sparse matrix output format will be the same as the left (A) input sparse matrix.
`dense=True` will directly produce a dense array during sparse matrix multiplication. 
`dense` has no effect if a dense array would be produced anyway. 
Dense array outputs may be row-ordered or column-ordered, depending on input ordering.

`copy` is deprecated and has no effect.

`reorder_output=True` will order sparse matrix indices in the output matrix. 
It has no effect if the output is a dense array.
Input sparse matrices may be reordered without warning in place. 
This will not change data, only the way it is stored.
Scipy matrix multiplication does not produce ordered outputs, so this defaults to `False`.

#### sparse_qr_solve_mkl
`sparse_qr_solve_mkl(matrix_a, matrix_b, cast=False, debug=False)`

This is a QR solver for systems of linear equations (AX = B) where `matrix_a` is a sparse CSR matrix 
and `matrix_b` is a dense matrix.
It will return a dense array X.

`cast=True` will convert data to compatible floats by making an internal copy if necessary.
It will also convert a CSC matrix to a CSR matrix if necessary.

#### Requirements

This package requires `libmkl_rt.so` (or `libmkl_rt.dylib` for OSX, or `mkl_rt.dll` for WIN).
If the MKL library cannot be loaded an `ImportError` will be raised when the package is first imported. 
MKL is distributed with the full version of conda,
and can be installed into Miniconda with `conda install -c intel mkl`.
Alternatively, you may add need to add the path to MKL shared objects to `LD_LIBRARY_PATH`
(e.g. `export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH`).
There are some known bugs in MKL v2019 which may cause intermittent segfaults.
Updating to MKL v2020 is highly recommended if you encounter any problems.