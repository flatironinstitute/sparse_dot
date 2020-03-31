# sparse_dot_mkl
[![Build Status](https://travis-ci.org/flatironinstitute/sparse_dot.svg?branch=master)](https://travis-ci.org/flatironinstitute/sparse_dot)
[![codecov](https://codecov.io/gh/flatironinstitute/sparse_dot/branch/master/graph/badge.svg)](https://codecov.io/gh/flatironinstitute/sparse_dot)
[![PyPI version](https://badge.fury.io/py/sparse-dot-mkl.svg)](https://badge.fury.io/py/sparse-dot-mkl)

This is a wrapper for the sparse matrix multiplication in the intel MKL library.
It is implemented entirely in native python using `ctypes`.
The main advantage to MKL (which motivated this) is multithreaded sparse matrix multiplication. 
The scipy sparse implementation is single-threaded at the time of writing (2020-01-03).
A secondary advantage is the direct multiplication of a sparse and a dense matrix without requiring any
intermediate conversion (also multithreaded). 

The only function explicitly available is `dot_product_mkl`, which takes two matrices
`dot_product_mkl(A, B)` and returns a matrix that is `A (dot) B`. 
If both matrices are dense, the output will be dense.
If one matrix is dense and one is sparse, the output will be dense.
If both matrices are sparse, the output will be sparse unless the `dense=True` flag is passed.
The dense flag will directly multiply to a dense matrix without requiring intermediate conversion.
It has no effect if set when a dense output would normally be produced.

If both matrices are sparse, they must be of the same type (CSC or CSR).
There is no support currently for COO or BSR sparse matrices. 
Numpy (dense) arrays must be C-ordered and contiguous (these are the defaults in most situations).

This only does floating point data, and both matrices must be identical types.
If `cast=True` non-float matrices will be converted to doubles,
and a single-precision matrix will be promoted to doubles unless both matrices are single-precision. 
`cast=True` will ***not*** change data in-place, but will instead make an internal copy. 
This function may also reorder sparse data structures without warning while creating MKL's internal matrix representation
(reordering does not change data, only the way it is stored).

This package requires `libmkl_rt.so` (or `libmkl_rt.dylib` for OSX, or `libmkl_rt.dll` for WIN).
If the MKL library cannot be loaded an `ImportError` will be raised when the package is first imported. 
MKL is distributed with the full version of conda,
and can be installed into Miniconda with `conda install -c intel mkl`.
Alternatively, you may add need to add the path to MKL shared objects to `LD_LIBRARY_PATH`
(e.g. `export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH`).
There are some known bugs in MKL v2019 which may cause intermittent segfaults.
Updating to MKL v2020 is highly recommended if you encounter any problems.