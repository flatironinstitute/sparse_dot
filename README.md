# sparse_dot_mkl

This is a wrapper for the sparse matrix multiplication in the intel MKL library.
It is implemented entirely in native python using `ctypes`.
The main advantage to MKL (which motivated this) is multithreaded sparse matrix multiplication. 
The scipy sparse implementation is single-threaded at the time of writing (2020-01-03).

The only function explicitly available is `dot_product_mkl`, which takes two CSR or CSC sparse matrices
`dot_product_mkl(A, B)` and produces a CSR or CSC matrix that is `A (dot) B`. 

This only does floating point data, and both matrices must be identical types.
If `cast=True` non-float matrices will be converted to doubles,
and a single-precision matrix will be promoted to doubles unless both matrices are single-precision. 
`cast=True` will change data in-place. This function may also reorder the underlying data structures
without warning while creating MKL's internal matrix representation.

This package requires `libmkl_rt.so`. This is distributed with the full version of conda,
and can be installed into Miniconda with `conda install -c intel mkl`.
Alternatively, you may add need to add the path to MKL shared objects to `LD_LIBRARY_PATH`
(e.g. `export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH`).