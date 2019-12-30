# sparse_dot

This is a wrapper for the sparse matrix multiplication in the intel MKL library. 
The main advantage to MKL (which motivated this) is multithreaded sparse matrix multiplication. 
The scipy sparse implementation is single-threaded.

The only function explicitly available is `dot_product_mkl`, which takes two CSR or CSC sparse matrices
`dot_product_mkl(A, B)` and produces a CSR or CSC matrix that is `A (dot) B`. 

This only does floating point data, and both matrices must be identical types.
If `cast=True` non-float matrices will be converted to doubles,
and a single-precision matrix will be promoted to doubles unless both matrices are single-precision. 
`cast` will change data in-place. This function may also reorder the underlying data structures
without warning while creating MKL's internal matrix representation.