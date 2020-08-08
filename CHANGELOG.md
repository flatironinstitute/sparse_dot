### Version 0.6.0

* Added an `out` parameter that will add a matrix multiplication to an already-allocated dense array

### Version 0.5.4

* Updated error message for a specific sparse product multiplication failure

### Version 0.5.3

* Fixed a bug that was hanging MKL threads if scikit-learn was imported prior to sparse_dot_mkl and 
mkl functions were run from a child process

### Version 0.5.2

* Fixed a bug that was not searching $LD_LIBRARY_PATH for `mkl_rt`
* Clarified several error messages

### Version 0.5.1

* Added gram_matrix_mkl to access [mkl_sparse_syrk](https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-syrk)
* Fixed a bug that was not correctly multiplying vectors for outer products

### Version 0.5.0

* Added sparse_qr_solve_mkl to access the [MKL QR solver](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-sparse-qr-solver-multifrontal-sparse-qr-factorization-method-for-solving-a-sparse.html)

### Version 0.4.1

* Added support for vector (dot) sparse matrix through [mkl_sparse_d_mv](https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-mv)

### Version 0.4.0

* Added support for sparse matrix (dot) vector through [mkl_sparse_d_mv](https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-mv)

### Version 0.3.4

* Added tests to raise python error when matrix indices overflow MKL interface type

### Version 0.3.3

* Refactored _sparse_dense.py for efficiency and readability

### Version 0.3.2

* Fixed windows library dll name

### Version 0.3.1

* Fixed incorrect return type in certain rare cases

### Version 0.3.0

* Full support for numpy arrays in row-major and column-major format