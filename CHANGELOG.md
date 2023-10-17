### Version 0.9.0

* Support for scipy sparse arrays (introduced in scipy 1.11)
* Additional fallback support for finding mkl DLLs with OneAPI

### Version 0.8.3

* Explicit error message when complex data is put into the QR solver
* Fix edge condition memory leak in the QR solver

### Version 0.8.2

* Corrected ImportError on windows with oneMKL by explicitly library searching for mkl_rt.{i}.dll,
  where i is between 1 and 5. That should last till 2026.

### Version 0.8.1

* `cast=True` will now cast to compatible complex floats if one array is complex and one is real

### Version 0.8.0

* Added support for complex data types
* Refactored _mkl_interface.py into a subpackage

### Version 0.7.3

* Fixed a memory leak when a CSC matrix was multiplied by a dense matrix in column-major format
* Switched continuous integration from Travis to Github Actions

### Version 0.7.2

* Corrected ImportError on windows with oneMKL by explicitly library searching for mkl_rt.1.dll
* Switched test runner from nose to pytest

### Version 0.7.1

* Fixed a bug in sparse * dense multiplication when one of the output array dimensions is 1

### Version 0.7.0

* Added support for block sparse row (BSR) format matrices
* Added `sparse_dot_mkl.set_debug_mode(True)` as a module-level debug mode flag instead of calling debug mode from 
`dot_product_mkl`. Added a number of debug messages for troubleshooting during development.

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