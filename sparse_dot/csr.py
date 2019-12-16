import ctypes as _ctypes
from sparse_dot import (MKL_INT, MKL_INT_NUMPY, RETURN_CODES, sparse_matrix_t,
                        _mkl_sparse_d_create_csr, _mkl_sparse_d_export_csr,
                        _mkl_sparse_s_create_csr, _mkl_sparse_s_export_csr)
from sparse_dot._common import _matmul_mkl, _destroy_mkl_handle, _check_mkl_typing
import numpy as np
import scipy.sparse as _spsparse
from numpy.ctypeslib import as_array


def csr_dot_product_mkl(csr_matrix_a, csr_matrix_b):
    """
    Multiply together two scipy CSR matrixes using the intel Math Kernel Library

    :param csr_matrix_a: Sparse matrix A in CSR format
    :type csr_matrix_a: scipy.sparse.csr_matrix
    :param csr_matrix_b: Sparse matrix B in CSR format
    :type csr_matrix_b: scipy.sparse.csr_matrix
    :return: Sparse matrix that is the result of A * B in CSR format
    :rtype: scipy.sparse.csr_matrix
    """

    mkl_double_precision = _check_mkl_typing(csr_matrix_a, csr_matrix_b)

    # Create intel MKL objects
    csr_mkl_a = _create_mkl_csr(csr_matrix_a, double_precision=mkl_double_precision)
    csr_mkl_b = _create_mkl_csr(csr_matrix_b, double_precision=mkl_double_precision)

    # Dot product
    csr_mkl_c = _matmul_mkl(csr_mkl_a, csr_mkl_b)

    # Extract
    csr_python_c = _export_csr_mkl(csr_mkl_c)

    # Destroy
    _destroy_mkl_handle(csr_mkl_c)

    return csr_python_c


def _create_mkl_csr(csr_data, double_precision=True, copy=False):
    """
    Create MKL internal representation in CSR format
    https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr

    :param csr_data: Sparse data in CSR format
    :type csr_data: scipy.sparse.csr_matrix
    :param double_precision: Use float64 if True, float32 if False
    :type double_precision: bool
    :param copy: If the dtype must be cast to a different format, make a copy if True or change the underlying data if
        False.
    :type copy: bool

    :return ref: Handle for the MKL internal representation
        Also returns a tuple of references to the data structures which MKL is using.
        This prevents the python garbage collector from deallocating them
    :rtype: sparse_matrix_t
    """

    ref = sparse_matrix_t()

    # Cast indexes to MKL_INT type
    if csr_data.indptr.dtype != MKL_INT_NUMPY:
        csr_data.indptr = csr_data.indptr.astype(MKL_INT_NUMPY)
    if csr_data.indices.dtype != MKL_INT_NUMPY:
        csr_data.indices = csr_data.indices.astype(MKL_INT_NUMPY)

    # Figure out which dtype for data
    if double_precision:
        csr_func = _mkl_sparse_d_create_csr
        csr_dtype = np.float64
    else:
        csr_func = _mkl_sparse_s_create_csr
        csr_dtype = np.float32

    # Cast data if it has to be cast
    if csr_data.data.dtype != csr_dtype and copy:
        data = csr_data.data.astype(csr_dtype)
    elif csr_data.dtype != csr_dtype:
        csr_data.data = csr_data.data.astype(csr_dtype)
        data = csr_data.data
    else:
        data = csr_data.data

    # Load into a MKL data structure and check return
    ret_val = csr_func(_ctypes.byref(ref),
                       _ctypes.c_int(0),
                       MKL_INT(csr_data.shape[0]),
                       MKL_INT(csr_data.shape[1]),
                       csr_data.indptr[0:-1],
                       csr_data.indptr[1:],
                       csr_data.indices,
                       data)

    # Check return
    if ret_val != 0:
        fname = "mkl_sparse_d_create" if double_precision else "mkl_sparse_s_create"
        err = "{fn} returned {v} ({e})".format(fn=fname, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err)

    return ref


def _export_csr_mkl(csr_mkl_handle, double_precision=True):
    """
    Export a MKL sparse handle in CSR format

    :param csr_mkl_handle: Handle for the MKL internal representation
    :type csr_mkl_handle: sparse_matrix_t
    :param double_precision: Use float64 if True, float32 if False. This MUST match the underlying float type - this
        defines a memory view, it does not cast.
    :type double_precision: bool

    :return:
    """
    indptrb = _ctypes.POINTER(MKL_INT)()
    indptren = _ctypes.POINTER(MKL_INT)()
    indices = _ctypes.POINTER(MKL_INT)()

    ordering = _ctypes.c_int()
    nrows = MKL_INT()
    ncols = MKL_INT()

    if double_precision:
        data = _ctypes.POINTER(_ctypes.c_double)()
        csr_func = _mkl_sparse_d_export_csr
        final_dtype = np.float64
    else:
        data = _ctypes.POINTER(_ctypes.c_float)()
        csr_func = _mkl_sparse_s_export_csr
        final_dtype = np.float64

    ret_val = csr_func(csr_mkl_handle,
                       _ctypes.byref(ordering),
                       _ctypes.byref(nrows),
                       _ctypes.byref(ncols),
                       _ctypes.byref(indptrb),
                       _ctypes.byref(indptren),
                       _ctypes.byref(indices),
                       _ctypes.byref(data))

    # Check return
    if ret_val != 0:
        fname = "mkl_sparse_d_export" if double_precision else "mkl_sparse_s_export"
        err = "{fn} returned {v} ({e})".format(fn=fname, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err)

    # Check ordering
    if ordering.value != 0:
        raise ValueError("1-indexing (F-style) is not supported")

    # Get matrix dims
    ncols = ncols.value
    nrows = nrows.value

    # If any axis is 0 return an empty matrix
    if nrows == 0 or ncols == 0:
        return _spsparse.csr_matrix((nrows, ncols), dtype=final_dtype)

    # Construct a numpy array from row end index pointer
    # Add 0 to first position for scipy.sparse's 3-array indexing
    indptren = as_array(indptren, shape=(nrows,))
    indptren = np.insert(indptren, 0, 0)
    nnz = indptren[-1]

    # If there are no non-zeros, return an empty matrix
    # If the number of non-zeros is insane, raise a ValueError
    if nnz == 0:
        return _spsparse.csr_matrix((nrows, ncols), dtype=final_dtype)
    elif nnz < 0 or nnz > ncols * nrows:
        raise ValueError("Matrix ({m} x {n}) is attempting to index {z} elements".format(m=nrows, n=ncols, z=nnz))

    # Construct numpy arrays from data pointer and from column indicies pointer
    data = np.array(as_array(data, shape=(nnz,)), copy=True)
    indices = np.array(as_array(indices, shape=(nnz,)), copy=True)

    # Pack and return the CSR matrix
    return _spsparse.csr_matrix((data, indices, indptren), shape=(nrows, ncols))
