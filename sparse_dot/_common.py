import ctypes as _ctypes
import numpy as np
from sparse_dot import sparse_matrix_t, _mkl_sparse_spmm, _mkl_sparse_destroy, RETURN_CODES, NUMPY_FLOAT_DTYPES
import warnings


def _matmul_mkl(sp_ref_a, sp_ref_b):
    """
    Dot product two MKL objects together and return a handle to the result

    :param sp_ref_a: Sparse matrix A handle
    :type sp_ref_a: sparse_matrix_t
    :param sp_ref_b: Sparse matrix B handle
    :param sp_ref_b: sparse_matrix_t
    :return: Sparse matrix handle that is the dot product A * B
    :rtype: sparse_matrix_t
    """

    ref_handle = sparse_matrix_t()
    ret_val = _mkl_sparse_spmm(_ctypes.c_int(10),
                               sp_ref_a,
                               sp_ref_b,
                               _ctypes.byref(ref_handle))

    # Check return
    if ret_val != 0:
        raise ValueError("mkl_sparse_spmm returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))

    return ref_handle


def _destroy_mkl_handle(ref_handle):
    """
    Deallocate a MKL sparse handle

    :param ref_handle:
    :type ref_handle: sparse_matrix_t
    :return:
    """

    ret_val = _mkl_sparse_destroy(ref_handle)

    if ret_val != 0:
        raise ValueError("mkl_sparse_destroy returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))


def _check_mkl_typing(mat_a, mat_b):
    """
    Check the data type for sparse arrays to be multiplied.
    Return True if the data is either in float64s or should be coerced to float64s for double precision mkl
    Return False if the data is in float32 for single precision mkl

    :param mat_a: Sparse matrix A in any format
    :type mat_a: scipy.sparse.spmatrix
    :param mat_b: Sparse matrix B in any format
    :type mat_b: scipy.sparse.spmatrix
    :return: True if double precision. False if single precision.
    :rtype: bool
    """

    # Check dtypes
    if mat_a.dtype == np.float32 and mat_b.dtype == np.float32:
        mkl_double_precision = False
    else:
        mkl_double_precision = True

    # Warn if dtypes are not the same
    if mat_a.dtype != mat_b.dtype:
        warnings.warn("Matrix dtypes are not identical. All data will be coerced to float64.")

    # Warn if dtypes are not floats
    if (mat_a.dtype not in NUMPY_FLOAT_DTYPES) or (mat_b.dtype not in NUMPY_FLOAT_DTYPES):
        warnings.warn("Matrix dtypes are not float32 or float64. All data will be coerced to float64.")

    return mkl_double_precision
