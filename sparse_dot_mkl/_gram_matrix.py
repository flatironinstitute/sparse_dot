from sparse_dot_mkl._mkl_interface import (MKL, sparse_matrix_t, RETURN_CODES, _create_mkl_sparse,
                                           _export_mkl, _order_mkl_handle, _destroy_mkl_handle, _type_check,
                                           _get_numpy_layout, _convert_to_csr, _empty_output_check, LAYOUT_CODE_C)

import scipy.sparse as _sps
import ctypes as _ctypes
import numpy as np


def _gram_matrix_sparse(matrix_a, aat=False, reorder_output=False):
    """
    Calculate the gram matrix aTa for sparse matrix and return a sparse matrix

    :param matrix_a: Sparse matrix
    :type matrix_a: scipy.sparse.csr_matrix, scipy.sparse.csc_matrix
    :param aat: Return A (dot) AT instead of AT (dot) A
    :type aat: bool
    :param reorder_output:
    :type reorder_output: bool
    :return: Sparse matrix
    :rtype: scipy.sparse.csr_matrix
    """

    sp_ref_a, double_prec = _create_mkl_sparse(matrix_a)

    if _sps.isspmatrix_csc(matrix_a):
        sp_ref_a = _convert_to_csr(sp_ref_a)

    _order_mkl_handle(sp_ref_a)

    ref_handle = sparse_matrix_t()

    ret_val = MKL._mkl_sparse_syrk(10 if aat else 11,
                                   sp_ref_a,
                                   _ctypes.byref(ref_handle))

    # Check return
    if ret_val != 0:
        raise ValueError("mkl_sparse_syrk returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))

    if reorder_output:
        _order_mkl_handle(ref_handle)

    output_arr = _export_mkl(ref_handle, double_prec, output_type="csr")

    _destroy_mkl_handle(sp_ref_a)
    _destroy_mkl_handle(ref_handle)

    return output_arr


def _gram_matrix_sparse_to_dense(matrix_a, aat=False, scalar=1.):
    """
    Calculate the gram matrix aTa for sparse matrix and return a dense matrix

    :param matrix_a: Sparse matrix
    :type matrix_a: scipy.sparse.csr_matrix, scipy.sparse.csc_matrix
    :param aat: Return A (dot) AT instead of AT (dot) A
    :type aat: bool
    :param scalar: Multiply output by a scalar value
    :type scalar: float
    :return: Dense matrix
    :rtype: numpy.ndarray
    """

    sp_ref_a, double_prec = _create_mkl_sparse(matrix_a)

    if _sps.isspmatrix_csc(matrix_a):
        sp_ref_a = _convert_to_csr(sp_ref_a, destroy_original=True)

    _order_mkl_handle(sp_ref_a)

    out_dtype = np.float64 if double_prec else np.float32
    output_ctype = _ctypes.c_double if double_prec else _ctypes.c_float
    out_dim = matrix_a.shape[0] if aat else matrix_a.shape[1]

    output_arr = np.zeros((out_dim, out_dim), dtype=out_dtype, order="C")
    _, output_ld = _get_numpy_layout(output_arr)

    if _empty_output_check(matrix_a, matrix_a):
        return output_arr

    func = MKL._mkl_sparse_d_syrkd if double_prec else MKL._mkl_sparse_s_syrkd

    ret_val = func(10 if aat else 11,
                   sp_ref_a,
                   scalar,
                   1.,
                   output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
                   LAYOUT_CODE_C,
                   output_ld)

    # Check return
    if ret_val != 0:
        err_msg = "{fn} returned {v} ({e})".format(fn=func.__name__, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err_msg)

    _destroy_mkl_handle(sp_ref_a)

    # This fixes a specific bug in mkl_sparse_d_syrkd which returns a full matrix
    # This stupid thing only happens with specific flags
    # I could probably leave it but it's pretty annoying
    if not aat:
        output_arr[np.tril_indices(output_arr.shape[0], k=-1)] = 0.

    return output_arr


def _gram_matrix_dense_to_dense(matrix_a, aat=False, scalar=1.):
    """
    Calculate the gram matrix aTa for dense matrix and return a dense matrix

    :param matrix_a: Dense matrix
    :type matrix_a: numpy.ndarray
    :param aat: Return A (dot) AT instead of AT (dot) A
    :type aat: bool
    :param scalar: Multiply output by a scalar value
    :type scalar: float
    :return: Dense matrix
    :rtype: numpy.ndarray
    """

    # Get dimensions
    n, k = matrix_a.shape if aat else matrix_a.shape[::-1]

    # Get the memory order for arrays
    layout_a, ld_a = _get_numpy_layout(matrix_a)
    double_precision = matrix_a.dtype == np.float64

    # Set the MKL function for precision
    func = MKL._cblas_dsyrk if double_precision else MKL._cblas_ssyrk
    output_ctype = _ctypes.c_double if double_precision else _ctypes.c_float

    # Allocate an array for outputs and set functions and types for float or doubles
    output_arr = np.zeros((n, n), dtype=matrix_a.dtype, order="C" if layout_a == LAYOUT_CODE_C else "F")

    func(layout_a,
         121,
         111 if aat else 112,
         n,
         k,
         scalar,
         matrix_a,
         ld_a,
         1.,
         output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
         n)

    return output_arr


def _gram_matrix(matrix, transpose=False, cast=False, dense=False, reorder_output=False, dprint=print):
    """
    Calculate a gram matrix (AT (dot) A) from a sparse matrix.

    :param matrix: Sparse matrix (CSR format is required but will convert if cast is True)
    :type matrix: scipy.sparse.csr_matrix
    :param transpose: Calculate A (dot) AT instead
    :type transpose: bool
    :param cast: Make internal copies to convert matrix to a float matrix or convert to a CSR matrix if necessary
    :type cast: bool
    :param dense: Produce a dense matrix output instead of a sparse matrix
    :type dense: bool
    :return: Gram matrix
    :rtype: scipy.sparse.csr_matrix, np.ndarray
    """

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix, matrix):
        dprint("Skipping multiplication because AT (dot) A must yield an empty matrix")
        output_shape = (matrix.shape[1], matrix.shape[1]) if transpose else (matrix.shape[0], matrix.shape[0])
        output_func = _sps.csr_matrix if _sps.isspmatrix(matrix) else np.zeros
        return output_func(output_shape, dtype=matrix.dtype)

    matrix = _type_check(matrix, cast=cast, dprint=dprint)

    if _sps.isspmatrix(matrix) and not (_sps.isspmatrix_csr(matrix) or _sps.isspmatrix_csc(matrix)):
        raise ValueError("gram_matrix requires sparse matrix to be CSR or CSC format")
    if _sps.isspmatrix_csc(matrix) and not cast:
        raise ValueError("gram_matrix cannot use a CSC matrix unless cast=True")
    elif not _sps.isspmatrix(matrix):
        return _gram_matrix_dense_to_dense(matrix, aat=transpose)
    elif dense:
        return _gram_matrix_sparse_to_dense(matrix, aat=transpose)
    else:
        return _gram_matrix_sparse(matrix, aat=transpose, reorder_output=reorder_output)


