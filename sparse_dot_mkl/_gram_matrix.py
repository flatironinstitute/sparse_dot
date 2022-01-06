from sparse_dot_mkl._mkl_interface import (MKL, sparse_matrix_t, _create_mkl_sparse,
                                           _export_mkl, _order_mkl_handle, _destroy_mkl_handle, _type_check,
                                           _get_numpy_layout, _convert_to_csr, _empty_output_check,
                                           _out_matrix, _check_return_value, debug_print, _output_dtypes, _mkl_scalar,
                                           _is_double)
                                           
from sparse_dot_mkl._mkl_interface._constants import *

import scipy.sparse as _sps
import ctypes as _ctypes
import numpy as np

# Dict keyed by ('transpose_bool', 'complex_bool')
_mkl_sp_transpose_ops = {(False, False): SPARSE_OPERATION_NON_TRANSPOSE,
                         (True, False): SPARSE_OPERATION_TRANSPOSE,
                         (False, True): SPARSE_OPERATION_NON_TRANSPOSE,
                         (True, True): SPARSE_OPERATION_CONJUGATE_TRANSPOSE}

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

    sp_ref_a, double_prec, complex_type = _create_mkl_sparse(matrix_a)

    if _sps.isspmatrix_csc(matrix_a):
        sp_ref_a = _convert_to_csr(sp_ref_a)

    _order_mkl_handle(sp_ref_a)

    ref_handle = sparse_matrix_t()

    ret_val = MKL._mkl_sparse_syrk(_mkl_sp_transpose_ops[(not aat, complex_type)],
                                   sp_ref_a,
                                   _ctypes.byref(ref_handle))

    # Check return
    _check_return_value(ret_val, "mkl_sparse_syrk")

    if reorder_output:
        _order_mkl_handle(ref_handle)

    output_arr = _export_mkl(ref_handle, double_prec, complex_type=complex_type, output_type="csr")

    _destroy_mkl_handle(sp_ref_a)
    _destroy_mkl_handle(ref_handle)

    return output_arr

# Dict keyed by ('double_precision_bool', 'complex_bool')
_mkl_skryd_funcs = {(False, False): MKL._mkl_sparse_s_syrkd,
                    (True, False): MKL._mkl_sparse_d_syrkd,
                    (False, True): MKL._mkl_sparse_c_syrkd,
                    (True, True): MKL._mkl_sparse_z_syrkd}

def _gram_matrix_sparse_to_dense(matrix_a, aat=False, scalar=1., out=None, out_scalar=None):
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

    sp_ref_a, double_prec, complex_type = _create_mkl_sparse(matrix_a)

    if _sps.isspmatrix_csc(matrix_a):
        sp_ref_a = _convert_to_csr(sp_ref_a, destroy_original=True)

    _order_mkl_handle(sp_ref_a)

    out_dtype = _output_dtypes[(double_prec, complex_type)]
    func = _mkl_skryd_funcs[(double_prec, complex_type)]

    out_dim = matrix_a.shape[0 if aat else 1]

    output_arr = _out_matrix((out_dim, out_dim), out_dtype, order="C", out_arr=out)
    _, output_ld = _get_numpy_layout(output_arr)

    if _empty_output_check(matrix_a, matrix_a):
        _destroy_mkl_handle(sp_ref_a)
        return output_arr

    scalar = _mkl_scalar(scalar, complex_type, double_prec)
    out_scalar = _mkl_scalar(out_scalar, complex_type, double_prec)

    ret_val = func(_mkl_sp_transpose_ops[(not aat, complex_type)],
                   sp_ref_a,
                   scalar,
                   out_scalar,
                   output_arr,
                   LAYOUT_CODE_C,
                   output_ld)

    # Check return
    _check_return_value(ret_val, func.__name__)

    _destroy_mkl_handle(sp_ref_a)

    # This fixes a specific bug in mkl_sparse_d_syrkd which returns a full matrix
    # This stupid thing only happens with specific flags
    # I could probably leave it but it's pretty annoying
    
    if not aat and out is None and not complex_type:
        output_arr[np.tril_indices(output_arr.shape[0], k=-1)] = 0.

    return output_arr

# Dict keyed by ('double_precision_bool', 'complex_bool')
_mkl_blas_skry_funcs = {(False, False): MKL._cblas_ssyrk,
                        (True, False): MKL._cblas_dsyrk,
                        (False, True): MKL._cblas_csyrk,
                        (True, True): MKL._cblas_zsyrk}

# Dict keyed by ('transpose_bool', 'complex_bool')
_mkl_cblas_transpose_ops = {(False, False): CBLAS_NO_TRANS,
                            (True, False): CBLAS_TRANS,
                            (False, True): CBLAS_NO_TRANS,
                            (True, True): CBLAS_CONJ_TRANS}

def _gram_matrix_dense_to_dense(matrix_a, aat=False, scalar=1., out=None, out_scalar=None):
    """
    Calculate the gram matrix aTa for dense matrix and return a dense matrix

    :param matrix_a: Dense matrix
    :type matrix_a: numpy.ndarray
    :param aat: Return A (dot) AT instead of AT (dot) A
    :type aat: bool
    :param scalar: Multiply output by a scalar value
    :type scalar: float
    :param out: Add the dot product to this array if provided.
    :type out: np.ndarray, None
    :param out_scalar: Multiply the out array by this scalar if provided.
    :type out_scalar: float, None
    :return: Dense matrix
    :rtype: numpy.ndarray
    """

    # Get dimensions
    n, k = matrix_a.shape if aat else matrix_a.shape[::-1]

    # Get the memory order for arrays
    layout_a, ld_a = _get_numpy_layout(matrix_a)
    double_precision, complex_type = _is_double(matrix_a)

    out_dtype = _output_dtypes[(double_precision, complex_type)]
    func = _mkl_blas_skry_funcs[(double_precision, complex_type)]

    # Allocate an array for outputs and set functions and types for float or doubles
    output_arr = _out_matrix((n, n), out_dtype, order="C" if layout_a == LAYOUT_CODE_C else "F", out_arr=out)

    # The complex versions of these functions take void pointers instead of passed structs
    # So create a C struct if necessary to be passed by reference
    scalar = _mkl_scalar(scalar, complex_type, double_precision)
    out_scalar = _mkl_scalar(out_scalar, complex_type, double_precision)

    func(layout_a,
         MKL_UPPER,
         _mkl_cblas_transpose_ops[(not aat, complex_type)],
         n,
         k,
         scalar if not complex_type else _ctypes.byref(scalar),
         matrix_a,
         ld_a,
         out_scalar if not complex_type else _ctypes.byref(scalar),
         output_arr,
         n)

    return output_arr


def _gram_matrix(matrix, transpose=False, cast=False, dense=False, reorder_output=False, out=None, out_scalar=None):
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
    :param out: Add the dot product to this array if provided.
    :type out: np.ndarray, None
    :param out_scalar: Multiply the out array by this scalar if provided.
    :type out_scalar: float, None
    :return: Gram matrix
    :rtype: scipy.sparse.csr_matrix, np.ndarray
    """

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix, matrix):
        debug_print("Skipping multiplication because AT (dot) A must yield an empty matrix")
        output_shape = (matrix.shape[1], matrix.shape[1]) if transpose else (matrix.shape[0], matrix.shape[0])
        output_func = _sps.csr_matrix if _sps.isspmatrix(matrix) else np.zeros
        return output_func(output_shape, dtype=matrix.dtype)

    if np.iscomplexobj(matrix):
        raise ValueError("gram_matrix_mkl does not support complex datatypes")

    matrix = _type_check(matrix, cast=cast)

    if _sps.isspmatrix(matrix) and not (_sps.isspmatrix_csr(matrix) or _sps.isspmatrix_csc(matrix)):
        raise ValueError("gram_matrix requires sparse matrix to be CSR or CSC format")
    elif _sps.isspmatrix_csc(matrix) and not cast:
        raise ValueError("gram_matrix cannot use a CSC matrix unless cast=True")
    elif not _sps.isspmatrix(matrix):
        return _gram_matrix_dense_to_dense(matrix, aat=transpose, out=out, out_scalar=out_scalar)
    elif dense:
        return _gram_matrix_sparse_to_dense(matrix, aat=transpose,  out=out, out_scalar=out_scalar)
    elif out is not None:
        raise ValueError("out argument cannot be used with sparse (dot) sparse matrix multiplication")
    else:
        return _gram_matrix_sparse(matrix, aat=transpose, reorder_output=reorder_output)


