from sparse_dot_mkl._mkl_interface import (MKL, _sanity_check, _empty_output_check, _type_check, _create_mkl_sparse,
                                           _destroy_mkl_handle, matrix_descr, RETURN_CODES, _convert_to_csr,
                                           _get_numpy_layout, LAYOUT_CODE_C, LAYOUT_CODE_F)
import numpy as np
import ctypes as _ctypes
import scipy.sparse as _spsparse


def _dense_sparse_matmul(matrix_a, matrix_b, scalar=1.):
    """
    Calculate BT (dot) AT and then transpose this product. It will be equal to A (dot) B

    :param matrix_a: Left (A) matrix
    :type matrix_a: np.ndarray
    :param matrix_b: Right (B) matrix
    :type matrix_b: sp.spmatrix.csr, sp.spmatrix.csc
    :param scalar: float
    :return: A (dot) B as a dense array in either column-major or row-major format
    :rtype: np.ndarray
    """

    matrix_a = matrix_a.T
    return _sparse_dense_matmul(matrix_b, matrix_a, scalar=scalar, transpose=True).T


def _sparse_dense_matmul(matrix_a, matrix_b, scalar=1., transpose=False):
    """
    Multiply together a sparse and a dense matrix
    mkl_sparse_?_mm requires the left (A) matrix to be sparse and the right (B) matrix to be dense
    If B is sparse, calculate BT (dot) AT; the transpose of this product is equal to A (dot) B
    This requires conversion of the sparse matrix to CSR format for some dense arrays;
    If A must be CSR iff B is column-major

    :param matrix_a: Left (A) matrix
    :type matrix_a: sp.spmatrix.csr, sp.spmatrix.csc
    :param matrix_b: Right (B) matrix
    :type matrix_b: np.ndarray
    :param scalar: A value to multiply the result matrix by. Defaults to 1.
    :type scalar: float
    :param transpose: Return AT (dot) B instead of A (dot) B.
    :type transpose: bool
    :return: A (dot) B as a dense array in either column-major or row-major format
    :rtype: np.ndarray
    """

    output_shape = (matrix_a.shape[1] if transpose else matrix_a.shape[0], matrix_b.shape[1])

    # Prep MKL handles and check that matrixes are compatible types
    # MKL requires the sparse matrix to be CSR if the dense matrix is column-major (F)

    mkl_a, dbl = _create_mkl_sparse(matrix_a)
    layout_b, ld_b = _get_numpy_layout(matrix_b)

    # MKL requires CSR format if the dense array is column-major
    if layout_b == LAYOUT_CODE_F and not _spsparse.isspmatrix_csr(matrix_a):
        mkl_a = _convert_to_csr(mkl_a, destroy_original=True)

    # Set functions and types for float or doubles
    output_ctype = _ctypes.c_double if dbl else _ctypes.c_float
    output_dtype = np.float64 if dbl else np.float32
    func = MKL._mkl_sparse_d_mm if dbl else MKL._mkl_sparse_s_mm

    # Allocate an output array
    output_arr = np.zeros(output_shape, dtype=output_dtype, order="C" if layout_b == LAYOUT_CODE_C else "F")
    _, output_ld = _get_numpy_layout(output_arr)

    ret_val = func(11 if transpose else 10,
                   scalar,
                   mkl_a,
                   matrix_descr(),
                   layout_b,
                   matrix_b,
                   output_shape[1],
                   ld_b,
                   1.,
                   output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
                   output_ld)

    # Check return
    if ret_val != 0:
        err_msg = "{fn} returned {v} ({e})".format(fn=func.__name__, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err_msg)

    _destroy_mkl_handle(mkl_a)

    return output_arr


def _sparse_dot_dense(matrix_a, matrix_b, cast=False, dprint=print, scalar=1.):
    """
    Multiply together a dense and a sparse matrix
    :param matrix_a:
    :param matrix_b:
    :param cast:
    :param dprint:
    :param scalar:
    :return:
    """
    _sanity_check(matrix_a, matrix_b)

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix_a, matrix_b):
        dprint("Skipping multiplication because A (dot) B must yield an empty matrix")
        final_dtype = np.float64 if matrix_a.dtype != matrix_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return np.zeros((matrix_a.shape[0], matrix_b.shape[1]), dtype=final_dtype)

    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast, dprint=dprint)

    if sum([_spsparse.isspmatrix(matrix_a), _spsparse.isspmatrix(matrix_b)]) != 1:
        raise ValueError("_sparse_dot_dense takes one sparse and one dense array")
    elif _spsparse.isspmatrix(matrix_a):
        return _sparse_dense_matmul(matrix_a, matrix_b, scalar=scalar)
    elif _spsparse.isspmatrix(matrix_b):
        return _dense_sparse_matmul(matrix_a, matrix_b, scalar=scalar)
