from sparse_dot_mkl._mkl_interface import (MKL, _sanity_check, _empty_output_check, _type_check, _create_mkl_sparse,
                                           _destroy_mkl_handle, matrix_descr, RETURN_CODES, _convert_to_csr,
                                           _get_numpy_layout, LAYOUT_CODE_C, LAYOUT_CODE_F)
import numpy as np
import ctypes as _ctypes
import scipy.sparse as _spsparse


def _sparse_dense_matmul(matrix_a, matrix_b, scalar=1.):
    """
    Multiply together a dense and a sparse matrix
    mkl_sparse_?_mm requires the left (A) matrix to be sparse and the right (B) matrix to be dense
    If B is sparse, calculate BT (dot) AT; the transpose of this product is equal to A (dot) B
    This requires conversion of the sparse matrix to CSR format for some dense arrays;
    If A is sparse, it must be CSR iff B is column-major
    If B is sparse, it must be CSR iff B is row-major

    :param matrix_a: Left (A) matrix
    :type matrix_a: np.ndarray, sp.spmatrix.csr, sp.spmatrix.csc
    :param matrix_b: Right (B) matrix
    :type matrix_b: np.ndarray, sp.spmatrix.csr, sp.spmatrix.csc
    :param scalar: A value to multiply the result matrix by. Defaults to 1.
    :param scalar: float
    :return: A (dot) B as a dense array in either column-major or row-major format
    :rtype: np.ndarray
    """
    output_shape = (matrix_a.shape[0], matrix_b.shape[1])
    b_is_sparse = _spsparse.isspmatrix(matrix_b)

    # Prep MKL handles and check that matrixes are compatible types
    # MKL requires the sparse matrix to be CSR if the dense matrix is column-major (F)
    if not b_is_sparse:
        mkl_a, dbl = _create_mkl_sparse(matrix_a)

        layout_b, ld_b = _get_numpy_layout(matrix_b)

        if layout_b == LAYOUT_CODE_F and not _spsparse.isspmatrix_csr(matrix_a):
            mkl_a = _convert_to_csr(mkl_a, destroy_original=True)
    else:
        mkl_b, dbl = _create_mkl_sparse(matrix_b)

        matrix_a = matrix_a.T
        layout_a, ld_a = _get_numpy_layout(matrix_a)

        if layout_a == LAYOUT_CODE_F and not _spsparse.isspmatrix_csr(matrix_b):
            mkl_b = _convert_to_csr(mkl_b, destroy_original=True)

    # Allocate an array for outputs and set functions and types for float or doubles
    output_ctype = _ctypes.c_double if dbl else _ctypes.c_float
    func = MKL._mkl_sparse_d_mm if dbl else MKL._mkl_sparse_s_mm

    # matrix_a is an mkl handle and matrix_b is dense
    if not b_is_sparse:

        output_arr = np.zeros(output_shape, dtype=np.float64 if dbl else np.float32,
                              order="C" if layout_b == LAYOUT_CODE_C else "F")
        output_ld = output_shape[1] if layout_b == LAYOUT_CODE_C else output_shape[0]

        ret_val = func(10,
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

    # matrix_a is dense and and matrix_b is a mkl handle
    # calculate BT (dot) AT and return the transpose (which will be A dot B)
    elif b_is_sparse:
        output_arr = np.zeros(output_shape, dtype=np.float64 if dbl else np.float32,
                              order="F" if layout_a == LAYOUT_CODE_C else "C").T
        output_ld = output_shape[0] if layout_a == LAYOUT_CODE_C else output_shape[1]

        ret_val = func(11,
                       scalar,
                       mkl_b,
                       matrix_descr(),
                       layout_a,
                       matrix_a,
                       output_shape[0],
                       ld_a,
                       1.,
                       output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
                       output_ld)

        # Check return
        if ret_val != 0:
            err_msg = "{fn} returned {v} ({e})".format(fn=func.__name__, v=ret_val, e=RETURN_CODES[ret_val])
            raise ValueError(err_msg)

        output_arr = output_arr.T
        _destroy_mkl_handle(mkl_b)

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
        final_dtype = np.float64 if matrix_a.dtype != matrix_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return np.zeros((matrix_a.shape[0], matrix_b.shape[1]), dtype=final_dtype)

    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast, dprint=dprint)

    if sum([_spsparse.isspmatrix(matrix_a), _spsparse.isspmatrix(matrix_b)]) != 1:
        raise ValueError("_sparse_dot_dense takes one sparse and one dense array")

    return _sparse_dense_matmul(matrix_a, matrix_b, scalar=scalar)
