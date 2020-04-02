from sparse_dot_mkl._mkl_interface import (MKL, _sanity_check, _empty_output_check, _type_check, _create_mkl_sparse,
                                           _destroy_mkl_handle, matrix_descr, RETURN_CODES, _convert_to_csr,
                                           _get_numpy_layout, LAYOUT_CODE_C)
import numpy as np
import ctypes as _ctypes
import scipy.sparse as _spsparse


def _sparse_dense_matmul(matrix_a, matrix_b, double_precision, m, n, scalar=1., b_is_sparse=False):
    output_shape = (m, n)

    # Allocate an array for outputs and set functions and types for float or doubles
    output_ctype = _ctypes.c_double if double_precision else _ctypes.c_float
    func = MKL._mkl_sparse_d_mm if double_precision else MKL._mkl_sparse_s_mm

    # matrix_a is an mkl handle and matrix_b is dense
    if not b_is_sparse:

        layout_b, ld_b = _get_numpy_layout(matrix_b)

        output_arr = np.zeros(output_shape, dtype=np.float64 if double_precision else np.float32,
                              order="C" if layout_b == LAYOUT_CODE_C else "F")
        output_ld = output_shape[1] if layout_b == LAYOUT_CODE_C else output_shape[0]

        ret_val = func(10,
                       scalar,
                       matrix_a,
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

    # matrix_a is dense and and matrix_b is a mkl handle
    # calculate BT (dot) AT and return the transpose (which will be A dot B)
    elif b_is_sparse:

        matrix_a = matrix_a.T
        layout_a, ld_a = _get_numpy_layout(matrix_a)

        output_arr = np.zeros(output_shape, dtype=np.float64 if double_precision else np.float32,
                              order="F" if layout_a == LAYOUT_CODE_C else "C").T
        output_ld = output_shape[0] if layout_a == LAYOUT_CODE_C else output_shape[1]

        ret_val = func(11,
                       scalar,
                       matrix_b,
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
    else:
        raise ValueError("Boolean switch error in _sparse_dense_matmul")

    return output_arr


def _sparse_dot_dense(matrix_a, matrix_b, cast=False, dprint=print, scalar=1.):
    _sanity_check(matrix_a, matrix_b)

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix_a, matrix_b):
        final_dtype = np.float64 if matrix_a.dtype != matrix_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return np.zeros((matrix_a.shape[0], matrix_b.shape[1]), dtype=final_dtype)

    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast, dprint=dprint)

    if _spsparse.isspmatrix_csr(matrix_a) or _spsparse.isspmatrix_csc(matrix_a):
        mkl_a, a_dbl = _create_mkl_sparse(matrix_a)

        if _spsparse.isspmatrix_csc(matrix_a):
            mkl_a = _convert_to_csr(mkl_a)

        b_dbl = matrix_b.dtype == np.float64
        matrix_c = _sparse_dense_matmul(mkl_a, matrix_b,
                                        a_dbl or b_dbl, matrix_a.shape[0], matrix_b.shape[1],
                                        scalar=scalar, b_is_sparse=False)
        _destroy_mkl_handle(mkl_a)
        return matrix_c

    elif _spsparse.isspmatrix_csr(matrix_b) or _spsparse.isspmatrix_csc(matrix_b):
        mkl_b, b_dbl = _create_mkl_sparse(matrix_b)

        if _spsparse.isspmatrix_csc(matrix_b):
            mkl_b = _convert_to_csr(mkl_b)

        a_dbl = matrix_a.dtype == np.float64
        matrix_c = _sparse_dense_matmul(matrix_a, mkl_b,
                                        a_dbl or b_dbl, matrix_a.shape[0], matrix_b.shape[1],
                                        scalar=scalar, b_is_sparse=True)
        _destroy_mkl_handle(mkl_b)
        return matrix_c

    else:
        raise ValueError("The sparse input matrix to dot_product_mkl must be CSR or CSC")
