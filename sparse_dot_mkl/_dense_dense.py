import numpy as np
import ctypes as _ctypes
from sparse_dot_mkl._mkl_interface import MKL, _type_check, _sanity_check, _empty_output_check


def _dense_matmul(matrix_a, matrix_b, double_precision, scalar=1.):

    m, n, k = matrix_a.shape[0], matrix_b.shape[1], matrix_a.shape[1]
    output_shape = (m, n)

    # Allocate an array for outputs and set functions and types for float or doubles
    output_arr = np.zeros(output_shape, dtype=np.float64 if double_precision else np.float32)
    output_ctype = _ctypes.c_double if double_precision else _ctypes.c_float
    func = MKL._cblas_dgemm if double_precision else MKL._cblas_sgemm

    func(101,
         111,
         111,
         m,
         n,
         k,
         scalar,
         matrix_a,
         matrix_a.shape[1],
         matrix_b,
         matrix_b.shape[1],
         1.,
         output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
         output_shape[1])

    return output_arr


def _dense_dot_dense(matrix_a, matrix_b, cast=False, dprint=print, scalar=1.):

    _sanity_check(matrix_a, matrix_b)

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix_a, matrix_b):
        final_dtype = np.float64 if matrix_a.dtype != matrix_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return np.zeros((matrix_a.shape[0], matrix_b.shape[1]), dtype=final_dtype)

    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast, dprint=dprint)

    a_dbl, b_dbl = matrix_a.dtype == np.float64, matrix_b.dtype == np.float64

    return _dense_matmul(matrix_a, matrix_b, a_dbl or b_dbl, scalar=scalar)
