from sparse_dot_mkl._mkl_interface import (MKL, _type_check, _sanity_check, _empty_output_check, _get_numpy_layout,
                                           LAYOUT_CODE_C, LAYOUT_CODE_F)

import numpy as np
import ctypes as _ctypes


def _dense_matmul(matrix_a, matrix_b, double_precision, scalar=1.):

    # Reshape matrix_b to a column instead of a vector if it's 1d
    flatten_output = matrix_b.ndim == 1
    matrix_b = matrix_b.reshape(-1, 1) if flatten_output else matrix_b

    # Get dimensions
    m, n, k = matrix_a.shape[0], matrix_b.shape[1], matrix_a.shape[1]
    output_shape = (m, n)

    # Set the MKL function for precision
    func = MKL._cblas_dgemm if double_precision else MKL._cblas_sgemm

    # Get the memory order for arrays
    layout_a, ld_a = _get_numpy_layout(matrix_a)
    layout_b, ld_b = _get_numpy_layout(matrix_b)

    # If they aren't the same, use the order for matrix a and have matrix b transposed
    op_b = 112 if layout_b != layout_a else 111

    # Set output array; use the memory order from matrix_a
    out_order, ld_out = ("C", output_shape[1]) if layout_a == LAYOUT_CODE_C else ("F", output_shape[0])

    # Allocate an array for outputs and set functions and types for float or doubles
    output_arr = np.zeros(output_shape, dtype=np.float64 if double_precision else np.float32, order=out_order)
    output_ctype = _ctypes.c_double if double_precision else _ctypes.c_float

    func(layout_a,
         111,
         op_b,
         m,
         n,
         k,
         scalar,
         matrix_a,
         ld_a,
         matrix_b,
         ld_b,
         1.,
         output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
         ld_out)

    return output_arr.ravel() if flatten_output else output_arr


def _dense_dot_dense(matrix_a, matrix_b, cast=False, dprint=print, scalar=1.):

    _sanity_check(matrix_a, matrix_b, allow_vector=True)

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix_a, matrix_b):
        dprint("Skipping multiplication because A (dot) B must yield an empty matrix")
        final_dtype = np.float64 if matrix_a.dtype != matrix_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return np.zeros((matrix_a.shape[0], matrix_b.shape[1]), dtype=final_dtype)

    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast, dprint=dprint)

    a_dbl, b_dbl = matrix_a.dtype == np.float64, matrix_b.dtype == np.float64

    return _dense_matmul(matrix_a, matrix_b, a_dbl or b_dbl, scalar=scalar)
