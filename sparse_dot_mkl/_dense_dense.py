from sparse_dot_mkl._mkl_interface import (MKL, _type_check, _sanity_check, _empty_output_check, _get_numpy_layout,
                                           LAYOUT_CODE_C, LAYOUT_CODE_F, _out_matrix, debug_print, _is_double, 
                                           _output_dtypes, _mkl_scalar)

import numpy as np
import ctypes as _ctypes

# Dict keyed by ('double_precision_bool', 'complex_bool')
_mkl_gemm_funcs = {(False, False): MKL._cblas_sgemm,
                   (True, False): MKL._cblas_dgemm,
                   (False, True): MKL._cblas_cgemm,
                   (True, True): MKL._cblas_zgemm}

def _dense_matmul(matrix_a, matrix_b, scalar=1., out=None, out_scalar=None):

    double_precision, complex_type = _is_double(matrix_a)

    # Reshape matrix_b to a column instead of a vector if it's 1d
    flatten_output = matrix_b.ndim == 1
    matrix_b = matrix_b.reshape(-1, 1) if flatten_output else matrix_b

    # Get dimensions
    m, n, k = matrix_a.shape[0], matrix_b.shape[1], matrix_a.shape[1]
    output_shape = (m, n)

    # Set the MKL function for precision
    func = _mkl_gemm_funcs[(double_precision, complex_type)]

    # Get the memory order for arrays
    layout_a, ld_a = _get_numpy_layout(matrix_a)
    layout_b, ld_b = _get_numpy_layout(matrix_b)

    # If they aren't the same, use the order for matrix a and have matrix b transposed
    op_b = 112 if layout_b != layout_a else 111

    # Set output array; use the memory order from matrix_a
    out_order, ld_out = ("C", output_shape[1]) if layout_a == LAYOUT_CODE_C else ("F", output_shape[0])

    # Allocate an array for outputs and set functions and types for float or doubles
    output_arr = _out_matrix(output_shape, _output_dtypes[(double_precision, complex_type)],
                             order=out_order, out_arr=out)

    output_ctype = _ctypes.c_double if double_precision else _ctypes.c_float

    # The complex functions all take void pointers
    output_ctype = _ctypes.c_void_p if complex_type else output_ctype

    # The complex versions of these functions take void pointers instead of passed structs
    # So create a C struct if necessary to be passed by reference
    scalar = _mkl_scalar(scalar, complex_type, double_precision)
    out_scalar = _mkl_scalar(out_scalar, complex_type, double_precision)

    func(layout_a,
         111,
         op_b,
         m,
         n,
         k,
         scalar if not complex_type else _ctypes.byref(scalar),
         matrix_a.ctypes.data_as(_ctypes.POINTER(output_ctype)),
         ld_a,
         matrix_b.ctypes.data_as(_ctypes.POINTER(output_ctype)),
         ld_b,
         out_scalar if not complex_type else _ctypes.byref(out_scalar),
         output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
         ld_out)

    return output_arr.ravel() if flatten_output else output_arr


def _dense_dot_dense(matrix_a, matrix_b, cast=False, scalar=1., out=None, out_scalar=None):

    _sanity_check(matrix_a, matrix_b, allow_vector=True)

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix_a, matrix_b):
        debug_print("Skipping multiplication because A (dot) B must yield an empty matrix")
        final_dtype = np.float64 if matrix_a.dtype != matrix_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return _out_matrix((matrix_a.shape[0], matrix_b.shape[1]), final_dtype, out_arr=out)

    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast)

    return _dense_matmul(matrix_a, matrix_b, scalar=scalar, out=out, out_scalar=out_scalar)
