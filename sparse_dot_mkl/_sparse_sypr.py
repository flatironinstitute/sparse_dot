import warnings
from sparse_dot_mkl._mkl_interface import (MKL, sparse_matrix_t, _create_mkl_sparse, _get_numpy_layout, debug_timer,
                                           _export_mkl, _order_mkl_handle, _destroy_mkl_handle, _type_check,
                                           _empty_output_check, _sanity_check, _is_allowed_sparse_format,
                                           _check_return_value, matrix_descr, debug_print, _convert_to_csr,
                                           SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_FILL_MODE_LOWER,
                                           SPARSE_STAGE_FULL_MULT, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT,
                                           _get_numpy_layout, _check_return_value, LAYOUT_CODE_C, LAYOUT_CODE_F,
                                           _out_matrix, SPARSE_OPERATION_NON_TRANSPOSE, SPARSE_OPERATION_TRANSPOSE)
import ctypes as _ctypes
import numpy as np
import scipy.sparse as _spsparse
from scipy.sparse import (isspmatrix_csr as is_csr, isspmatrix_csc as is_csc, isspmatrix_bsr as is_bsr,
                          isspmatrix as is_sparse)

def _sypr_sparse_A_dense_B(matrix_a, matrix_b, transpose_a=False, out=None, out_scalar=None, a_scalar=None):

    _output_dim = 1 if transpose_a else 0
    output_shape = (matrix_b.shape[_output_dim], matrix_b.shape[_output_dim])
    layout_b, ld_b = _get_numpy_layout(matrix_b, second_arr=out)

    mkl_a, dbl, cplx = _create_mkl_sparse(matrix_a)

    # Set functions and types for float or doubles
    output_dtype = np.float64 if dbl else np.float32
    func = MKL._mkl_sparse_d_syprd if dbl else MKL._mkl_sparse_s_syprd

    # Allocate an output array
    output_arr = _out_matrix(output_shape, output_dtype, order="C" if layout_b == LAYOUT_CODE_C else "F",
                             out_arr=out, out_t=False)

    output_layout, output_ld = _get_numpy_layout(output_arr, second_arr=matrix_b)

    ret_val = func(SPARSE_OPERATION_TRANSPOSE if transpose_a else SPARSE_OPERATION_NON_TRANSPOSE,
                   mkl_a,
                   matrix_b,
                   layout_b,
                   ld_b,
                   float(out_scalar) if a_scalar is not None else 1.,
                   float(out_scalar) if out_scalar is not None else 1.,
                   output_arr,
                   output_layout,
                   output_ld)

    # Check return
    _check_return_value(ret_val, func.__name__)

    _destroy_mkl_handle(mkl_a)

    return output_arr

def _sypr_sparse_A_sparse_B(matrix_a, matrix_b, transpose_a=False):

    mkl_c = sparse_matrix_t()

    mkl_a, a_dbl, a_cplx = _create_mkl_sparse(matrix_a)
    mkl_b, b_dbl, b_cplx = _create_mkl_sparse(matrix_b)

    _order_mkl_handle(mkl_a)
    _order_mkl_handle(mkl_b)

    if is_csr(matrix_b):
        output_type = "csr"
    elif is_bsr(matrix_b):
        output_type = "bsr"
    else:
        raise ValueError("matrix B must be CSR or BSR")

    descrB = matrix_descr(sparse_matrix_type_t=SPARSE_MATRIX_TYPE_SYMMETRIC,
                          sparse_fill_mode_t=SPARSE_FILL_MODE_UPPER,
                          sparse_diag_type_t=SPARSE_DIAG_NON_UNIT)

    try:
        ret_val = MKL._mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE if transpose_a else SPARSE_OPERATION_NON_TRANSPOSE,
                                    mkl_a,
                                    mkl_b,
                                    descrB,
                                    _ctypes.byref(mkl_c),
                                    SPARSE_STAGE_FULL_MULT)

        _check_return_value(ret_val, "mkl_sparse_sypr")
    
    finally:
        _destroy_mkl_handle(mkl_a)
        _destroy_mkl_handle(mkl_b)

    # Extract
    try:
        python_c = _export_mkl(mkl_c, b_dbl, output_type=output_type)
    finally:
        _destroy_mkl_handle(mkl_c)

    return python_c


def _sparse_sypr(matrix_a, matrix_b, transpose_a=False, cast=False, out=None, out_scalar=None, scalar=None):

     # Check dtypes
    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast)

    if is_csr(matrix_b):
        default_output, output_type = _spsparse.csr_matrix, "csr"
    elif is_bsr(matrix_b):
        default_output, output_type = _spsparse.bsr_matrix, "bsr"
    elif not is_sparse(matrix_b):
        default_output, output_type = np.zeros, "dense"

    if not (is_csr(matrix_a) or is_bsr(matrix_a)) or not (is_csr(matrix_b) or is_bsr(matrix_b) or not is_sparse(matrix_b)):
        raise ValueError("Input matrices to spyr must be CSR or BSR; CSC and COO is not supported")

    # Call sypr if B is sparse
    if is_sparse(matrix_b):
        if out is not None or out_scalar is not None or scalar is not None:
            _msg = "out, out_scalar, and scalar have no effect if matrix B is not sparse"
            warnings.warn(_msg, RuntimeWarning)

        return _sypr_sparse_A_sparse_B(matrix_a, matrix_b, transpose_a=transpose_a)

    # Call syprd if B is dense
    else:
        return _sypr_sparse_A_dense_B(matrix_a, matrix_b, transpose_a=transpose_a, out=out,
                                      out_scalar=out_scalar, a_scalar=scalar)
