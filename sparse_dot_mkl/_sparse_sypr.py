import warnings
from sparse_dot_mkl._mkl_interface import (
    MKL,
    sparse_matrix_t,
    _create_mkl_sparse,
    _get_numpy_layout,
    _export_mkl,
    _order_mkl_handle,
    _destroy_mkl_handle,
    _type_check,
    _check_return_value,
    matrix_descr,
    SPARSE_MATRIX_TYPE_SYMMETRIC,
    SPARSE_FILL_MODE_UPPER,
    SPARSE_DIAG_NON_UNIT,
    SPARSE_STAGE_FULL_MULT,
    LAYOUT_CODE_C,
    _out_matrix,
    SPARSE_OPERATION_NON_TRANSPOSE,
    SPARSE_OPERATION_TRANSPOSE,
    is_csr,
    is_bsr
)
import ctypes as _ctypes
import numpy as np
from scipy.sparse import issparse


def _sypr_sparse_A_dense_B(
    matrix_a,
    matrix_b,
    transpose_a=False,
    out=None,
    out_scalar=None,
    a_scalar=None
):
    _output_dim = 1 if transpose_a else 0
    output_shape = (matrix_b.shape[_output_dim], matrix_b.shape[_output_dim])
    layout_b, ld_b = _get_numpy_layout(matrix_b, second_arr=out)

    mkl_a, dbl, cplx = _create_mkl_sparse(matrix_a)

    # Set functions and types for float or doubles
    output_dtype = np.float64 if dbl else np.float32
    func = MKL._mkl_sparse_d_syprd if dbl else MKL._mkl_sparse_s_syprd

    # Allocate an output array
    output_arr = _out_matrix(
        output_shape,
        output_dtype,
        order="C" if layout_b == LAYOUT_CODE_C else "F",
        out_arr=out,
        out_t=False,
    )

    output_layout, output_ld = _get_numpy_layout(
        output_arr,
        second_arr=matrix_b
    )

    if transpose_a:
        t_flag = SPARSE_OPERATION_TRANSPOSE
    else:
        t_flag = SPARSE_OPERATION_NON_TRANSPOSE

    ret_val = func(
        t_flag,
        mkl_a,
        matrix_b,
        layout_b,
        ld_b,
        float(out_scalar) if a_scalar is not None else 1.0,
        float(out_scalar) if out_scalar is not None else 1.0,
        output_arr,
        output_layout,
        output_ld,
    )

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

    descrB = matrix_descr(
        sparse_matrix_type_t=SPARSE_MATRIX_TYPE_SYMMETRIC,
        sparse_fill_mode_t=SPARSE_FILL_MODE_UPPER,
        sparse_diag_type_t=SPARSE_DIAG_NON_UNIT,
    )

    try:
        ret_val = MKL._mkl_sparse_sypr(
            SPARSE_OPERATION_TRANSPOSE
            if transpose_a
            else SPARSE_OPERATION_NON_TRANSPOSE,
            mkl_a,
            mkl_b,
            descrB,
            _ctypes.byref(mkl_c),
            SPARSE_STAGE_FULL_MULT,
        )

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


def _sparse_sypr(
    matrix_a,
    matrix_b,
    transpose_a=False,
    cast=False,
    out=None,
    out_scalar=None,
    scalar=None,
):
    # Check dtypes
    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast)

    if (
        not (is_csr(matrix_a) or is_bsr(matrix_a)) or
        not (is_csr(matrix_b) or is_bsr(matrix_b) or not issparse(matrix_b))
    ):
        raise ValueError(
            "Input matrices to spyr must be CSR or BSR; "
            "CSC and COO are not supported"
        )

    # Call sypr if B is sparse
    if issparse(matrix_b):
        if out is not None or out_scalar is not None or scalar is not None:
            warnings.warn(
                "out, out_scalar, and scalar have no effect if matrix B "
                "is not sparse",
                RuntimeWarning
            )

        return _sypr_sparse_A_sparse_B(
            matrix_a,
            matrix_b,
            transpose_a=transpose_a
        )

    # Call syprd if B is dense
    else:
        return _sypr_sparse_A_dense_B(
            matrix_a,
            matrix_b,
            transpose_a=transpose_a,
            out=out,
            out_scalar=out_scalar,
            a_scalar=scalar,
        )
