from sparse_dot_mkl._mkl_interface import (
    MKL,
    _get_numpy_layout,
    _type_check,
    _create_mkl_sparse,
    _destroy_mkl_handle,
    matrix_descr,
    _convert_to_csr,
    _check_return_value,
    LAYOUT_CODE_C
)

import numpy as np
import ctypes as _ctypes
import scipy.sparse as _spsparse

# Keyed by bool for double-precision
SOLVE_FUNCS = {
    True: MKL._mkl_sparse_d_qr_solve,
    False: MKL._mkl_sparse_s_qr_solve
}

# Keyed by bool for double-precision
FACTORIZE_FUNCS = {
    True: MKL._mkl_sparse_d_qr_factorize,
    False: MKL._mkl_sparse_s_qr_factorize
}

def _sparse_qr(
    matrix_a,
    matrix_b
):
    """
    Solve AX = B for X

    :param matrix_a: Sparse matrix A as CSR or CSC [M x N]
    :type matrix_a: scipy.sparse.spmatrix
    :param matrix_b: Dense matrix B [M x 1]
    :type matrix_b: numpy.ndarray
    :return: Dense matrix X [N x 1]
    :rtype: numpy.ndarray
    """

    _mkl_handles = []

    try:
        mkl_a, dbl, _ = _create_mkl_sparse(matrix_a)
        _mkl_handles.append(mkl_a)

        layout_b, ld_b = _get_numpy_layout(matrix_b)

        output_shape = matrix_a.shape[1], matrix_b.shape[1]

        # Convert a CSC matrix to CSR
        if _spsparse.isspmatrix_csc(matrix_a):
            mkl_a = _convert_to_csr(mkl_a)
            _mkl_handles.append(mkl_a)

        # QR Reorder ##
        ret_val_r = MKL._mkl_sparse_qr_reorder(mkl_a, matrix_descr())

        # Check return
        _check_return_value(ret_val_r, "mkl_sparse_qr_reorder")

        # QR Factorize ##
        factorize_func = FACTORIZE_FUNCS[dbl]

        ret_val_f = factorize_func(mkl_a, None)

        # Check return
        _check_return_value(ret_val_f, factorize_func.__name__)

        # QR Solve ##
        output_dtype = np.float64 if dbl else np.float32
        output_ctype = _ctypes.c_double if dbl else _ctypes.c_float

        output_arr = np.zeros(
            output_shape,
            dtype=output_dtype,
            order="C" if layout_b == LAYOUT_CODE_C else "F"
        )

        layout_out, ld_out = _get_numpy_layout(output_arr)

        solve_func = SOLVE_FUNCS[dbl]

        ret_val_s = solve_func(
            10,
            mkl_a,
            None,
            layout_b,
            output_shape[1],
            output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
            ld_out,
            matrix_b,
            ld_b
        )

        # Check return
        _check_return_value(ret_val_s, solve_func.__name__)

        return output_arr

    finally:
        for _handle in _mkl_handles:
            _destroy_mkl_handle(_handle)


def sparse_qr_solver(
    matrix_a,
    matrix_b,
    cast=False
):
    """
    Run the MKL QR solver for Ax=B
    and return x

    :param matrix_a: Sparse matrix A as CSR or CSC [M x N]
    :type matrix_a: scipy.sparse.spmatrix
    :param matrix_b: Dense matrix B [M x 1]
    :type matrix_b: numpy.ndarray
    :param cast: Convert data to compatible floats and
        convert CSC matrix to CSR matrix if necessary
    :raise ValueError: Raise a ValueError if the input matrices
        cannot be multiplied
    :return: Dense matrix X [N x 1]
    :rtype: numpy.ndarray
    """

    if _spsparse.isspmatrix_csc(matrix_a) and not cast:
        raise ValueError(
            "sparse_qr_solver only accepts CSR matrices if cast=False"
        )

    elif (
        not _spsparse.isspmatrix_csr(matrix_a) and
        not _spsparse.isspmatrix_csc(matrix_a)
    ):
        raise ValueError(
            "sparse_qr_solver requires matrix A to be CSR or CSC sparse matrix"
        )

    elif matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError(
            f"Bad matrix shapes for AX=B solver: "
            f"A {matrix_a.shape} & B {matrix_b.shape}"
        )

    else:
        matrix_a, matrix_b = _type_check(
            matrix_a,
            matrix_b,
            cast=cast,
            allow_complex=False
        )

        x_arr = _sparse_qr(
            matrix_a,
            matrix_b if matrix_b.ndim == 2 else matrix_b.reshape(-1, 1)
        )

        return x_arr if matrix_b.ndim == 2 else x_arr.ravel()
