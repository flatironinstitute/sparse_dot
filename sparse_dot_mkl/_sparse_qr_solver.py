from sparse_dot_mkl._mkl_interface import (MKL, _sanity_check, _get_numpy_layout, _type_check, _create_mkl_sparse,
                                           _destroy_mkl_handle, matrix_descr, RETURN_CODES, _convert_to_csr,
                                           LAYOUT_CODE_C)

import numpy as np
import ctypes as _ctypes
import scipy.sparse as _spsparse


def _sparse_qr(matrix_a, matrix_b):
    """
    Solve AX = B for X

    :param matrix_a: Sparse matrix A
    :type matrix_a: scipy.sparse.csr_matrix
    :param matrix_b: Dense matrix B
    :type matrix_b: numpy.ndarray
    :return: Dense matrix X
    :rtype: numpy.ndarray
    """

    mkl_a, dbl = _create_mkl_sparse(matrix_a)
    layout_b, ld_b = _get_numpy_layout(matrix_b)

    output_shape = matrix_a.shape[1], matrix_b.shape[1]

    if _spsparse.isspmatrix_csc(matrix_a):
        mkl_a = _convert_to_csr(mkl_a)

    # QR Reorder ##
    ret_val_r = MKL._mkl_sparse_qr_reorder(mkl_a, matrix_descr())

    # Check return
    if ret_val_r != 0:
        err_msg = "mkl_sparse_qr_reorder returned {v} ({e})".format(v=ret_val_r, e=RETURN_CODES[ret_val_r])
        raise ValueError(err_msg)

    # QR Factorize ##
    factorize_func = MKL._mkl_sparse_d_qr_factorize if dbl else MKL._mkl_sparse_s_qr_factorize

    ret_val_f = factorize_func(mkl_a, None)

    # Check return
    if ret_val_f != 0:
        err_msg = "{fn} returned {v} ({e})".format(fn=factorize_func.__name__, v=ret_val_f, e=RETURN_CODES[ret_val_f])
        raise ValueError(err_msg)

    # QR Solve ##
    output_dtype = np.float64 if dbl else np.float32
    output_ctype = _ctypes.c_double if dbl else _ctypes.c_float

    output_arr = np.zeros(output_shape, dtype=output_dtype, order="C" if layout_b == LAYOUT_CODE_C else "F")
    layout_out, ld_out = _get_numpy_layout(output_arr)

    solve_func = MKL._mkl_sparse_d_qr_solve if dbl else MKL._mkl_sparse_s_qr_solve

    ret_val_s = solve_func(10,
                           mkl_a,
                           None,
                           layout_b,
                           output_shape[1],
                           output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
                           ld_out,
                           matrix_b,
                           ld_b)

    # Check return
    if ret_val_s != 0:
        err_msg = "{fn} returned {v} ({e})".format(fn=solve_func.__name__, v=ret_val_s, e=RETURN_CODES[ret_val_s])
        raise ValueError(err_msg)

    _destroy_mkl_handle(mkl_a)

    return output_arr


def sparse_qr_solver(matrix_a, matrix_b, cast=False, dprint=print):
    """

    :param matrix_a:
    :param matrix_b:
    :param cast:
    :param dprint:
    :return:
    """

    if _spsparse.isspmatrix_csc(matrix_a) and not cast:
        raise ValueError("sparse_qr_solver only accepts CSR matrices if cast=False")
    elif not _spsparse.isspmatrix_csr(matrix_a) and not _spsparse.isspmatrix_csc(matrix_a):
        raise ValueError("sparse_qr_solver requires matrix A to be CSR or CSC sparse matrix")
    elif matrix_a.shape[0] != matrix_b.shape[0]:
        err_msg = "Bad matrix shapes for AX=B solver: A {sha} & B {shb}".format(sha=matrix_a.shape, shb=matrix_b.shape)
        raise ValueError(err_msg)
    else:
        matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast, dprint=dprint)
        x_arr = _sparse_qr(matrix_a, matrix_b if matrix_b.ndim == 2 else matrix_b.reshape(-1, 1))
        return x_arr if matrix_b.ndim == 2 else x_arr.ravel()
