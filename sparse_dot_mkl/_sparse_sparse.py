from sparse_dot_mkl._mkl_interface import (MKL, sparse_matrix_t, _create_mkl_sparse, debug_print, debug_timer,
                                           _export_mkl, _order_mkl_handle, _destroy_mkl_handle, _type_check,
                                           _empty_output_check, _sanity_check, _is_allowed_sparse_format,
                                           _check_return_value)
import ctypes as _ctypes
import numpy as np
import scipy.sparse as _spsparse
from scipy.sparse import isspmatrix_csr as is_csr, isspmatrix_csc as is_csc, isspmatrix_bsr as is_bsr


def _matmul_mkl(sp_ref_a, sp_ref_b):
    """
    Dot product two MKL objects together and return a handle to the result

    :param sp_ref_a: Sparse matrix A handle
    :type sp_ref_a: sparse_matrix_t
    :param sp_ref_b: Sparse matrix B handle
    :param sp_ref_b: sparse_matrix_t
    :return: Sparse matrix handle that is the dot product A * B
    :rtype: sparse_matrix_t
    """

    ref_handle = sparse_matrix_t()

    ret_val = MKL._mkl_sparse_spmm(_ctypes.c_int(10),
                                   sp_ref_a,
                                   sp_ref_b,
                                   _ctypes.byref(ref_handle))

    # Check return
    _check_return_value(ret_val, "mkl_sparse_spmm")
    return ref_handle


def _matmul_mkl_dense(sp_ref_a, sp_ref_b, output_shape, double_precision):
    """
    Dot product two MKL objects together into a dense numpy array and return the result

    :param sp_ref_a: Sparse matrix A handle
    :type sp_ref_a: sparse_matrix_t
    :param sp_ref_b: Sparse matrix B handle
    :type sp_ref_b: sparse_matrix_t
    :param output_shape: The shape of the output array
    This must be correct or the preallocated numpy array won't be correct and this will probably segfault
    :type output_shape: tuple(int, int)
    :param double_precision: The resulting array will be float64
    :type double_precision: bool

    :return: Dense numpy array that's the output of A dot B
    :rtype: np.array
    """

    # Allocate an array for outputs and set functions and types for float or doubles
    output_arr = np.zeros(output_shape, dtype=np.float64 if double_precision else np.float32)
    output_ctype = _ctypes.c_double if double_precision else _ctypes.c_float
    func = MKL._mkl_sparse_d_spmmd if double_precision else MKL._mkl_sparse_s_spmmd

    ret_val = func(10,
                   sp_ref_a,
                   sp_ref_b,
                   101,
                   output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
                   output_shape[1])

    # Check return
    _check_return_value(ret_val, func.__name__)

    return output_arr


def _sparse_dot_sparse(matrix_a, matrix_b, cast=False, reorder_output=False, dense=False):
    """
    Multiply together two scipy sparse matrixes using the intel Math Kernel Library.
    This currently only supports float32 and float64 data

    :param matrix_a: Sparse matrix A in CSC/CSR format
    :type matrix_a: scipy.sparse.spmatrix
    :param matrix_b: Sparse matrix B in CSC/CSR format
    :type matrix_b: scipy.sparse.spmatrix
    :param cast: Should the data be coerced into float64 if it isn't float32 or float64
    If set to True and any other dtype is passed, the matrix data will be modified in-place
    If set to False and any dtype that isn't float32 or float64 is passed, a ValueError will be raised
    Defaults to False
    :param reorder_output: Should the array indices be reordered using MKL
    If set to True, the object in C will be ordered and then exported into python
    If set to False, the array column indices will not be ordered.
    The scipy sparse dot product does not yield ordered column indices so this defaults to False
    :type reorder_output: bool
    :param dense: Should the matrix multiplication yield a dense numpy array
    This does not require any copy and is memory efficient if the output array density is > 50%
    :type dense: bool
    :return: Sparse matrix that is the result of A * B in CSR format
    :rtype: scipy.sparse.csr_matrix
    """

    # Check for allowed sparse matrix types
    if not _is_allowed_sparse_format(matrix_a) or not _is_allowed_sparse_format(matrix_b):
        raise ValueError("Input matrices to dot_product_mkl must be CSR, CSC, or BSR; COO is not supported")

    if is_csr(matrix_a):
        default_output, output_type = _spsparse.csr_matrix, "csr"
    elif is_csc(matrix_a):
        default_output, output_type = _spsparse.csc_matrix, "csc"
    elif is_bsr(matrix_a):
        default_output, output_type = _spsparse.bsr_matrix, "bsr"
    else:
        raise ValueError("Input matrices to dot_product_mkl must be CSR, CSC, or BSR; COO is not supported")

    # Override output if dense flag is set
    default_output = default_output if not dense else np.zeros

    # Check to make sure that this multiplication can work and check dtypes
    _sanity_check(matrix_a, matrix_b)

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix_a, matrix_b):
        debug_print("Skipping multiplication because A (dot) B must yield an empty matrix")
        final_dtype = np.float64 if matrix_a.dtype != matrix_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return default_output((matrix_a.shape[0], matrix_b.shape[1]), dtype=final_dtype)

    # Check dtypes
    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast)

    t = debug_timer()

    # Create intel MKL objects
    mkl_a, a_dbl = _create_mkl_sparse(matrix_a)
    mkl_b, b_dbl = _create_mkl_sparse(matrix_b)

    t = debug_timer("Created MKL sparse handles", t)

    # Call spmmd for dense output directly if the dense flag is set
    if dense:
        dense_arr = _matmul_mkl_dense(mkl_a, mkl_b, (matrix_a.shape[0], matrix_b.shape[1]), a_dbl or b_dbl)

        debug_timer("Multiplied matrices", t)

        _destroy_mkl_handle(mkl_a)
        _destroy_mkl_handle(mkl_b)

        return dense_arr

    # Call spmm for sparse output if the dense flag is not set and then export the sparse matrix to python
    else:
        # Dot product
        mkl_c = _matmul_mkl(mkl_a, mkl_b)

        _destroy_mkl_handle(mkl_a)
        _destroy_mkl_handle(mkl_b)

        t = debug_timer("Multiplied matrices", t)

        # Reorder
        if reorder_output:
            _order_mkl_handle(mkl_c)

            t = debug_timer("Reordered output indices", t)

        # Extract
        python_c = _export_mkl(mkl_c, a_dbl or b_dbl, output_type=output_type)
        _destroy_mkl_handle(mkl_c)

        debug_timer("Created python handle", t)

        return python_c
