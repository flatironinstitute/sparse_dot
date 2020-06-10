from sparse_dot_mkl._mkl_interface import (MKL, sparse_matrix_t, RETURN_CODES, _create_mkl_sparse,
                                           _export_mkl, _order_mkl_handle, _destroy_mkl_handle, _type_check,
                                           _empty_output_check, _sanity_check)
import ctypes as _ctypes
import numpy as np
import time
import scipy.sparse as _spsparse
from scipy.sparse import isspmatrix_csr as is_csr, isspmatrix_csc as is_csc


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
    if ret_val != 0:
        raise ValueError("mkl_sparse_spmm returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))

    return ref_handle


def _syrk_mkl(sp_ref_a):
    """
    Dot product an MKL object with its transpose and return a handle to the result

    :param sp_ref_a: Sparse matrix A handle
    :type sp_ref_a: sparse_matrix_t
    :return: Sparse matrix handle that is the dot product A * A.T
    :rtype: sparse_matrix_t
    """

    ref_handle = sparse_matrix_t()

    ret_val = MKL._mkl_sparse_syrk(_ctypes.c_int(10),
                                   sp_ref_a,
                                   _ctypes.byref(ref_handle))

    # Check return
    if ret_val != 0:
        raise ValueError("mkl_sparse_spmm returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))

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
    if ret_val != 0:
        err_msg = "{fn} returned {v} ({e})".format(fn=func.__name__, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err_msg)

    return output_arr


def _sparse_dot_sparse(matrix_a, matrix_b, cast=False, reorder_output=False, dense=False, dprint=print):
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
    :param dprint: Should debug and timing messages be printed. Defaults to false.
    :type dprint: function
    :return: Sparse matrix that is the result of A * B in CSR format
    :rtype: scipy.sparse.csr_matrix
    """

    # Check for allowed sparse matrix types

    if is_csr(matrix_a) and (is_csc(matrix_b) or is_csr(matrix_b)):
        default_output = _spsparse.csr_matrix
        output_type = "csr"
    elif is_csc(matrix_a) and (is_csc(matrix_b) or is_csr(matrix_b)):
        default_output = _spsparse.csc_matrix
        output_type = "csc"
    else:
        raise ValueError("Both input matrices to dot_product_mkl must be CSR or CSC; COO and BSR are not supported")

    # Override output if dense flag is set
    default_output = default_output if not dense else np.zeros

    # Check to make sure that this multiplication can work and check dtypes
    _sanity_check(matrix_a, matrix_b)

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix_a, matrix_b):
        dprint("Skipping multiplication because A (dot) B must yield an empty matrix")
        final_dtype = np.float64 if matrix_a.dtype != matrix_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return default_output((matrix_a.shape[0], matrix_b.shape[1]), dtype=final_dtype)

    # Check dtypes
    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast, dprint=dprint)

    t0 = time.time()

    # Create intel MKL objects
    mkl_a, a_dbl = _create_mkl_sparse(matrix_a)
    mkl_b, b_dbl = _create_mkl_sparse(matrix_b)

    t1 = time.time()
    dprint("Created MKL sparse handles: {0:.6f} seconds".format(t1 - t0))

    # Call spmmd for dense output directly if the dense flag is set
    if dense:
        dense_arr = _matmul_mkl_dense(mkl_a, mkl_b, (matrix_a.shape[0], matrix_b.shape[1]), a_dbl or b_dbl)

        t2 = time.time()
        dprint("Multiplied matrices: {0:.6f} seconds".format(t2 - t1))

        _destroy_mkl_handle(mkl_a)
        _destroy_mkl_handle(mkl_b)

        return dense_arr

    # Call spmm for sparse output if the dense flag is not set and then export the sparse matrix to python
    else:
        # Dot product
        mkl_c = _matmul_mkl(mkl_a, mkl_b)

        _destroy_mkl_handle(mkl_a)
        _destroy_mkl_handle(mkl_b)

        t2 = time.time()
        dprint("Multiplied matrices: {0:.6f} seconds".format(t2 - t1))

        # Reorder
        if reorder_output:
            _order_mkl_handle(mkl_c)

            dprint("Reordered indicies: {0:.6f} seconds".format(time.time() - t2))
            t2 = time.time()

        # Extract
        python_c = _export_mkl(mkl_c, a_dbl or b_dbl, output_type=output_type)
        _destroy_mkl_handle(mkl_c)

        dprint("Created python handle: {0:.6f} seconds".format(time.time() - t2))

        return python_c
