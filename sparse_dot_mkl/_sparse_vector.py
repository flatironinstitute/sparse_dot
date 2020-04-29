from sparse_dot_mkl._mkl_interface import (MKL, _sanity_check, _empty_output_check, _type_check, _create_mkl_sparse,
                                           _destroy_mkl_handle, matrix_descr, RETURN_CODES, _is_dense_vector)

import numpy as np
import ctypes as _ctypes


def _sparse_dense_vector_mult(matrix_a, vector_b, scalar=1., transpose=False):
    """
    Multiply together a sparse matrix and a dense vector

    :param matrix_a: Left (A) matrix
    :type matrix_a: sp.spmatrix.csr, sp.spmatrix.csc
    :param vector_b: Right (B) vector with shape (N, ) or (N, 1)
    :type vector_b: np.ndarray
    :param scalar: A value to multiply the result matrix by. Defaults to 1.
    :type scalar: float
    :param transpose: Return AT (dot) B instead of A (dot) B.
    :type transpose: bool
    :return: A (dot) B as a dense array
    :rtype: np.ndarray
    """

    output_shape = matrix_a.shape[1] if transpose else matrix_a.shape[0]
    output_shape = (output_shape, ) if vector_b.ndim == 1 else (output_shape, 1)

    if _empty_output_check(matrix_a, vector_b):
        final_dtype = np.float64 if matrix_a.dtype != vector_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return np.zeros(output_shape, dtype=final_dtype)

    mkl_a, dbl = _create_mkl_sparse(matrix_a)
    vector_b = vector_b.ravel()

    # Set functions and types for float or doubles
    output_ctype = _ctypes.c_double if dbl else _ctypes.c_float
    output_dtype = np.float64 if dbl else np.float32
    func = MKL._mkl_sparse_d_mv if dbl else MKL._mkl_sparse_s_mv

    output_arr = np.zeros(output_shape, dtype=output_dtype)

    ret_val = func(11 if transpose else 10,
                   scalar,
                   mkl_a,
                   matrix_descr(),
                   vector_b,
                   1.,
                   output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)))

    # Check return
    if ret_val != 0:
        err_msg = "{fn} returned {v} ({e})".format(fn=func.__name__, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err_msg)

    _destroy_mkl_handle(mkl_a)

    return output_arr


def _sparse_dot_vector(mv_a, mv_b, cast=False, dprint=print, scalar=1.):
    """
    Multiply a sparse matrix by a dense vector.
    The matrix must be CSR or CSC format.
    The vector must be (N,) or (N, 1) shape.
    Returns a dense vector of (N,) or (N, 1) shape (depending on vector)

    :param mv_a: Left (A) matrix or vector
    :type mv_a: np.ndarray, sp.spmatrix.csr, sp.spmatrix.csc
    :param mv_b: Right (B) matrix or vector
    :type mv_b: np.ndarray, sp.spmatrix.csr, sp.spmatrix.csc
    :param scalar: A value to multiply the result matrix by. Defaults to 1.
    :type scalar: float
    :param cast: Convert values to compatible floats if True. Raise an error if they are not compatible if False.
    Defaults to False.
    :type cast: bool
    :param dprint: A function that will handle debug strings. Defaults to print.
    :type dprint: function

    :return: A (dot) B as a dense matrix
    :rtype: np.ndarray
    """

    _sanity_check(mv_a, mv_b, allow_vector=True)
    mv_a, mv_b = _type_check(mv_a, mv_b, cast=cast, dprint=dprint)

    if _is_dense_vector(mv_b):
        return _sparse_dense_vector_mult(mv_a, mv_b, scalar=scalar)
    elif _is_dense_vector(mv_a):
        return _sparse_dense_vector_mult(mv_b, mv_a.T, scalar=scalar, transpose=True).T
