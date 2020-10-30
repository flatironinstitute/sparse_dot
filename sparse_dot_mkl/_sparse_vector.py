from sparse_dot_mkl._mkl_interface import (MKL, _sanity_check, _empty_output_check, _type_check, _create_mkl_sparse,
                                           _destroy_mkl_handle, matrix_descr, RETURN_CODES, _is_dense_vector,
                                           _out_matrix, _check_return_value, _is_allowed_sparse_format)

import numpy as np
import ctypes as _ctypes


def _sparse_dense_vector_mult(matrix_a, vector_b, scalar=1., transpose=False, out=None, out_scalar=None, out_t=None):
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
    :param out: Add the dot product to this array if provided.
    :type out: np.ndarray, None
    :param out_scalar: Multiply the out array by this scalar if provided.
    :type out_scalar: float, None
    :return: A (dot) B as a dense array
    :rtype: np.ndarray
    """

    output_shape = matrix_a.shape[1] if transpose else matrix_a.shape[0]
    output_shape = (output_shape,) if vector_b.ndim == 1 else (output_shape, 1)

    if _empty_output_check(matrix_a, vector_b):
        final_dtype = np.float64 if matrix_a.dtype != vector_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return _out_matrix(output_shape, final_dtype, out_arr=out)

    mkl_a, dbl = _create_mkl_sparse(matrix_a)
    vector_b = vector_b.ravel()

    # Set functions and types for float or doubles
    output_ctype = _ctypes.c_double if dbl else _ctypes.c_float
    output_dtype = np.float64 if dbl else np.float32
    func = MKL._mkl_sparse_d_mv if dbl else MKL._mkl_sparse_s_mv

    output_arr = _out_matrix(output_shape, output_dtype, out_arr=out, out_t=out_t)

    ret_val = func(11 if transpose else 10,
                   scalar,
                   mkl_a,
                   matrix_descr(),
                   vector_b,
                   float(out_scalar) if out_scalar is not None else 1.,
                   output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)))

    # Check return
    _check_return_value(ret_val, func.__name__)

    _destroy_mkl_handle(mkl_a)

    return output_arr


def _sparse_dot_vector(mv_a, mv_b, cast=False, scalar=1., out=None, out_scalar=None):
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
    :param out: Add the dot product to this array if provided.
    :type out: np.ndarray, None
    :param out_scalar: Multiply the out array by this scalar if provided.
    :type out_scalar: float, None
    :return: A (dot) B as a dense matrix
    :rtype: np.ndarray
    """

    _sanity_check(mv_a, mv_b, allow_vector=True)
    mv_a, mv_b = _type_check(mv_a, mv_b, cast=cast)

    if not _is_allowed_sparse_format(mv_a) or not _is_allowed_sparse_format(mv_b):
        raise ValueError("Only CSR, CSC, and BSR-type sparse matrices are supported")
    elif _is_dense_vector(mv_b):
        return _sparse_dense_vector_mult(mv_a, mv_b, scalar=scalar, out=out, out_scalar=out_scalar)
    elif _is_dense_vector(mv_a) and out is None:
        return _sparse_dense_vector_mult(mv_b, mv_a.T, scalar=scalar, transpose=True).T
    elif _is_dense_vector(mv_a) and out is not None:
        _ = _sparse_dense_vector_mult(mv_b, mv_a.T, scalar=scalar, transpose=True,
                                      out=out.T, out_scalar=out_scalar, out_t=True)
        return out
    else:
        raise ValueError("Neither mv_a or mv_b is a dense vector")
