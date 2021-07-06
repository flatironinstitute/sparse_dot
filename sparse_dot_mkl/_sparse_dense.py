from sparse_dot_mkl._mkl_interface import (MKL, _sanity_check, _empty_output_check, _type_check, _create_mkl_sparse,
                                           _destroy_mkl_handle, matrix_descr, debug_print, _convert_to_csr,
                                           _get_numpy_layout, _check_return_value, LAYOUT_CODE_C, LAYOUT_CODE_F,
                                           _out_matrix)
import numpy as np
import ctypes as _ctypes
import scipy.sparse as _spsparse


def _sparse_dense_matmul(matrix_a, matrix_b, scalar=1., transpose=False, out=None, out_scalar=None, out_t=None):
    """
    Multiply together a sparse and a dense matrix
    mkl_sparse_?_mm requires the left (A) matrix to be sparse and the right (B) matrix to be dense
    This requires conversion of the sparse matrix to CSR format for some dense arrays.
    A must be CSR if B is column-major. Otherwise CSR or CSC are acceptable.

    :param matrix_a: Left (A) matrix
    :type matrix_a: sp.spmatrix.csr, sp.spmatrix.csc
    :param matrix_b: Right (B) matrix
    :type matrix_b: np.ndarray
    :param scalar: A value to multiply the result matrix by. Defaults to 1.
    :type scalar: float
    :param transpose: Return AT (dot) B instead of A (dot) B.
    :type transpose: bool
    :param out: Add the dot product to this array if provided.
    :type out: np.ndarray, None
    :param out_scalar: Multiply the out array by this scalar if provided.
    :type out_scalar: float, None
    :return: A (dot) B as a dense array in either column-major or row-major format
    :rtype: np.ndarray
    """

    _mkl_handles = []

    output_shape = (matrix_a.shape[1] if transpose else matrix_a.shape[0], matrix_b.shape[1])
    layout_b, ld_b = _get_numpy_layout(matrix_b, second_arr=out)

    try:
        # Prep MKL handles and check that matrixes are compatible types
        # MKL requires CSR format if the dense array is column-major
        if layout_b == LAYOUT_CODE_F and not _spsparse.isspmatrix_csr(matrix_a):
            mkl_non_csr, dbl = _create_mkl_sparse(matrix_a)
            _mkl_handles.append(mkl_non_csr)
            mkl_a = _convert_to_csr(mkl_non_csr)
        else:
            mkl_a, dbl = _create_mkl_sparse(matrix_a)

        _mkl_handles.append(mkl_a)

        # Set functions and types for float or doubles
        output_ctype = _ctypes.c_double if dbl else _ctypes.c_float
        output_dtype = np.float64 if dbl else np.float32
        func = MKL._mkl_sparse_d_mm if dbl else MKL._mkl_sparse_s_mm

        # Allocate an output array
        output_arr = _out_matrix(output_shape, output_dtype, order="C" if layout_b == LAYOUT_CODE_C else "F",
                                out_arr=out, out_t=out_t)

        output_layout, output_ld = _get_numpy_layout(output_arr, second_arr=matrix_b)

        ret_val = func(11 if transpose else 10,
                    scalar,
                    mkl_a,
                    matrix_descr(),
                    layout_b,
                    matrix_b,
                    output_shape[1],
                    ld_b,
                    float(out_scalar) if out_scalar is not None else 1.,
                    output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
                    output_ld)

        # Check return
        _check_return_value(ret_val, func.__name__)

        return output_arr

    finally:
        for _mhandle in _mkl_handles:
            _destroy_mkl_handle(_mhandle)


def _sparse_dot_dense(matrix_a, matrix_b, cast=False, scalar=1., out=None, out_scalar=None):
    """
    Multiply together a dense and a sparse matrix.
    If the sparse matrix is not CSR, it may need to be reordered, depending on the order of the dense array.

    :param matrix_a: Left (A) matrix
    :type matrix_a: np.ndarray, sp.spmatrix.csr, sp.spmatrix.csc
    :param matrix_b: Right (B) matrix
    :type matrix_b: np.ndarray, sp.spmatrix.csr, sp.spmatrix.csc
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
    _sanity_check(matrix_a, matrix_b)

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix_a, matrix_b):
        debug_print("Skipping multiplication because A (dot) B must yield an empty matrix")
        final_dtype = np.float64 if matrix_a.dtype != matrix_b.dtype or matrix_a.dtype != np.float32 else np.float32
        return _out_matrix((matrix_a.shape[0], matrix_b.shape[1]), final_dtype, out_arr=out)

    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast)

    if sum([_spsparse.isspmatrix(matrix_a), _spsparse.isspmatrix(matrix_b)]) != 1:
        raise ValueError("_sparse_dot_dense takes one sparse and one dense array")
    elif _spsparse.isspmatrix(matrix_a):
        return _sparse_dense_matmul(matrix_a, matrix_b, scalar=scalar, out=out, out_scalar=out_scalar)
    elif _spsparse.isspmatrix(matrix_b) and out is not None:
        _ = _sparse_dense_matmul(matrix_b, matrix_a.T, scalar=scalar, transpose=True,
                                 out=out.T, out_scalar=out_scalar, out_t=True)
        return out
    elif _spsparse.isspmatrix(matrix_b) and out is None:
        return _sparse_dense_matmul(matrix_b, matrix_a.T, scalar=scalar, transpose=True).T
