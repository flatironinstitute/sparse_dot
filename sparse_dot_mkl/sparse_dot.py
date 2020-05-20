from sparse_dot_mkl._sparse_sparse import _sparse_dot_sparse as _sds
from sparse_dot_mkl._sparse_dense import _sparse_dot_dense as _sdd
from sparse_dot_mkl._dense_dense import _dense_dot_dense as _ddd
from sparse_dot_mkl._sparse_vector import _sparse_dot_vector as _sdv
from sparse_dot_mkl._sparse_qr_solver import sparse_qr_solver as _qrs
from sparse_dot_mkl._mkl_interface import get_version_string, _is_dense_vector
import scipy.sparse as _spsparse
import numpy as _np


def dot_product_mkl(matrix_a, matrix_b, cast=False, copy=True, reorder_output=False, dense=False, debug=False):
    """
    Multiply together matrixes using the intel Math Kernel Library.
    This currently only supports float32 and float64 data

    :param matrix_a: Sparse matrix A in CSC/CSR format or dense matrix in numpy format
    :type matrix_a: scipy.sparse.spmatrix, np.ndarray
    :param matrix_b: Sparse matrix B in CSC/CSR format or dense matrix in numpy format
    :type matrix_b: scipy.sparse.spmatrix, np.ndarray
    :param cast: Should the data be coerced into float64 if it isn't float32 or float64
    If set to True and any other dtype is passed, the matrix data will copied internally before multiplication
    If set to False and any dtype that isn't float32 or float64 is passed, a ValueError will be raised
    Defaults to False
    :param copy: Deprecated flag to force copy. Removed because the behavior was inconsistent.
    :type copy: bool
    :param reorder_output: Should the array indices be reordered using MKL
    If set to True, the object in C will be ordered and then exported into python
    If set to False, the array column indices will not be ordered.
    The scipy sparse dot product does not yield ordered column indices so this defaults to False
    :type reorder_output: bool
    :param dense: Should the matrix multiplication be put into a dense numpy array
    This does not require any copy and is memory efficient if the output array density is > 50%
    Note that this flag has no effect if one input array is dense; then the output will always be dense
    :type dense: bool
    :param debug: Should debug and timing messages be printed. Defaults to false.
    :type debug: bool
    :return: Matrix that is the result of A * B in input-dependent format
    :rtype: scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, np.ndarray
    """

    dprint = print if debug else lambda *x: x

    if get_version_string() is None and debug:
        dprint("mkl-service must be installed to get full debug messaging")
    elif debug:
        dprint(get_version_string())

    num_sparse = sum((_spsparse.issparse(matrix_a), _spsparse.issparse(matrix_b)))
    num_vectors = sum((_is_dense_vector(matrix_a), _is_dense_vector(matrix_b)))

    # SPARSE (DOT) SPARSE #
    if num_sparse == 2:
        return _sds(matrix_a, matrix_b, cast=cast, reorder_output=reorder_output, dense=dense, dprint=dprint)

    # SPARSE (DOT) VECTOR #
    elif num_sparse == 1 and num_vectors == 1:
        return _sdv(matrix_a, matrix_b, cast=cast, dprint=dprint)

    # SPARSE (DOT) DENSE & DENSE (DOT) SPARSE #
    elif num_sparse == 1:
        return _sdd(matrix_a, matrix_b, cast=cast, dprint=dprint)

    # SPECIAL CASE OF VECTOR (DOT) VECTOR #
    # THIS IS JUST EASIER THAN GETTING THIS EDGE CONDITION RIGHT IN MKL #
    elif num_vectors == 2:
        return _np.dot(matrix_a, matrix_b)

    # DENSE (DOT) DENSE
    else:
        return _ddd(matrix_a, matrix_b, cast=cast, dprint=dprint)


def sparse_qr_solve_mkl(matrix_a, matrix_b, cast=False, debug=False):
    """
    Solve AX = B for X where A is sparse and B is dense

    :param matrix_a: Sparse matrix (solver requires CSR; will convert if cast=True)
    :type matrix_a: np.ndarray
    :param matrix_b: Dense matrix
    :type matrix_b: np.ndarray
    :param cast: Should the data be coerced into float64 if it isn't float32 or float64,
    and should a CSR matrix be cast to a CSC matrix.
    Defaults to False
    :type cast: bool
    :param debug: Should debug messages be printed. Defaults to false.
    :type debug: bool
    :return: Dense array X
    :rtype: np.ndarray
    """

    dprint = print if debug else lambda *x: x

    if get_version_string() is None and debug:
        dprint("mkl-service must be installed to get full debug messaging")
    elif debug:
        dprint(get_version_string())

    return _qrs(matrix_a, matrix_b, cast=cast, dprint=dprint)
