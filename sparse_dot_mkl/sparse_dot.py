from sparse_dot_mkl._sparse_sparse import _sparse_dot_sparse as _sds
from sparse_dot_mkl._sparse_dense import _sparse_dot_dense as _sdd
from sparse_dot_mkl._dense_dense import _dense_dot_dense as _ddd
from sparse_dot_mkl._sparse_vector import _sparse_dot_vector as _sdv
from sparse_dot_mkl._gram_matrix import _gram_matrix as _gm
from sparse_dot_mkl._sparse_qr_solver import sparse_qr_solver as _qrs
from sparse_dot_mkl._mkl_interface import (
    print_mkl_debug,
    _is_dense_vector,
    set_debug_mode,
    get_version_string,
)
import scipy.sparse as _spsparse
import numpy as _np
import warnings


def dot_product_mkl(
    matrix_a,
    matrix_b,
    cast=False,
    copy=True,
    reorder_output=False,
    dense=False,
    debug=False,
    out=None,
    out_scalar=None,
):
    """
    Multiply together matrixes using the intel Math Kernel Library.
    This currently only supports float32 and float64 data

    :param matrix_a: Sparse matrix A in CSC/CSR format or dense matrix
        in numpy format
    :type matrix_a: scipy.sparse.spmatrix, np.ndarray
    :param matrix_b: Sparse matrix B in CSC/CSR format or dense matrix
        in numpy format
    :type matrix_b: scipy.sparse.spmatrix, np.ndarray
    :param cast: Should the data be coerced into float64 if it isn't
        float32 or float64
        If set to True and any other dtype is passed, the matrix data will
        be copied internally before multiplication
        If set to False and any dtype that isn't float32 or float64 is passed,
        a ValueError will be raised
        Defaults to False
    :param copy: Deprecated flag to force copy.
        Removed because the behavior was inconsistent.
    :type copy: bool
    :param reorder_output: Should the array indices be reordered using MKL
        If set to True, the array column indices will be ordered before return
        If set to False, the array column indices will not be ordered.
        The scipy sparse dot product does not yield ordered column indices so
        this defaults to False
    :type reorder_output: bool
    :param dense: Should the matrix multiplication be put into a dense array
        This does not require a copy from a sparse format.
        Note that this flag has no effect if one input array is dense;
        then the output will always be dense
    :type dense: bool
    :param debug: Deprecated debug flag.
        Use `sparse_dot_mkl.set_debug_mode(True)`
    :type debug: bool
    :param out: Add the dot product to this array if provided.
    :type out: np.ndarray, None
    :param out_scalar: Multiply the out array by this scalar if provided.
    :type out_scalar: float, None
    :return: Matrix that is the result of A * B in input-dependent format
    :rtype: scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, np.ndarray
    """

    if debug:
        warnings.warn(
            "Set debug mode with sparse_dot_mkl.set_debug_mode(True)",
            DeprecationWarning
        )

    print_mkl_debug()

    num_sparse = sum((
        _spsparse.issparse(matrix_a),
        _spsparse.issparse(matrix_b)
    ))

    # SPARSE (DOT) SPARSE #
    if num_sparse == 2 and out is not None:
        raise ValueError(
            "out argument cannot be used with sparse (dot) sparse "
            "matrix multiplication"
        )

    elif num_sparse == 2:
        return _sds(
            matrix_a,
            matrix_b,
            cast=cast,
            reorder_output=reorder_output,
            dense=dense
        )

    # SPARSE (DOT) VECTOR #
    elif (
        num_sparse == 1
        and _is_dense_vector(matrix_a)
        and (matrix_a.ndim == 1 or matrix_a.shape[0] == 1)
    ):
        return _sdv(
            matrix_a,
            matrix_b,
            cast=cast,
            out=out,
            out_scalar=out_scalar
        )

    # SPARSE (DOT) VECTOR #
    elif (
        num_sparse == 1
        and _is_dense_vector(matrix_b)
        and (matrix_b.ndim == 1 or matrix_b.shape[1] == 1)
    ):
        return _sdv(
            matrix_a,
            matrix_b,
            cast=cast,
            out=out,
            out_scalar=out_scalar
        )

    # SPARSE (DOT) DENSE & DENSE (DOT) SPARSE #
    elif num_sparse == 1:
        return _sdd(
            matrix_a,
            matrix_b,
            cast=cast,
            out=out,
            out_scalar=out_scalar
        )

    # SPECIAL CASE OF VECTOR (DOT) VECTOR #
    # THIS IS JUST EASIER THAN GETTING THIS EDGE CONDITION RIGHT IN MKL #
    elif (
        _is_dense_vector(matrix_a)
        and _is_dense_vector(matrix_b)
        and (matrix_a.ndim == 1 or matrix_b.ndim == 1)
    ):
        if out_scalar is not None:
            out *= out_scalar
        return _np.dot(matrix_a, matrix_b, out=out)

    # DENSE (DOT) DENSE
    else:
        return _ddd(
            matrix_a,
            matrix_b,
            cast=cast,
            out=out,
            out_scalar=out_scalar
        )


def gram_matrix_mkl(
    matrix,
    transpose=False,
    cast=False,
    dense=False,
    debug=False,
    reorder_output=False,
    out=None,
    out_scalar=None,
):
    """
    Calculate a gram matrix (AT (dot) A) matrix.
    Note that this should calculate only the upper triangular matrix.
    However providing a sparse matrix with transpose=False and
    dense=True will calculate a full matrix
    (this appears to be a bug in mkl_sparse_?_syrkd)

    :param matrix: Sparse matrix in CSR or CSC format or numpy array
    :type matrix: csr_matrix, csc_matrix, numpy.ndarray
    :param transpose: Calculate A (dot) AT instead
    :type transpose: bool
    :param cast: Make internal copies to convert matrix to a float matrix
        or convert to a CSR matrix if necessary
    :type cast: bool
    :param dense: Produce a dense matrix output instead of a sparse matrix
    :type dense: bool
    :param debug: Deprecated debug flag.
        Use `sparse_dot_mkl.set_debug_mode(True)`
    :type debug: bool
    :param reorder_output: Should the array indices be reordered using MKL
        The scipy sparse dot product does not yield ordered column indices
        so this defaults to False
    :type reorder_output: bool
    :param out: Add the dot product to this array if provided.
    :type out: np.ndarray, None
    :param out_scalar: Multiply the out array by this scalar if provided.
    :type out_scalar: float, None
    :return: Gram matrix
    :rtype: scipy.sparse.csr_matrix, np.ndarray"""

    if debug:
        warnings.warn(
            "Set debug mode with sparse_dot_mkl.set_debug_mode(True)",
            DeprecationWarning
        )

    print_mkl_debug()

    return _gm(
        matrix,
        transpose=transpose,
        cast=cast,
        dense=dense,
        reorder_output=reorder_output,
        out=out,
        out_scalar=out_scalar,
    )


def sparse_qr_solve_mkl(
    matrix_a,
    matrix_b,
    cast=False,
    debug=False
):
    """
    Solve AX = B for X where A is sparse and B is dense

    :param matrix_a: Sparse matrix
        (solver requires CSR; will convert if cast=True)
    :type matrix_a: np.ndarray
    :param matrix_b: Dense matrix
    :type matrix_b: np.ndarray
    :param cast: Should the data be coerced into float64
        if it isn't float32 or float64,
    and should a CSR matrix be cast to a CSC matrix.
    Defaults to False
    :type cast: bool
    :param debug: Deprecated debug flag.
        Use `sparse_dot_mkl.set_debug_mode(True)`
    :type debug: bool
    :return: Dense array X
    :rtype: np.ndarray
    """

    if debug:
        warnings.warn(
            "Set debug mode with sparse_dot_mkl.set_debug_mode(True)",
            DeprecationWarning
        )

    print_mkl_debug()

    return _qrs(matrix_a, matrix_b, cast=cast)


# Alias for backwards compatibility
dot_product_transpose_mkl = gram_matrix_mkl
