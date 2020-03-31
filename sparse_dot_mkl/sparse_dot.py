from sparse_dot_mkl._sparse_sparse import _sparse_dot_sparse as _sds
from sparse_dot_mkl._sparse_dense import _sparse_dot_dense as _sdd
from sparse_dot_mkl._dense_dense import _dense_dot_dense as _ddd
import scipy.sparse as _spsparse


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

    if _spsparse.issparse(matrix_a) and _spsparse.issparse(matrix_b):
        return _sds(matrix_a, matrix_b, cast=cast, reorder_output=reorder_output, dense=dense, dprint=dprint)
    elif _spsparse.issparse(matrix_a) or _spsparse.issparse(matrix_b):
        return _sdd(matrix_a, matrix_b, cast=cast, dprint=dprint)
    else:
        return _ddd(matrix_a, matrix_b, cast=cast, dprint=dprint)
