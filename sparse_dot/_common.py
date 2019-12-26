import numpy as np
from sparse_dot import NUMPY_FLOAT_DTYPES
import warnings


def _check_mkl_typing(mat_a, mat_b):
    """
    Check the data type for sparse arrays to be multiplied.
    Return True if the data is either in float64s or should be coerced to float64s for double precision mkl
    Return False if the data is in float32 for single precision mkl

    :param mat_a: Sparse matrix A in any format
    :type mat_a: scipy.sparse.spmatrix
    :param mat_b: Sparse matrix B in any format
    :type mat_b: scipy.sparse.spmatrix
    :return: True if double precision. False if single precision.
    :rtype: bool
    """

    # Check dtypes
    if mat_a.dtype == np.float32 and mat_b.dtype == np.float32:
        mkl_double_precision = False
    else:
        mkl_double_precision = True

    # Warn if dtypes are not the same
    if mat_a.dtype != mat_b.dtype:
        warnings.warn("Matrix dtypes are not identical. All data will be coerced to float64.")

    # Warn if dtypes are not floats
    if (mat_a.dtype not in NUMPY_FLOAT_DTYPES) or (mat_b.dtype not in NUMPY_FLOAT_DTYPES):
        warnings.warn("Matrix dtypes are not float32 or float64. All data will be coerced to float64.")

    return mkl_double_precision


def _check_alignment(mat_a, mat_b):
    """
    Make sure these matrices can be multiplied

    :param mat_a: Sparse matrix A in any format
    :type mat_a: scipy.sparse.spmatrix
    :param mat_b: Sparse matrix B in any format
    :type mat_b: scipy.sparse.spmatrix
    """

    if mat_a.shape[1] != mat_b.shape[0]:
        err = "Matrix alignment error: {m1} * {m2}".format(m1=mat_a.shape, m2=mat_b.shape)
        raise ValueError(err)
