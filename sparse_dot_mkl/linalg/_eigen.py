from sparse_dot_mkl._mkl_interface import (MKL, _libmkl, sparse_matrix_t, _is_double, _create_mkl_sparse,
                                           _destroy_mkl_handle, matrix_descr)

import warnings
import numpy as _np
import scipy.sparse as _sps
import ctypes as _ctypes
from numpy.ctypeslib import ndpointer

class _MKL_Eigen:

    # SVD
    # https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/extended-eigensolver-routines/extended-eigensolver-interfaces-for-extremal-eigenvalues-singular-values/extended-eigensolver-interfaces-to-find-largest-smallest-singular-values/mkl-sparse-svd.html
    _mkl_sparse_s_svd = _libmkl.mkl_sparse_s_svd
    _mkl_sparse_d_svd = _libmkl.mkl_sparse_d_svd
    _mkl_sparse_ee_init = _libmkl.mkl_sparse_ee_init

    @classmethod
    def _set_int_type(cls):
        
        cls._mkl_sparse_s_svd.argtypes = cls._set_svd_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_svd.restype = _ctypes.c_int

        cls._mkl_sparse_d_svd.argtypes = cls._set_svd_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_svd.restype = _ctypes.c_int

        cls._mkl_sparse_ee_init.argtypes = [ndpointer(shape=(128,), ndim=1, dtype=MKL.MKL_INT_NUMPY)]
        cls._mkl_sparse_ee_init.restype = None

    @staticmethod
    def _set_svd_argtypes(prec):
        return [_ctypes.POINTER(_ctypes.c_char),
                _ctypes.POINTER(_ctypes.c_char),
                ndpointer(dtype=MKL.MKL_INT_NUMPY, ndim=1, shape=(128, ), flags="CONTIGUOUS"),
                sparse_matrix_t,
                matrix_descr,
                MKL.MKL_INT,
                ndpointer(dtype=MKL.MKL_INT_NUMPY, ndim=1, shape=(1, ), flags="CONTIGUOUS"),
                ndpointer(dtype=prec, ndim=1, flags="CONTIGUOUS"),
                ndpointer(dtype=prec, ndim=2),
                ndpointer(dtype=prec, ndim=2),
                ndpointer(dtype=prec, ndim=1, flags="CONTIGUOUS")]


# Set argtypes based on MKL interface type
_MKL_Eigen._set_int_type()