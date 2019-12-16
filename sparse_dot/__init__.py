import ctypes as _ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

# Define dtypes
# TODO: Figure out some way to figure this out from libmkl
MKL_INT = _ctypes.c_int
MKL_INT_NUMPY = np.int32

NUMPY_FLOAT_DTYPES = [np.float32, np.float64]

# Define standard return codes
RETURN_CODES = {0: "SPARSE_STATUS_SUCCESS",
                1: "SPARSE_STATUS_NOT_INITIALIZED",
                2: "SPARSE_STATUS_ALLOC_FAILED",
                3: "SPARSE_STATUS_INVALID_VALUE",
                4: "SPARSE_STATUS_EXECUTION_FAILED",
                5: "SPARSE_STATUS_INTERNAL_ERROR",
                6: "SPARSE_STATUS_NOT_SUPPORTED"}


# Load mkl_spblas.so through the common interface
_libmkl = _ctypes.cdll.LoadLibrary("libmkl_rt.so")


# Construct opaque struct & type
class _sparse_matrix(_ctypes.Structure):
    pass


sparse_matrix_t = _ctypes.POINTER(_sparse_matrix)

# Import function for creating a MKL CSR object
# https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr
_mkl_sparse_d_create_csr = _libmkl.mkl_sparse_d_create_csr
_mkl_sparse_d_create_csr.argtypes = [_ctypes.POINTER(sparse_matrix_t),
                                     _ctypes.c_int,
                                     MKL_INT,
                                     MKL_INT,
                                     ndpointer(dtype=MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                                     ndpointer(dtype=MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                                     ndpointer(dtype=MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                                     ndpointer(dtype=_ctypes.c_double, ndim=1, flags='C_CONTIGUOUS')]
_mkl_sparse_d_create_csr.restypes = _ctypes.c_int

# Import function for creating a MKL CSR object
# https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr
_mkl_sparse_s_create_csr = _libmkl.mkl_sparse_s_create_csr
_mkl_sparse_s_create_csr.argtypes = [_ctypes.POINTER(sparse_matrix_t),
                                     _ctypes.c_int,
                                     MKL_INT,
                                     MKL_INT,
                                     ndpointer(dtype=MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                                     ndpointer(dtype=MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                                     ndpointer(dtype=MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                                     ndpointer(dtype=_ctypes.c_float, ndim=1, flags='C_CONTIGUOUS')]
_mkl_sparse_s_create_csr.restypes = _ctypes.c_int

# Export function for exporting a MKL CSR object
# https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-export-csr
_mkl_sparse_d_export_csr = _libmkl.mkl_sparse_d_export_csr
_mkl_sparse_d_export_csr.argtypes = [sparse_matrix_t,
                                     _ctypes.POINTER(_ctypes.c_int),
                                     _ctypes.POINTER(MKL_INT),
                                     _ctypes.POINTER(MKL_INT),
                                     _ctypes.POINTER(_ctypes.POINTER(MKL_INT)),
                                     _ctypes.POINTER(_ctypes.POINTER(MKL_INT)),
                                     _ctypes.POINTER(_ctypes.POINTER(MKL_INT)),
                                     _ctypes.POINTER(_ctypes.POINTER(_ctypes.c_double))]
_mkl_sparse_d_export_csr.restypes = _ctypes.c_int

# Export function for exporting a MKL CSR object
# https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-export-csr
_mkl_sparse_s_export_csr = _libmkl.mkl_sparse_s_export_csr
_mkl_sparse_s_export_csr.argtypes = [sparse_matrix_t,
                                     _ctypes.POINTER(_ctypes.c_int),
                                     _ctypes.POINTER(MKL_INT),
                                     _ctypes.POINTER(MKL_INT),
                                     _ctypes.POINTER(_ctypes.POINTER(MKL_INT)),
                                     _ctypes.POINTER(_ctypes.POINTER(MKL_INT)),
                                     _ctypes.POINTER(_ctypes.POINTER(MKL_INT)),
                                     _ctypes.POINTER(_ctypes.POINTER(_ctypes.c_float))]
_mkl_sparse_s_export_csr.restypes = _ctypes.c_int


# Import function for matmul
# https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-spmm
_mkl_sparse_spmm = _libmkl.mkl_sparse_spmm
_mkl_sparse_spmm.argtypes = [_ctypes.c_int,
                             sparse_matrix_t,
                             sparse_matrix_t,
                             _ctypes.POINTER(sparse_matrix_t)]
_mkl_sparse_spmm.restypes = _ctypes.c_int

# Import function for cleaning up MKL objects
# https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-destroy
_mkl_sparse_destroy = _libmkl.mkl_sparse_destroy
_mkl_sparse_destroy.argtypes = [sparse_matrix_t]
_mkl_sparse_destroy.restypes = _ctypes.c_int

# Import function for ordering MKL objects
# https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-order
_mkl_sparse_order = _libmkl.mkl_sparse_order
_mkl_sparse_order.argtypes = [sparse_matrix_t]
_mkl_sparse_order.restypes = _ctypes.c_int
