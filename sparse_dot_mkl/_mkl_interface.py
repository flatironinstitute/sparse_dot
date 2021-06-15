import os
import time
import warnings
import ctypes as _ctypes
import ctypes.util as _ctypes_util

# Workaround for this stupid sklearn thing
if 'KMP_INIT_AT_FORK' in os.environ:
    _sklearn_env = os.environ['KMP_INIT_AT_FORK']
    del os.environ['KMP_INIT_AT_FORK']
else:
    _sklearn_env = None

# Load mkl_spblas through the libmkl_rt common interface
_libmkl = None
try:
    _so_file = _ctypes_util.find_library('mkl_rt')

    if _so_file is None:
        _so_file = _ctypes_util.find_library('mkl_rt.1')

    if _so_file is None:
        # For some reason, find_library is not checking LD_LIBRARY_PATH
        # If the ctypes.util approach doesn't work, try this (crude) approach

        # Check each of these library names
        # Also include derivatives because windows find_library implementation won't match partials
        for so_file in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_rt.dll", "mkl_rt.1.dll"]:
            try:
                _libmkl = _ctypes.cdll.LoadLibrary(so_file)
                break
            except (OSError, ImportError) as err:
                pass

        if _libmkl is None:
            raise ImportError("mkl_rt not found.")
    else:
        _libmkl = _ctypes.cdll.LoadLibrary(_so_file)
except (OSError, ImportError) as err:
    _ierr_msg = "Unable to load the MKL libraries through libmkl_rt. Try setting $LD_LIBRARY_PATH. " + str(err)
    raise ImportError(_ierr_msg)

# Use mkl-service to check version if it's installed
# Since it's not on PyPi I don't want to make this an actual package dependency
# So without it just create mock functions and don't do version checking
try:
    from mkl import get_version, get_version_string
except ImportError:
    def get_version():
        return None


    def get_version_string():
        return None

if get_version() is not None and get_version()["MajorVersion"] < 2020:
    _verr_msg = "Loaded version of MKL is out of date: {v}".format(v=get_version_string())
    warnings.warn(_verr_msg)

import numpy as np
import scipy.sparse as _spsparse
from numpy.ctypeslib import ndpointer, as_array

NUMPY_FLOAT_DTYPES = [np.float32, np.float64]


class MKL:
    """ This class holds shared object references to C functions with arg and returntypes that can be adjusted"""

    MKL_INT = None
    MKL_INT_NUMPY = None
    MKL_DEBUG = False

    # Import function for creating a MKL CSR object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr
    _mkl_sparse_d_create_csr = _libmkl.mkl_sparse_d_create_csr

    # Import function for creating a MKL CSR object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr
    _mkl_sparse_s_create_csr = _libmkl.mkl_sparse_s_create_csr

    # Import function for creating a MKL CSC object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csc
    _mkl_sparse_d_create_csc = _libmkl.mkl_sparse_d_create_csc

    # Import function for creating a MKL CSC object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csc
    _mkl_sparse_s_create_csc = _libmkl.mkl_sparse_s_create_csc

    # Import function for creating a MKL BSR object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-bsr
    _mkl_sparse_d_create_bsr = _libmkl.mkl_sparse_d_create_bsr

    # Import function for creating a MKL BSR object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-bsr
    _mkl_sparse_s_create_bsr = _libmkl.mkl_sparse_s_create_bsr

    # Export function for exporting a MKL CSR object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-export-csr
    _mkl_sparse_d_export_csr = _libmkl.mkl_sparse_d_export_csr

    # Export function for exporting a MKL CSR object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-export-csr
    _mkl_sparse_s_export_csr = _libmkl.mkl_sparse_s_export_csr

    # Export function for exporting a MKL CSC object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-export-csc
    _mkl_sparse_d_export_csc = _libmkl.mkl_sparse_d_export_csc

    # Export function for exporting a MKL CSC object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-export-csc
    _mkl_sparse_s_export_csc = _libmkl.mkl_sparse_s_export_csc

    # Export function for exporting a MKL BSR object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-export-bsr
    _mkl_sparse_d_export_bsr = _libmkl.mkl_sparse_d_export_bsr

    # Export function for exporting a MKL BSR object
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-export-bsr
    _mkl_sparse_s_export_bsr = _libmkl.mkl_sparse_s_export_bsr

    # Import function for matmul
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-spmm
    _mkl_sparse_spmm = _libmkl.mkl_sparse_spmm

    # Import function for cleaning up MKL objects
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-destroy
    _mkl_sparse_destroy = _libmkl.mkl_sparse_destroy

    # Import function for ordering MKL objects
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-order
    _mkl_sparse_order = _libmkl.mkl_sparse_order

    # Import function for coverting to CSR
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-convert-csr
    _mkl_sparse_convert_csr = _libmkl.mkl_sparse_convert_csr

    # Import function for matmul single dense
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-spmm
    _mkl_sparse_s_spmmd = _libmkl.mkl_sparse_s_spmmd

    # Import function for matmul double dense
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-spmm
    _mkl_sparse_d_spmmd = _libmkl.mkl_sparse_d_spmmd

    # Import function for matmul single sparse*dense
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-mm
    _mkl_sparse_s_mm = _libmkl.mkl_sparse_s_mm

    # Import function for matmul double sparse*dense
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-mm
    _mkl_sparse_d_mm = _libmkl.mkl_sparse_d_mm

    # Import function for matmul single dense*dense
    # https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm
    _cblas_sgemm = _libmkl.cblas_sgemm

    # Import function for matmul double dense*dense
    # https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm
    _cblas_dgemm = _libmkl.cblas_dgemm

    # Import function for matrix * vector
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-mv
    _mkl_sparse_s_mv = _libmkl.mkl_sparse_s_mv

    # Import function for matrix * vector
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-mv
    _mkl_sparse_d_mv = _libmkl.mkl_sparse_d_mv

    # Import function for sparse gram matrix
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-syrk
    _mkl_sparse_syrk = _libmkl.mkl_sparse_syrk

    # Import function for dense single gram matrix from sparse
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-syrkd
    _mkl_sparse_s_syrkd = _libmkl.mkl_sparse_s_syrkd

    # Import function for dense double gram matrix from sparse
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-syrkd
    _mkl_sparse_d_syrkd = _libmkl.mkl_sparse_d_syrkd

    # Import function for dense single gram matrix
    # https://software.intel.com/en-us/mkl-developer-reference-c-cblas-syrk
    _cblas_ssyrk = _libmkl.cblas_ssyrk

    # Import function for dense double gram matrix
    # https://software.intel.com/en-us/mkl-developer-reference-c-cblas-syrk
    _cblas_dsyrk = _libmkl.cblas_dsyrk

    # Import function for QR solver - reorder
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-qr-reorder
    _mkl_sparse_qr_reorder = _libmkl.mkl_sparse_qr_reorder

    # Import function for QR solver - factorize
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-qr-factorize
    _mkl_sparse_d_qr_factorize = _libmkl.mkl_sparse_d_qr_factorize

    # Import function for QR solver - factorize
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-qr-factorize
    _mkl_sparse_s_qr_factorize = _libmkl.mkl_sparse_s_qr_factorize

    # Import function for QR solver - solve
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-qr-solve
    _mkl_sparse_d_qr_solve = _libmkl.mkl_sparse_d_qr_solve

    # Import function for QR solver - solve
    # https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-qr-solve
    _mkl_sparse_s_qr_solve = _libmkl.mkl_sparse_s_qr_solve

    @classmethod
    def _set_int_type(cls, c_type, np_type):
        cls.MKL_INT = c_type
        cls.MKL_INT_NUMPY = np_type

        cls._mkl_sparse_d_create_csr.argtypes = cls._mkl_sparse_create_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_create_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_s_create_csr.argtypes = cls._mkl_sparse_create_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_create_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_d_create_csc.argtypes = cls._mkl_sparse_create_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_create_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_s_create_csc.argtypes = cls._mkl_sparse_create_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_create_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_d_create_bsr.argtypes = cls._mkl_sparse_create_bsr_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_create_bsr.restypes = _ctypes.c_int

        cls._mkl_sparse_s_create_bsr.argtypes = cls._mkl_sparse_create_bsr_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_create_bsr.restypes = _ctypes.c_int

        cls._mkl_sparse_d_export_csr.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_export_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_csr.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_export_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_d_export_csc.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_export_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_bsr.argtypes = cls._mkl_sparse_export_bsr_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_export_bsr.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_bsr.argtypes = cls._mkl_sparse_export_bsr_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_export_bsr.restypes = _ctypes.c_int

        cls._mkl_sparse_spmm.argtypes = [_ctypes.c_int,
                                         sparse_matrix_t,
                                         sparse_matrix_t,
                                         _ctypes.POINTER(sparse_matrix_t)]
        cls._mkl_sparse_spmm.restypes = _ctypes.c_int

        cls._mkl_sparse_s_spmmd.argtypes = cls._mkl_sparse_spmmd_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_spmmd.restypes = _ctypes.c_int

        cls._mkl_sparse_d_spmmd.argtypes = cls._mkl_sparse_spmmd_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_spmmd.restypes = _ctypes.c_int

        cls._mkl_sparse_s_mm.argtypes = cls._mkl_sparse_mm_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_mm.restypes = _ctypes.c_int

        cls._mkl_sparse_d_mm.argtypes = cls._mkl_sparse_mm_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_mm.restypes = _ctypes.c_int

        cls._cblas_sgemm.argtypes = cls._cblas_gemm_argtypes(_ctypes.c_float)
        cls._cblas_sgemm.restypes = None

        cls._cblas_dgemm.argtypes = cls._cblas_gemm_argtypes(_ctypes.c_double)
        cls._cblas_dgemm.restypes = None

        cls._mkl_sparse_destroy.argtypes = [sparse_matrix_t]
        cls._mkl_sparse_destroy.restypes = _ctypes.c_int

        cls._mkl_sparse_order.argtypes = [sparse_matrix_t]
        cls._mkl_sparse_order.restypes = _ctypes.c_int

        cls._mkl_sparse_s_mv.argtypes = cls._mkl_sparse_mv_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_mv.restypes = _ctypes.c_int

        cls._mkl_sparse_d_mv.argtypes = cls._mkl_sparse_mv_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_mv.restypes = _ctypes.c_int

        cls._mkl_sparse_syrk.argtypes = [_ctypes.c_int,
                                         sparse_matrix_t,
                                         _ctypes.POINTER(sparse_matrix_t)]
        cls._mkl_sparse_syrk.restypes = _ctypes.c_int

        cls._mkl_sparse_s_syrkd.argtypes = cls._mkl_sparse_syrkd_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_syrkd.restypes = _ctypes.c_int

        cls._mkl_sparse_d_syrkd.argtypes = cls._mkl_sparse_syrkd_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_syrkd.restypes = _ctypes.c_int

        cls._cblas_ssyrk.argtypes = cls._cblas_syrk_argtypes(_ctypes.c_float)
        cls._cblas_ssyrk.restypes = None

        cls._cblas_dsyrk.argtypes = cls._cblas_syrk_argtypes(_ctypes.c_double)
        cls._cblas_dsyrk.restypes = None

        cls._mkl_sparse_qr_reorder.argtypes = [sparse_matrix_t, matrix_descr]
        cls._mkl_sparse_qr_reorder.restypes = _ctypes.c_int

        cls._mkl_sparse_d_qr_factorize.argtypes = [sparse_matrix_t, _ctypes.POINTER(_ctypes.c_double)]
        cls._mkl_sparse_d_qr_factorize.restypes = _ctypes.c_int

        cls._mkl_sparse_s_qr_factorize.argtypes = [sparse_matrix_t, _ctypes.POINTER(_ctypes.c_float)]
        cls._mkl_sparse_s_qr_factorize.restypes = _ctypes.c_int

        cls._mkl_sparse_d_qr_solve.argtypes = cls._mkl_sparse_qr_solve(_ctypes.c_double)
        cls._mkl_sparse_d_qr_solve.restypes = _ctypes.c_int

        cls._mkl_sparse_s_qr_solve.argtypes = cls._mkl_sparse_qr_solve(_ctypes.c_float)
        cls._mkl_sparse_s_qr_solve.restypes = _ctypes.c_int

    def __init__(self):
        raise NotImplementedError("This class is not intended to be instanced")

    """ The following methods return the argtype lists for each MKL function that has s and d variants"""

    @staticmethod
    def _mkl_sparse_create_argtypes(prec_type):
        return [_ctypes.POINTER(sparse_matrix_t),
                _ctypes.c_int,
                MKL.MKL_INT,
                MKL.MKL_INT,
                ndpointer(dtype=MKL.MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                ndpointer(dtype=MKL.MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                ndpointer(dtype=MKL.MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                ndpointer(dtype=prec_type, ndim=1, flags='C_CONTIGUOUS')]

    @staticmethod
    def _mkl_sparse_create_bsr_argtypes(prec_type):
        return [_ctypes.POINTER(sparse_matrix_t),
                _ctypes.c_int,
                _ctypes.c_int,
                MKL.MKL_INT,
                MKL.MKL_INT,
                MKL.MKL_INT,
                ndpointer(dtype=MKL.MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                ndpointer(dtype=MKL.MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                ndpointer(dtype=MKL.MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                ndpointer(dtype=prec_type, ndim=3, flags='C_CONTIGUOUS')]

    @staticmethod
    def _mkl_sparse_export_argtypes(prec_type):
        return [sparse_matrix_t,
                _ctypes.POINTER(_ctypes.c_int),
                _ctypes.POINTER(MKL.MKL_INT),
                _ctypes.POINTER(MKL.MKL_INT),
                _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
                _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
                _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
                _ctypes.POINTER(_ctypes.POINTER(prec_type))]

    @staticmethod
    def _mkl_sparse_export_bsr_argtypes(prec_type):
        return [sparse_matrix_t,
                _ctypes.POINTER(_ctypes.c_int),
                _ctypes.POINTER(_ctypes.c_int),
                _ctypes.POINTER(MKL.MKL_INT),
                _ctypes.POINTER(MKL.MKL_INT),
                _ctypes.POINTER(MKL.MKL_INT),
                _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
                _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
                _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
                _ctypes.POINTER(_ctypes.POINTER(prec_type))]

    @staticmethod
    def _cblas_gemm_argtypes(prec_type):
        return [_ctypes.c_int,
                _ctypes.c_int,
                _ctypes.c_int,
                MKL.MKL_INT,
                MKL.MKL_INT,
                MKL.MKL_INT,
                prec_type,
                ndpointer(dtype=prec_type, ndim=2),
                MKL.MKL_INT,
                ndpointer(dtype=prec_type, ndim=2),
                MKL.MKL_INT,
                prec_type,
                _ctypes.POINTER(prec_type),
                MKL.MKL_INT]

    @staticmethod
    def _mkl_sparse_spmmd_argtypes(prec_type):
        return [_ctypes.c_int,
                sparse_matrix_t,
                sparse_matrix_t,
                _ctypes.c_int,
                _ctypes.POINTER(prec_type), MKL.MKL_INT]

    @staticmethod
    def _mkl_sparse_mm_argtypes(prec_type):
        return [_ctypes.c_int,
                prec_type,
                sparse_matrix_t,
                matrix_descr,
                _ctypes.c_int,
                ndpointer(dtype=prec_type, ndim=2),
                MKL.MKL_INT,
                MKL.MKL_INT,
                prec_type,
                _ctypes.POINTER(prec_type),
                MKL.MKL_INT]

    @staticmethod
    def _mkl_sparse_mv_argtypes(prec_type):
        return [_ctypes.c_int,
                prec_type,
                sparse_matrix_t,
                matrix_descr,
                ndpointer(dtype=prec_type, ndim=1),
                prec_type,
                _ctypes.POINTER(prec_type)]

    @staticmethod
    def _mkl_sparse_syrkd_argtypes(prec_type):
        return [_ctypes.c_int,
                sparse_matrix_t,
                prec_type,
                prec_type,
                _ctypes.POINTER(prec_type),
                _ctypes.c_int,
                MKL.MKL_INT]

    @staticmethod
    def _cblas_syrk_argtypes(prec_type):
        return [_ctypes.c_int,
                _ctypes.c_int,
                _ctypes.c_int,
                MKL.MKL_INT,
                MKL.MKL_INT,
                prec_type,
                ndpointer(dtype=prec_type, ndim=2),
                MKL.MKL_INT,
                prec_type,
                _ctypes.POINTER(prec_type),
                MKL.MKL_INT]

    @staticmethod
    def _mkl_sparse_qr_solve(prec_type):
        return [_ctypes.c_int,
                sparse_matrix_t,
                _ctypes.POINTER(prec_type),
                _ctypes.c_int,
                MKL.MKL_INT,
                _ctypes.POINTER(prec_type),
                MKL.MKL_INT,
                ndpointer(dtype=prec_type, ndim=2),
                MKL.MKL_INT]


# Construct opaque struct & type
class _sparse_matrix(_ctypes.Structure):
    pass


sparse_matrix_t = _ctypes.POINTER(_sparse_matrix)


# Matrix description struct
class matrix_descr(_ctypes.Structure):
    _fields_ = [("sparse_matrix_type_t", _ctypes.c_int),
                ("sparse_fill_mode_t", _ctypes.c_int),
                ("sparse_diag_type_t", _ctypes.c_int)]

    def __init__(self, sparse_matrix_type_t=20, sparse_fill_mode_t=0, sparse_diag_type_t=0):
        super(matrix_descr, self).__init__(sparse_matrix_type_t, sparse_fill_mode_t, sparse_diag_type_t)


# Define standard return codes
RETURN_CODES = {0: "SPARSE_STATUS_SUCCESS",
                1: "SPARSE_STATUS_NOT_INITIALIZED",
                2: "SPARSE_STATUS_ALLOC_FAILED",
                3: "SPARSE_STATUS_INVALID_VALUE",
                4: "SPARSE_STATUS_EXECUTION_FAILED",
                5: "SPARSE_STATUS_INTERNAL_ERROR",
                6: "SPARSE_STATUS_NOT_SUPPORTED"}

# Define order codes
LAYOUT_CODE_C = 101
LAYOUT_CODE_F = 102

# Define transpose codes
SPARSE_OPERATION_NON_TRANSPOSE = 10
SPARSE_OPERATION_TRANSPOSE = 11

# Define index codes
SPARSE_INDEX_BASE_ZERO = 0
SPARSE_INDEX_BASE_ONE = 1

# ILP64 message
ILP64_MSG = " Try changing MKL to int64 with the environment variable MKL_INTERFACE_LAYER=ILP64"


def set_debug_mode(debug_bool):
    """
    Activate or deactivate debug mode

    :param debug_bool: True to be printy. False to be quiet.
    :type debug_bool: bool
    """

    MKL.MKL_DEBUG = debug_bool


def print_mkl_debug():
    """
    Print the MKL interface status if debug mode is on
    """

    if not MKL.MKL_DEBUG:
        return

    if get_version_string() is None:
        print("mkl-service must be installed to get full debug messaging")
    else:
        print(get_version_string())

    print("MKL linked: {fn}".format(fn=_libmkl._name))
    print("MKL interface {np} | {c}".format(np=MKL.MKL_INT_NUMPY, c=MKL.MKL_INT))
    print("Set int32 interface with env MKL_INTERFACE_LAYER=LP64")
    print("Set int64 interface with env MKL_INTERFACE_LAYER=ILP64")


def debug_print(msg):
    """
    Print a message if debug mode is on
    :param msg: Message
    :type msg: str
    """

    if not MKL.MKL_DEBUG:
        return
    else:
        print(msg)


def debug_timer(msg=None, old_time=None):
    """
    Print a message with timing information if debug mode is on
    :param msg: Message
    :type msg: str
    :param old_time: Time to calculate difference for
    :type old_time: float
    """

    if not MKL.MKL_DEBUG:
        return

    t0 = time.time()

    if old_time is not None and msg is not None:
        print(msg + ": {0:.6f} seconds".format(t0 - old_time))

    return t0


def _check_scipy_index_typing(sparse_matrix):
    """
    Ensure that the sparse matrix indicies are in the correct integer type

    :param sparse_matrix: Scipy matrix in CSC or CSR format
    :type sparse_matrix: scipy.sparse.spmatrix
    """

    int_max = np.iinfo(MKL.MKL_INT_NUMPY).max
    if (sparse_matrix.nnz > int_max) or (max(sparse_matrix.shape) > int_max):
        msg = "MKL interface is {t} and cannot hold matrix {m}\n".format(m=repr(sparse_matrix), t=MKL.MKL_INT_NUMPY)
        msg += "Try changing MKL to int64 with the environment variable MKL_INTERFACE_LAYER=ILP64"
        raise ValueError(msg)

    # Cast indexes to MKL_INT type
    if sparse_matrix.indptr.dtype != MKL.MKL_INT_NUMPY:
        sparse_matrix.indptr = sparse_matrix.indptr.astype(MKL.MKL_INT_NUMPY)
    if sparse_matrix.indices.dtype != MKL.MKL_INT_NUMPY:
        sparse_matrix.indices = sparse_matrix.indices.astype(MKL.MKL_INT_NUMPY)


def _get_numpy_layout(numpy_arr, second_arr=None):
    """
    Get the array layout code for a dense array in C or F order.
    Raises a ValueError if the array is not contiguous.

    :param numpy_arr: Numpy dense array
    :type numpy_arr: np.ndarray
    :param second_arr: Numpy dense array; if numpy_arr is 1d (and therefore both C and F order), use this order
    :type second_arr: np.ndarray, None
    :return: The layout code for MKL and the leading dimension
    :rtype: int, int
    """

    # Return the second array order if the first is ambiguous
    if numpy_arr.flags.c_contiguous and numpy_arr.flags.f_contiguous and second_arr is not None:
        if second_arr.flags.c_contiguous:
            return LAYOUT_CODE_C, numpy_arr.shape[1]
        elif second_arr.flags.f_contiguous:
            return LAYOUT_CODE_F, numpy_arr.shape[0]

    # Return the first order array otherwise
    elif numpy_arr.flags.c_contiguous:
        return LAYOUT_CODE_C, numpy_arr.shape[1]
    elif numpy_arr.flags.f_contiguous:
        return LAYOUT_CODE_F, numpy_arr.shape[0]
    elif not numpy_arr.flags.contiguous:
        raise ValueError("Array is not contiguous")
    else:
        raise ValueError("Array layout check has failed for unknown reason")


def _create_mkl_sparse(matrix):
    """
    Create MKL internal representation

    :param matrix: Sparse data in CSR or CSC format
    :type matrix: scipy.sparse.spmatrix

    :return ref, double_precision: Handle for the MKL internal representation and boolean for double precision
    :rtype: sparse_matrix_t, float
    """

    double_precision = _is_double(matrix)

    # Figure out which matrix creation function to use
    if _spsparse.isspmatrix_csr(matrix):
        _check_scipy_index_typing(matrix)
        assert matrix.data.shape[0] == matrix.indices.shape[0]
        assert matrix.indptr.shape[0] == matrix.shape[0] + 1
        handle_func = MKL._mkl_sparse_d_create_csr if double_precision else MKL._mkl_sparse_s_create_csr

    elif _spsparse.isspmatrix_csc(matrix):
        _check_scipy_index_typing(matrix)
        assert matrix.data.shape[0] == matrix.indices.shape[0]
        assert matrix.indptr.shape[0] == matrix.shape[1] + 1
        handle_func = MKL._mkl_sparse_d_create_csc if double_precision else MKL._mkl_sparse_s_create_csc

    elif _spsparse.isspmatrix_bsr(matrix):
        _check_scipy_index_typing(matrix)
        return _create_mkl_sparse_bsr(matrix), double_precision

    else:
        raise ValueError("Matrix is not CSC, CSR, or BSR")

    return _pass_mkl_handle_csr_csc(matrix, handle_func), double_precision


def _pass_mkl_handle_csr_csc(data, handle_func):
    """
    Create MKL internal representation for CSR or CSC matrix

    :param data: Sparse data
    :type data: scipy.sparse.spmatrix
    :return ref: Handle for the MKL internal representation
    :rtype: sparse_matrix_t
    """

    # Create a pointer for the output matrix
    ref = sparse_matrix_t()

    # Load into a MKL data structure and check return
    ret_val = handle_func(_ctypes.byref(ref),
                          _ctypes.c_int(SPARSE_INDEX_BASE_ZERO),
                          MKL.MKL_INT(data.shape[0]),
                          MKL.MKL_INT(data.shape[1]),
                          data.indptr[0:-1],
                          data.indptr[1:],
                          data.indices,
                          data.data)

    # Check return
    _check_return_value(ret_val, handle_func.__name__)

    return ref


def _create_mkl_sparse_bsr(matrix):
    """
    Create MKL internal representation for BSR matrix

    :param matrix: Sparse data
    :type matrix: scipy.sparse.bsr_matrix
    :return ref: Handle for the MKL internal representation
    :rtype: sparse_matrix_t
    """

    double_precision = _is_double(matrix)
    handle_func = MKL._mkl_sparse_d_create_bsr if double_precision else MKL._mkl_sparse_s_create_bsr

    # Get the blocksize and check that the blocks are square
    _blocksize = matrix.blocksize[0]

    if _blocksize != matrix.blocksize[1]:
        _err = "MKL BSR representation requires square blocks; {n} blocks provided".format(n=matrix.blocksize)
        raise ValueError(_err)

    if (matrix.shape[0] % _blocksize != 0) or (matrix.shape[1] % _blocksize != 0):
        _err = "BSR blocks {n} do not align with dims {m}".format(n=matrix.blocksize, m=matrix.shape)
        raise ValueError(_err)

    _block_rows = int(matrix.shape[0] / _blocksize)
    _block_cols = int(matrix.shape[1] / _blocksize)

    # Get the data block array structure
    _layout, _ = _get_numpy_layout(matrix.data)

    # Create a pointer for the output matrix
    ref = sparse_matrix_t()

    # Load into a MKL data structure and check return
    ret_val = handle_func(_ctypes.byref(ref),
                          _ctypes.c_int(SPARSE_INDEX_BASE_ZERO),
                          _ctypes.c_int(_layout),
                          MKL.MKL_INT(_block_rows),
                          MKL.MKL_INT(_block_cols),
                          MKL.MKL_INT(_blocksize),
                          matrix.indptr[0:-1],
                          matrix.indptr[1:],
                          matrix.indices,
                          matrix.data)

    # Check return
    _check_return_value(ret_val, handle_func.__name__)

    return ref


def _export_mkl(csr_mkl_handle, double_precision, output_type="csr"):
    """
    Export a MKL sparse handle of CSR or CSC type

    :param csr_mkl_handle: Handle for the MKL internal representation
    :type csr_mkl_handle: sparse_matrix_t
    :param double_precision: Use float64 if True, float32 if False. This MUST match the underlying float type - this
        defines a memory view, it does not cast.
    :type double_precision: bool
    :param output_type: The structure of the MKL handle (and therefore the type of scipy sparse to create)
    :type output_type: str

    :return: Sparse matrix in scipy format
    :rtype: scipy.spmatrix
    """

    output_type = output_type.lower()

    if output_type == "csr":
        out_func = MKL._mkl_sparse_d_export_csr if double_precision else MKL._mkl_sparse_s_export_csr
        sp_matrix_constructor = _spsparse.csr_matrix
    elif output_type == "csc":
        out_func = MKL._mkl_sparse_d_export_csc if double_precision else MKL._mkl_sparse_s_export_csc
        sp_matrix_constructor = _spsparse.csc_matrix
    elif output_type == "bsr":
        return _export_mkl_sparse_bsr(csr_mkl_handle, double_precision)
    else:
        raise ValueError("Only CSR, CSC, and BSR output types are supported")

    # Allocate for output
    ordering, nrows, ncols, indptrb, indptren, indices, data = _allocate_for_export(double_precision)
    final_dtype = np.float64 if double_precision else np.float32

    ret_val = out_func(csr_mkl_handle,
                       _ctypes.byref(ordering),
                       _ctypes.byref(nrows),
                       _ctypes.byref(ncols),
                       _ctypes.byref(indptrb),
                       _ctypes.byref(indptren),
                       _ctypes.byref(indices),
                       _ctypes.byref(data))

    # Check return
    _check_return_value(ret_val, out_func.__name__)

    # Check ordering
    if ordering.value != 0:
        raise ValueError("1-indexing (F-style) is not supported")

    # Get matrix dims
    ncols, nrows = ncols.value, nrows.value

    # If any axis is 0 return an empty matrix
    if nrows == 0 or ncols == 0:
        return sp_matrix_constructor((nrows, ncols), dtype=final_dtype)

    # Get the index dimension
    index_dim = nrows if output_type == "csr" else ncols

    # Construct a numpy array and add 0 to first position for scipy.sparse's 3-array indexing
    indptrb = as_array(indptrb, shape=(index_dim,))
    indptren = as_array(indptren, shape=(index_dim,))

    indptren = np.insert(indptren, 0, indptrb[0])
    nnz = indptren[-1] - indptrb[0]

    # If there are no non-zeros, return an empty matrix
    # If the number of non-zeros is insane, raise a ValueError
    if nnz == 0:
        return sp_matrix_constructor((nrows, ncols), dtype=final_dtype)
    elif nnz < 0 or nnz > ncols * nrows:
        raise ValueError("Matrix ({m} x {n}) is attempting to index {z} elements".format(m=nrows, n=ncols, z=nnz))

    # Construct numpy arrays from data pointer and from indicies pointer
    data = np.array(as_array(data, shape=(nnz,)), copy=True)
    indices = np.array(as_array(indices, shape=(nnz,)), copy=True)

    # Pack and return the matrix
    return sp_matrix_constructor((data, indices, indptren), shape=(nrows, ncols))


def _export_mkl_sparse_bsr(bsr_mkl_handle, double_precision):
    """
    Export a BSR matrix from MKL's internal representation to scipy

    :param bsr_mkl_handle: MKL internal representation
    :type bsr_mkl_handle: sparse_matrix_t
    :param double_precision: Use float64 if True, float32 if False. This MUST match the underlying float type - this
        defines a memory view, it does not cast.
    :type double_precision: bool
    :return: Sparse BSR matrix
    :rtype:
    """

    # Allocate for output
    ordering, nrows, ncols, indptrb, indptren, indices, data = _allocate_for_export(double_precision)
    block_layout = _ctypes.c_int()
    block_size = MKL.MKL_INT()

    # Set output
    out_func = MKL._mkl_sparse_d_export_bsr if double_precision else MKL._mkl_sparse_s_export_bsr
    final_dtype = np.float64 if double_precision else np.float32

    ret_val = out_func(bsr_mkl_handle,
                       _ctypes.byref(ordering),
                       _ctypes.byref(block_layout),
                       _ctypes.byref(nrows),
                       _ctypes.byref(ncols),
                       _ctypes.byref(block_size),
                       _ctypes.byref(indptrb),
                       _ctypes.byref(indptren),
                       _ctypes.byref(indices),
                       _ctypes.byref(data))

    # Check return
    _check_return_value(ret_val, out_func.__name__)

    # Get matrix dims
    ncols, nrows, block_size = ncols.value, nrows.value, block_size.value
    index_dim, block_dims = nrows, (block_size, block_size)
    ncols, nrows = ncols * block_size, nrows * block_size

    # If any axis is 0 return an empty matrix
    if nrows == 0 or ncols == 0:
        return _spsparse.bsr_matrix((nrows, ncols), dtype=final_dtype, blocksize=block_dims)

    ordering = "F" if ordering.value == LAYOUT_CODE_F else "C"

    # Construct a numpy array and add 0 to first position for scipy.sparse's 3-array indexing
    indptrb = as_array(indptrb, shape=(index_dim,))
    indptren = as_array(indptren, shape=(index_dim,))

    indptren = np.insert(indptren, 0, indptrb[0])

    nnz_blocks = (indptren[-1] - indptrb[0])

    # If there's no non-zero data, return an empty matrix
    if nnz_blocks == 0:
        return _spsparse.bsr_matrix((nrows, ncols), dtype=final_dtype, blocksize=block_dims)
    elif nnz_blocks < 0 or (nnz_blocks * (block_size ** 2)) > ncols * nrows:
        nnz = nnz_blocks * (block_size ** 2)
        _err = "Matrix ({m} x {n}) is attempting to index {z} elements as {b} {bs} blocks"
        _err = _err.format(m=nrows, n=ncols, z=nnz, b=nnz_blocks, bs=(block_size, block_size))
        raise ValueError(_err)

    data = np.array(as_array(data, shape=(nnz_blocks, block_size, block_size)), copy=True, order=ordering,
                    dtype=final_dtype)
    indices = np.array(as_array(indices, shape=(nnz_blocks,)), copy=True)

    return _spsparse.bsr_matrix((data, indices, indptren), shape=(nrows, ncols), blocksize=block_dims)


def _allocate_for_export(double_precision):
    """
    Get pointers for output from MKL internal representation
    :param double_precision: Allocate an output pointer of doubles
    :type double_precision: bool
    :return: ordering, nrows, ncols, indptrb, indptren, indices, data
    :rtype: c_int, MKL_INT, MKL_INT, MKL_INT*, MKL_INT*, MKL_INT*, c_float|c_double*
    """
    # Create the pointers for the output data
    indptrb = _ctypes.POINTER(MKL.MKL_INT)()
    indptren = _ctypes.POINTER(MKL.MKL_INT)()
    indices = _ctypes.POINTER(MKL.MKL_INT)()

    ordering = _ctypes.c_int()
    nrows = MKL.MKL_INT()
    ncols = MKL.MKL_INT()

    data = _ctypes.POINTER(_ctypes.c_double)() if double_precision else _ctypes.POINTER(_ctypes.c_float)()

    return ordering, nrows, ncols, indptrb, indptren, indices, data


def _check_return_value(ret_val, func_name):
    """
    Check the return value from a sparse function

    :param ret_val:
    :param func_name:
    :return:
    """

    if ret_val != 0:
        err_msg = "{fn} returned {v} ({e})".format(fn=func_name, v=ret_val, e=RETURN_CODES[ret_val])
        if ret_val == 2:
            err_msg += "; " + ILP64_MSG
        raise ValueError(err_msg)
    elif MKL.MKL_DEBUG:
        print("{fn} returned {v} ({e})".format(fn=func_name, v=ret_val, e=RETURN_CODES[ret_val]))
    else:
        return


def _destroy_mkl_handle(ref_handle):
    """
    Deallocate a MKL sparse handle

    :param ref_handle:
    :type ref_handle: sparse_matrix_t
    """

    ret_val = MKL._mkl_sparse_destroy(ref_handle)
    _check_return_value(ret_val, "mkl_sparse_destroy")


def _order_mkl_handle(ref_handle):
    """
    Reorder indexes in a MKL sparse handle

    :param ref_handle:
    :type ref_handle: sparse_matrix_t
    """

    ret_val = MKL._mkl_sparse_order(ref_handle)
    _check_return_value(ret_val, "mkl_sparse_order")


def _convert_to_csr(ref_handle, destroy_original=False):
    """
    Convert a MKL sparse handle to CSR format

    :param ref_handle:
    :type ref_handle: sparse_matrix_t
    :return:
    """

    csr_ref = sparse_matrix_t()
    ret_val = MKL._mkl_sparse_convert_csr(ref_handle, _ctypes.c_int(10), _ctypes.byref(csr_ref))

    try:
        _check_return_value(ret_val, "mkl_sparse_convert_csr")
    except ValueError:
        try:
            _destroy_mkl_handle(csr_ref)
        except ValueError:
            pass

        raise

    if destroy_original:
        _destroy_mkl_handle(ref_handle)

    return csr_ref


def _sanity_check(matrix_a, matrix_b, allow_vector=False):
    """
    Check matrix dimensions
    :param matrix_a: sp.sparse or numpy array
    :param matrix_b: sp.sparse or numpy array
    """

    a_2d, b_2d = matrix_a.ndim == 2, matrix_b.ndim == 2
    a_vec, b_vec = _is_dense_vector(matrix_a), _is_dense_vector(matrix_b)

    # Check to make sure that both matrices are 2-d
    if not allow_vector and (not a_2d or not b_2d):
        err_msg = "Matrices must be 2d: {m1} * {m2} is not valid".format(m1=matrix_a.shape, m2=matrix_b.shape)
        raise ValueError(err_msg)

    invalid_ndims = not (a_2d or a_vec) or not (b_2d or b_vec)
    invalid_align = (matrix_a.shape[1] if not matrix_a.ndim == 1 else matrix_a.shape[0]) != matrix_b.shape[0]

    # Check to make sure that this multiplication can work
    if invalid_align or invalid_ndims:
        err_msg = "Matrix alignment error: {m1} * {m2} is not valid".format(m1=matrix_a.shape, m2=matrix_b.shape)
        raise ValueError(err_msg)


def _cast_to_float64(matrix):
    """ Make a copy of the array as double precision floats or return the reference if it already is"""
    return matrix.astype(np.float64) if matrix.dtype != np.float64 else matrix


def _type_check(matrix_a, matrix_b=None, cast=False):
    """
    Make sure that both matrices are single precision floats or both are double precision floats
    If not, convert to double precision floats if cast is True, or raise an error if cast is False
    """

    if matrix_b is None and matrix_a.dtype in NUMPY_FLOAT_DTYPES:
        return matrix_a
    elif matrix_b is None and cast:
        return _cast_to_float64(matrix_a)
    elif matrix_b is None:
        err_msg = "Matrix data type must be float32 or float64; {a} provided".format(a=matrix_a.dtype)
        raise ValueError(err_msg)

    # Check dtypes
    if matrix_a.dtype == np.float32 and matrix_b.dtype == np.float32:
        return matrix_a, matrix_b

    elif matrix_a.dtype == np.float64 and matrix_b.dtype == np.float64:
        return matrix_a, matrix_b

    elif (matrix_a.dtype != np.float64 or matrix_b.dtype != np.float64) and cast:
        debug_print("Recasting matrix data types {a} and {b} to np.float64".format(a=matrix_a.dtype, b=matrix_b.dtype))
        return _cast_to_float64(matrix_a), _cast_to_float64(matrix_b)

    elif matrix_a.dtype != np.float64 or matrix_b.dtype != np.float64:
        err_msg = "Matrix data types must be in concordance; {a} and {b} provided".format(a=matrix_a.dtype,
                                                                                          b=matrix_b.dtype)
        raise ValueError(err_msg)


def _out_matrix(shape, dtype, order="C", out_arr=None, out_t=False):
    """
    Create an all-zero matrix or check to make sure that the provided output array matches

    :param shape: Required output shape
    :type shape: tuple(int)
    :param dtype: Required output data type
    :type dtype: np.dtype
    :param order: Array order (row or column-major)
    :type order: str
    :param out_arr: Provided output array
    :type out_arr: np.ndarray
    :param out_t: Out array has been transposed 
    :type out_t: bool
    :return: Array
    :rtype: np.ndarray
    """

    out_t = False if out_t is None else out_t

    # If there's no output array allocate a new array and return it
    if out_arr is None:
        return np.zeros(shape, dtype=dtype, order=order)

    # Check and make sure the order is correct
    # Note 1d arrays have both flags set
    _order_match = out_arr.flags['C_CONTIGUOUS'] if order == "C" else out_arr.flags['F_CONTIGUOUS']

    # If there are any incompatible parameters, raise an error with the provided and required array parameters
    # Flip them if out_T is set so that the original values and the values which would have to be provided are correct
    if shape != out_arr.shape or dtype != out_arr.dtype or not _order_match or not out_arr.data.contiguous:
        if not out_t or out_arr.ndim == 1:
            _err_shape, _req_shape = out_arr.shape, shape
            _err_order, _req_order = "C" if out_arr.flags['C_CONTIGUOUS'] else "F", order
        else:
            _err_shape, _req_shape = out_arr.shape[::-1], shape[::-1]
            _err_order = "F" if out_arr.flags['C_CONTIGUOUS'] and not out_arr.flags['F_CONTIGUOUS'] else "C"
            _req_order = "C" if order == "F" else "F"

        try:
            _req_dtype = dtype.__name__
        except AttributeError:
            _req_dtype = dtype.name

        _err_msg = "Provided out array is "
        _err_msg += "{s} {d} [{o}_{c}]".format(s=_err_shape, d=out_arr.dtype, o=_err_order,
                                               c="CONTIGUOUS" if out_arr.data.contiguous else "NONCONTIGUOUS")

        _err_msg += "; product requires {s} {d} [{o}_{c}]".format(s=_req_shape, d=_req_dtype, o=_req_order,
                                                                  c="CONTIGUOUS")
        raise ValueError(_err_msg)

    else:
        return out_arr


def _is_dense_vector(m_or_v):
    return not _spsparse.issparse(m_or_v) and ((m_or_v.ndim == 1) or ((m_or_v.ndim == 2) and min(m_or_v.shape) == 1))


def _is_double(arr):
    """
    Return true if the array is doubles, false if singles, and raise an error if it's neither.

    :param arr:
    :type arr: np.ndarray, scipy.sparse.spmatrix
    :return:
    :rtype: bool
    """

    # Figure out which dtype for data
    if arr.dtype == np.float32:
        return False
    elif arr.dtype == np.float64:
        return True
    else:
        raise ValueError("Only float32 or float64 dtypes are supported")


def _is_allowed_sparse_format(matrix):
    """
    Return True if the matrix is dense or a sparse format we can turn into an MKL object. False otherwise.
    :param matrix:
    :return:
    :rtype: bool
    """
    if _spsparse.isspmatrix(matrix):
        return _spsparse.isspmatrix_csr(matrix) or _spsparse.isspmatrix_csc(matrix) or _spsparse.isspmatrix_bsr(matrix)
    else:
        return True


def _empty_output_check(matrix_a, matrix_b):
    """Check for trivial cases where an empty array should be produced"""

    # One dimension is zero
    if min([*matrix_a.shape, *matrix_b.shape]) == 0:
        return True

    # The sparse array is empty
    elif _spsparse.issparse(matrix_a) and min(matrix_a.data.size, matrix_a.indices.size) == 0:
        return True
    elif _spsparse.issparse(matrix_b) and min(matrix_b.data.size, matrix_b.indices.size) == 0:
        return True

    # Neither trivial condition
    else:
        return False


def _validate_dtype():
    """
    Test to make sure that this library works by creating a random sparse array in CSC format,
    then converting it to CSR format and making sure is has not raised an exception.

    """

    test_array = _spsparse.random(5, 5, density=0.5, format="csc", dtype=np.float32, random_state=50)
    test_comparison = test_array.A

    csc_ref, precision_flag = _create_mkl_sparse(test_array)

    try:
        csr_ref = _convert_to_csr(csc_ref)
        final_array = _export_mkl(csr_ref, precision_flag)
        if not np.allclose(test_comparison, final_array.A):
            raise ValueError("Match failed after matrix conversion")
        _destroy_mkl_handle(csr_ref)
    finally:
        _destroy_mkl_handle(csc_ref)


def _empirical_set_dtype():
    """
    Define dtypes empirically
    Basically just try with int64s and if that doesn't work try with int32s
    There's a way to do this with intel's mkl helper package but I don't want to add the dependency
    """
    MKL._set_int_type(_ctypes.c_longlong, np.int64)

    try:
        _validate_dtype()
    except ValueError as err:

        MKL._set_int_type(_ctypes.c_int, np.int32)

        try:
            _validate_dtype()
        except ValueError:
            raise ImportError("Unable to set MKL numeric type")


if MKL.MKL_INT is None:
    _empirical_set_dtype()

if _sklearn_env is not None:
    os.environ['KMP_INIT_AT_FORK'] = _sklearn_env
