import ctypes as _ctypes
import ctypes.util as _ctypes_util

from sparse_dot_mkl._mkl_interface._structs import sparse_matrix_t, matrix_descr, MKL_Complex8, MKL_Complex16

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

import numpy as _np
from numpy.ctypeslib import ndpointer

def mkl_library_name():
    return _libmkl._name

class MKL:
    """ This class holds shared object references to C functions with arg and returntypes that can be adjusted"""

    MKL_INT = None
    MKL_INT_NUMPY = None
    MKL_DEBUG = False

    # Import function for creating a MKL CSR object
    _mkl_sparse_d_create_csr = _libmkl.mkl_sparse_d_create_csr
    _mkl_sparse_s_create_csr = _libmkl.mkl_sparse_s_create_csr
    _mkl_sparse_c_create_csr = _libmkl.mkl_sparse_c_create_csr
    _mkl_sparse_z_create_csr = _libmkl.mkl_sparse_z_create_csr

    # Import function for creating a MKL CSC object
    _mkl_sparse_d_create_csc = _libmkl.mkl_sparse_d_create_csc
    _mkl_sparse_s_create_csc = _libmkl.mkl_sparse_s_create_csc
    _mkl_sparse_c_create_csc = _libmkl.mkl_sparse_c_create_csc
    _mkl_sparse_z_create_csc = _libmkl.mkl_sparse_z_create_csc

    # Import function for creating a MKL BSR object
    _mkl_sparse_d_create_bsr = _libmkl.mkl_sparse_d_create_bsr
    _mkl_sparse_s_create_bsr = _libmkl.mkl_sparse_s_create_bsr
    _mkl_sparse_c_create_bsr = _libmkl.mkl_sparse_c_create_bsr
    _mkl_sparse_z_create_bsr = _libmkl.mkl_sparse_z_create_bsr

    # Export function for exporting a MKL CSR object
    _mkl_sparse_d_export_csr = _libmkl.mkl_sparse_d_export_csr
    _mkl_sparse_s_export_csr = _libmkl.mkl_sparse_s_export_csr
    _mkl_sparse_c_export_csr = _libmkl.mkl_sparse_c_export_csr
    _mkl_sparse_z_export_csr = _libmkl.mkl_sparse_z_export_csr

    # Export function for exporting a MKL CSC object
    _mkl_sparse_d_export_csc = _libmkl.mkl_sparse_d_export_csc
    _mkl_sparse_s_export_csc = _libmkl.mkl_sparse_s_export_csc
    _mkl_sparse_z_export_csc = _libmkl.mkl_sparse_z_export_csc
    _mkl_sparse_c_export_csc = _libmkl.mkl_sparse_c_export_csc

    # Export function for exporting a MKL BSR object
    _mkl_sparse_d_export_bsr = _libmkl.mkl_sparse_d_export_bsr
    _mkl_sparse_s_export_bsr = _libmkl.mkl_sparse_s_export_bsr
    _mkl_sparse_c_export_bsr = _libmkl.mkl_sparse_c_export_bsr
    _mkl_sparse_z_export_bsr = _libmkl.mkl_sparse_z_export_bsr

    # Import function for matmul
    _mkl_sparse_spmm = _libmkl.mkl_sparse_spmm

    # Import function for cleaning up MKL objects
    _mkl_sparse_destroy = _libmkl.mkl_sparse_destroy

    # Import function for ordering MKL objects
    _mkl_sparse_order = _libmkl.mkl_sparse_order

    # Import function for coverting to CSR
    _mkl_sparse_convert_csr = _libmkl.mkl_sparse_convert_csr

    # Import function for matmul single dense
    _mkl_sparse_s_spmmd = _libmkl.mkl_sparse_s_spmmd
    _mkl_sparse_d_spmmd = _libmkl.mkl_sparse_d_spmmd
    _mkl_sparse_c_spmmd = _libmkl.mkl_sparse_c_spmmd
    _mkl_sparse_z_spmmd = _libmkl.mkl_sparse_z_spmmd

    # Import function for matmul single sparse*dense
    _mkl_sparse_s_mm = _libmkl.mkl_sparse_s_mm
    _mkl_sparse_d_mm = _libmkl.mkl_sparse_d_mm
    _mkl_sparse_c_mm = _libmkl.mkl_sparse_c_mm
    _mkl_sparse_z_mm = _libmkl.mkl_sparse_z_mm

    # Import function for matmul dense*dense
    _cblas_sgemm = _libmkl.cblas_sgemm
    _cblas_dgemm = _libmkl.cblas_dgemm
    _cblas_cgemm = _libmkl.cblas_cgemm
    _cblas_zgemm = _libmkl.cblas_zgemm

    # Import function for matrix * vector
    _mkl_sparse_s_mv = _libmkl.mkl_sparse_s_mv
    _mkl_sparse_d_mv = _libmkl.mkl_sparse_d_mv
    _mkl_sparse_c_mv = _libmkl.mkl_sparse_c_mv
    _mkl_sparse_z_mv = _libmkl.mkl_sparse_z_mv
    
    # Import function for sparse gram matrix
    _mkl_sparse_syrk = _libmkl.mkl_sparse_syrk

    # Import function for dense single gram matrix from sparse
    _mkl_sparse_s_syrkd = _libmkl.mkl_sparse_s_syrkd
    _mkl_sparse_d_syrkd = _libmkl.mkl_sparse_d_syrkd
    _mkl_sparse_c_syrkd = _libmkl.mkl_sparse_c_syrkd
    _mkl_sparse_z_syrkd = _libmkl.mkl_sparse_z_syrkd

    # Import function for dense single gram matrix
    _cblas_ssyrk = _libmkl.cblas_ssyrk
    _cblas_dsyrk = _libmkl.cblas_dsyrk
    _cblas_csyrk = _libmkl.cblas_csyrk
    _cblas_zsyrk = _libmkl.cblas_zsyrk

    # Import function for QR solver - reorder
    _mkl_sparse_qr_reorder = _libmkl.mkl_sparse_qr_reorder

    # Import function for QR solver - factorize
    _mkl_sparse_d_qr_factorize = _libmkl.mkl_sparse_d_qr_factorize
    _mkl_sparse_s_qr_factorize = _libmkl.mkl_sparse_s_qr_factorize

    # Import function for QR solver - solve
    _mkl_sparse_d_qr_solve = _libmkl.mkl_sparse_d_qr_solve
    _mkl_sparse_s_qr_solve = _libmkl.mkl_sparse_s_qr_solve

    @classmethod
    def _set_int_type(cls, c_type, _np_type):
        cls.MKL_INT = c_type
        cls.MKL_INT_NUMPY = _np_type

        cls._set_int_type_create()
        cls._set_int_type_export()
        cls._set_int_type_sparse_matmul()
        cls._set_int_type_dense_matmul()
        cls._set_int_type_vector_mul()
        cls._set_int_type_qr_solver()
        cls._set_int_type_misc()

    @classmethod
    def _set_int_type_create(cls):
        """Set the correct argtypes for handle creation functions"""
        cls._mkl_sparse_d_create_csr.argtypes = cls._mkl_sparse_create_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_create_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_s_create_csr.argtypes = cls._mkl_sparse_create_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_create_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_c_create_csr.argtypes = cls._mkl_sparse_create_argtypes(_np.csingle)
        cls._mkl_sparse_c_create_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_z_create_csr.argtypes = cls._mkl_sparse_create_argtypes(_np.cdouble)
        cls._mkl_sparse_z_create_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_d_create_csc.argtypes = cls._mkl_sparse_create_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_create_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_s_create_csc.argtypes = cls._mkl_sparse_create_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_create_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_c_create_csc.argtypes = cls._mkl_sparse_create_argtypes(_np.csingle)
        cls._mkl_sparse_c_create_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_z_create_csc.argtypes = cls._mkl_sparse_create_argtypes(_np.cdouble)
        cls._mkl_sparse_z_create_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_d_create_bsr.argtypes = cls._mkl_sparse_create_bsr_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_create_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_s_create_bsr.argtypes = cls._mkl_sparse_create_bsr_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_create_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_c_create_bsr.argtypes = cls._mkl_sparse_create_bsr_argtypes(_np.csingle)
        cls._mkl_sparse_c_create_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_z_create_bsr.argtypes = cls._mkl_sparse_create_bsr_argtypes(_np.cdouble)
        cls._mkl_sparse_z_create_bsr.restypes = _ctypes.c_int

    @classmethod
    def _set_int_type_export(cls):
        """Set the correct argtypes for handle export functions"""
        cls._mkl_sparse_d_export_csr.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_export_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_s_export_csr.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_export_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_z_export_csr.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_double)
        cls._mkl_sparse_z_export_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_c_export_csr.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_float)
        cls._mkl_sparse_c_export_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_csc.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_export_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_d_export_csc.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_export_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_c_export_csc.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_float)
        cls._mkl_sparse_c_export_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_z_export_csc.argtypes = cls._mkl_sparse_export_argtypes(_ctypes.c_double)
        cls._mkl_sparse_z_export_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_bsr.argtypes = cls._mkl_sparse_export_bsr_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_export_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_d_export_bsr.argtypes = cls._mkl_sparse_export_bsr_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_export_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_c_export_bsr.argtypes = cls._mkl_sparse_export_bsr_argtypes(_ctypes.c_float)
        cls._mkl_sparse_c_export_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_z_export_bsr.argtypes = cls._mkl_sparse_export_bsr_argtypes(_ctypes.c_double)
        cls._mkl_sparse_z_export_bsr.restypes = _ctypes.c_int

    @classmethod
    def _set_int_type_sparse_matmul(cls):
        """Set the correct argtypes for sparse (*) sparse functions and sparse (*) dense functions"""
        cls._mkl_sparse_spmm.argtypes = [_ctypes.c_int,
                                         sparse_matrix_t,
                                         sparse_matrix_t,
                                         _ctypes.POINTER(sparse_matrix_t)]
        cls._mkl_sparse_spmm.restypes = _ctypes.c_int

        cls._mkl_sparse_s_spmmd.argtypes = cls._mkl_sparse_spmmd_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_spmmd.restypes = _ctypes.c_int
        cls._mkl_sparse_d_spmmd.argtypes = cls._mkl_sparse_spmmd_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_spmmd.restypes = _ctypes.c_int
        cls._mkl_sparse_c_spmmd.argtypes = cls._mkl_sparse_spmmd_argtypes(_ctypes.c_float)
        cls._mkl_sparse_c_spmmd.restypes = _ctypes.c_int
        cls._mkl_sparse_z_spmmd.argtypes = cls._mkl_sparse_spmmd_argtypes(_ctypes.c_double)
        cls._mkl_sparse_z_spmmd.restypes = _ctypes.c_int

        cls._mkl_sparse_s_mm.argtypes = cls._mkl_sparse_mm_argtypes(_ctypes.c_float, _ctypes.c_float, _ctypes.c_float)
        cls._mkl_sparse_s_mm.restypes = _ctypes.c_int
        cls._mkl_sparse_d_mm.argtypes = cls._mkl_sparse_mm_argtypes(_ctypes.c_double, _ctypes.c_double, _ctypes.c_double)
        cls._mkl_sparse_d_mm.restypes = _ctypes.c_int
        cls._mkl_sparse_c_mm.argtypes = cls._mkl_sparse_mm_argtypes(MKL_Complex8, _ctypes.c_float, _np.csingle)
        cls._mkl_sparse_c_mm.restypes = _ctypes.c_int
        cls._mkl_sparse_z_mm.argtypes = cls._mkl_sparse_mm_argtypes(MKL_Complex16, _ctypes.c_double, _np.cdouble)
        cls._mkl_sparse_z_mm.restypes = _ctypes.c_int

    @classmethod
    def _set_int_type_dense_matmul(cls):
        """Set the correct argtypes for dense (*) dense functions"""
        cls._cblas_sgemm.argtypes = cls._cblas_gemm_argtypes(_ctypes.c_float)
        cls._cblas_sgemm.restypes = None
        cls._cblas_dgemm.argtypes = cls._cblas_gemm_argtypes(_ctypes.c_double)
        cls._cblas_dgemm.restypes = None
        cls._cblas_cgemm.argtypes = cls._cblas_gemm_argtypes(_ctypes.c_void_p)
        cls._cblas_cgemm.restypes = None
        cls._cblas_zgemm.argtypes = cls._cblas_gemm_argtypes(_ctypes.c_void_p)
        cls._cblas_zgemm.restypes = None
    
    @classmethod
    def _set_int_type_vector_mul(cls):
        """Set the correct argtypes for sparse (*) vector functions"""
        cls._mkl_sparse_s_mv.argtypes = cls._mkl_sparse_mv_argtypes(_ctypes.c_float, _np.float32)
        cls._mkl_sparse_s_mv.restypes = _ctypes.c_int
        cls._mkl_sparse_d_mv.argtypes = cls._mkl_sparse_mv_argtypes(_ctypes.c_double, _np.float64)
        cls._mkl_sparse_d_mv.restypes = _ctypes.c_int
        cls._mkl_sparse_c_mv.argtypes = cls._mkl_sparse_mv_argtypes(MKL_Complex8, _np.csingle)
        cls._mkl_sparse_c_mv.restypes = _ctypes.c_int
        cls._mkl_sparse_z_mv.argtypes = cls._mkl_sparse_mv_argtypes(MKL_Complex16, _np.cdouble)
        cls._mkl_sparse_z_mv.restypes = _ctypes.c_int

    @classmethod
    def _set_int_type_misc(cls):
        cls._mkl_sparse_destroy.argtypes = [sparse_matrix_t]
        cls._mkl_sparse_destroy.restypes = _ctypes.c_int

        cls._mkl_sparse_order.argtypes = [sparse_matrix_t]
        cls._mkl_sparse_order.restypes = _ctypes.c_int


    @classmethod
    def _set_int_type_misc(cls):
        cls._mkl_sparse_syrk.argtypes = [_ctypes.c_int,
                                         sparse_matrix_t,
                                         _ctypes.POINTER(sparse_matrix_t)]
        cls._mkl_sparse_syrk.restypes = _ctypes.c_int

        cls._mkl_sparse_s_syrkd.argtypes = cls._mkl_sparse_syrkd_argtypes(_ctypes.c_float, _np.float32)
        cls._mkl_sparse_s_syrkd.restypes = _ctypes.c_int
        cls._mkl_sparse_d_syrkd.argtypes = cls._mkl_sparse_syrkd_argtypes(_ctypes.c_double, _np.float64)
        cls._mkl_sparse_d_syrkd.restypes = _ctypes.c_int
        cls._mkl_sparse_c_syrkd.argtypes = cls._mkl_sparse_syrkd_argtypes(MKL_Complex8, _np.csingle)
        cls._mkl_sparse_c_syrkd.restypes = _ctypes.c_int
        cls._mkl_sparse_z_syrkd.argtypes = cls._mkl_sparse_syrkd_argtypes(MKL_Complex16, _np.cdouble)
        cls._mkl_sparse_z_syrkd.restypes = _ctypes.c_int

        cls._cblas_ssyrk.argtypes = cls._cblas_syrk_argtypes(_ctypes.c_float, _np.float32)
        cls._cblas_ssyrk.restypes = None
        cls._cblas_dsyrk.argtypes = cls._cblas_syrk_argtypes(_ctypes.c_double, _np.float64)
        cls._cblas_dsyrk.restypes = None
        cls._cblas_csyrk.argtypes = cls._cblas_syrk_argtypes(MKL_Complex8, _np.csingle, scalar_pointers=True)
        cls._cblas_csyrk.restypes = None
        cls._cblas_zsyrk.argtypes = cls._cblas_syrk_argtypes(MKL_Complex16, _np.cdouble, scalar_pointers=True)
        cls._cblas_zsyrk.restypes = None


    @classmethod
    def _set_int_type_qr_solver(cls):
        """Set the correct argtypes for QR solver functions"""
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
    def _cblas_gemm_argtypes(prec_type, scalar_pointers=False):
        return [_ctypes.c_int,
                _ctypes.c_int,
                _ctypes.c_int,
                MKL.MKL_INT,
                MKL.MKL_INT,
                MKL.MKL_INT,
                _ctypes.POINTER(prec_type) if scalar_pointers else prec_type,
                _ctypes.POINTER(prec_type),
                MKL.MKL_INT,
                _ctypes.POINTER(prec_type),
                MKL.MKL_INT,
                _ctypes.POINTER(prec_type) if scalar_pointers else prec_type,
                _ctypes.POINTER(prec_type),
                MKL.MKL_INT]

    @staticmethod
    def _mkl_sparse_spmmd_argtypes(prec_type):
        return [_ctypes.c_int,
                sparse_matrix_t,
                sparse_matrix_t,
                _ctypes.c_int,
                _ctypes.POINTER(prec_type), 
                MKL.MKL_INT]

    @staticmethod
    def _mkl_sparse_mm_argtypes(scalar_type, prec_type, _np_prec_type):
        return [_ctypes.c_int,
                scalar_type,
                sparse_matrix_t,
                matrix_descr,
                _ctypes.c_int,
                ndpointer(dtype=_np_prec_type, ndim=2),
                MKL.MKL_INT,
                MKL.MKL_INT,
                scalar_type,
                _ctypes.POINTER(prec_type),
                MKL.MKL_INT]

    @staticmethod
    def _mkl_sparse_mv_argtypes(prec_type, _np_type):
        return [_ctypes.c_int,
                prec_type,
                sparse_matrix_t,
                matrix_descr,
                ndpointer(dtype=_np_type, ndim=1),
                prec_type,
                ndpointer(dtype=_np_type)]

    @staticmethod
    def _mkl_sparse_syrkd_argtypes(prec_type, _np_type):
        return [_ctypes.c_int,
                sparse_matrix_t,
                prec_type,
                prec_type,
                ndpointer(dtype=_np_type),
                _ctypes.c_int,
                MKL.MKL_INT]

    @staticmethod
    def _cblas_syrk_argtypes(prec_type, _np_type, scalar_pointers=False):
        return [_ctypes.c_int,
                _ctypes.c_int,
                _ctypes.c_int,
                MKL.MKL_INT,
                MKL.MKL_INT,
                _ctypes.POINTER(prec_type) if scalar_pointers else prec_type,
                ndpointer(dtype=_np_type, ndim=2),
                MKL.MKL_INT,
                _ctypes.POINTER(prec_type) if scalar_pointers else prec_type,
                ndpointer(dtype=_np_type, ndim=2),
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
