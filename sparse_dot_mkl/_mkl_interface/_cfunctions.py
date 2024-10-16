import os
import ctypes as _ctypes

from sparse_dot_mkl._mkl_interface._structs import (
    sparse_matrix_t,
    matrix_descr,
    MKL_Complex8,
    MKL_Complex16,
    MKLVersion
)

from sparse_dot_mkl._mkl_interface._load_library import (
    mkl_library
)

import numpy as _np
from numpy.ctypeslib import ndpointer

# MKL_SET_INTERFACE_LAYER flags
MKL_INTERFACE_LP64  = 0
MKL_INTERFACE_ILP64 = 1
MKL_INTERFACE_GNU   = 2


_libmkl = mkl_library()


def mkl_library_name():
    return _libmkl._name


class MKL:
    """
    This class holds shared object references to C functions with arg and
    returntypes that can be adjusted
    """

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

    # Set interface function
    _mkl_set_interface_layer = _libmkl.MKL_Set_Interface_Layer
    _mkl_get_max_threads = _libmkl.MKL_Get_Max_Threads
    _mkl_set_num_threads = _libmkl.MKL_Set_Num_Threads
    _mkl_set_num_threads_local = _libmkl.MKL_Set_Num_Threads_Local
    _mkl_get_version = _libmkl.MKL_Get_Version
    _mkl_get_version_string = _libmkl.MKL_Get_Version_String
    _mkl_free_buffers = _libmkl.mkl_free_buffers

    # PARDISO
    _pardisoinit = _libmkl.pardisoinit
    _pardiso = _libmkl.pardiso

    # CG Solver
    _dcg_init = _libmkl.dcg_init
    _dcg_check = _libmkl.dcg_check
    _dcg = _libmkl.dcg
    _dcg_get = _libmkl.dcg_get

    _dcgmrhs_init = _libmkl.dcgmrhs_init
    _dcgmrhs_check = _libmkl.dcgmrhs_check
    _dcgmrhs = _libmkl.dcgmrhs
    _dcgmrhs_get = _libmkl.dcgmrhs_get

    # FGMRES Solver
    _dfgmres_init = _libmkl.dfgmres_init
    _dfgmres_check = _libmkl.dfgmres_check
    _dfgmres = _libmkl.dfgmres
    _dfgmres_get = _libmkl.dfgmres_get

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
        cls._set_int_type_syrk()
        cls._set_int_type_misc()
        cls._set_int_type_pardiso()
        cls._set_int_type_iss()

    @classmethod
    def _set_int_type_iss(cls):
        cls._dcg_init.argtypes = cls._create_iss_argtypes()
        cls._dcg_init.restype = None

        cls._dcg_check.argtypes = cls._create_iss_argtypes()
        cls._dcg_check.restype = None

        cls._dcg.argtypes = cls._create_iss_argtypes()
        cls._dcg.restype = None

        cls._dcg_get.argtypes = cls._create_iss_argtypes() + [
            _ctypes.POINTER(MKL.MKL_INT)
        ]
        cls._dcg_get.restype = None

        cls._dcgmrhs_init.argtypes = cls._create_iss_mrhs_argtypes(
            add_method=True
        )
        cls._dcgmrhs_init.restype = None

        cls._dcgmrhs_check.argtypes = cls._create_iss_mrhs_argtypes()
        cls._dcgmrhs_check.restype = None

        cls._dfgmres.argtypes = cls._create_iss_mrhs_argtypes()
        cls._dfgmres.restype = None

        cls._dfgmres_get.argtypes = cls._create_iss_mrhs_argtypes() + [
            _ctypes.POINTER(MKL.MKL_INT)
        ]
        cls._dfgmres_get.restype = None

        cls._dfgmres_init.argtypes = cls._create_iss_argtypes()
        cls._dfgmres_init.restype = None

        cls._dfgmres_check.argtypes = cls._create_iss_argtypes()
        cls._dfgmres_check.restype = None

        cls._dfgmres.argtypes = cls._create_iss_argtypes()
        cls._dfgmres.restype = None

        cls._dfgmres_get.argtypes = cls._create_iss_argtypes() + [
            _ctypes.POINTER(MKL.MKL_INT)
        ]
        cls._dfgmres_get.restype = None


    @classmethod
    def _set_int_type_pardiso(cls):
        cls._pardiso.argtypes = [
            ndpointer(shape=(64,), dtype=_np.int64),    #pt
            _ctypes.POINTER(MKL.MKL_INT),               #maxfct
            _ctypes.POINTER(MKL.MKL_INT),               #mnum
            _ctypes.POINTER(MKL.MKL_INT),               #mtype
            _ctypes.POINTER(MKL.MKL_INT),               #phase
            _ctypes.POINTER(MKL.MKL_INT),               #n
            ndpointer(ndim=1),                          #a
            ndpointer(dtype=MKL.MKL_INT, ndim=1),       #ia
            ndpointer(dtype=MKL.MKL_INT, ndim=1),       #ja
            ndpointer(dtype=MKL.MKL_INT, ndim=1),       #perm
            _ctypes.POINTER(MKL.MKL_INT),               #nrhs
            ndpointer(shape=(64,), dtype=MKL.MKL_INT),  #iparm
            _ctypes.POINTER(MKL.MKL_INT),               #msglvl
            ndpointer(),                                #b
            ndpointer(flags="C_CONTIGUOUS"),            #x
            _ctypes.POINTER(MKL.MKL_INT)                #error
        ]
        cls._pardiso.restype = None

        cls._pardisoinit.argtypes = [
            ndpointer(shape=(64,), dtype=_np.int64),
            _ctypes.POINTER(MKL.MKL_INT),
            ndpointer(shape=(64,), dtype=MKL.MKL_INT_NUMPY)
        ]
        cls._pardisoinit.restype = None

    @classmethod
    def _set_int_type_create(cls):
        """Set the correct argtypes for handle creation functions"""
        cls._mkl_sparse_d_create_csr.argtypes = cls._create_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_d_create_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_s_create_csr.argtypes = cls._create_argtypes(
            _ctypes.c_float
        )
        cls._mkl_sparse_s_create_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_c_create_csr.argtypes = cls._create_argtypes(
            _np.csingle
        )
        cls._mkl_sparse_c_create_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_z_create_csr.argtypes = cls._create_argtypes(
            _np.cdouble
        )
        cls._mkl_sparse_z_create_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_d_create_csc.argtypes = cls._create_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_d_create_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_s_create_csc.argtypes = cls._create_argtypes(
            _ctypes.c_float
        )
        cls._mkl_sparse_s_create_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_c_create_csc.argtypes = cls._create_argtypes(
            _np.csingle
        )
        cls._mkl_sparse_c_create_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_z_create_csc.argtypes = cls._create_argtypes(
            _np.cdouble
        )
        cls._mkl_sparse_z_create_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_d_create_bsr.argtypes = cls._create_bsr_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_d_create_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_s_create_bsr.argtypes = cls._create_bsr_argtypes(
            _ctypes.c_float
        )
        cls._mkl_sparse_s_create_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_c_create_bsr.argtypes = cls._create_bsr_argtypes(
            _np.csingle
        )
        cls._mkl_sparse_c_create_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_z_create_bsr.argtypes = cls._create_bsr_argtypes(
            _np.cdouble
        )
        cls._mkl_sparse_z_create_bsr.restypes = _ctypes.c_int

    @classmethod
    def _set_int_type_export(cls):
        """Set the correct argtypes for handle export functions"""
        cls._mkl_sparse_d_export_csr.argtypes = cls._export_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_d_export_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_s_export_csr.argtypes = cls._export_argtypes(
            _ctypes.c_float
        )
        cls._mkl_sparse_s_export_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_z_export_csr.argtypes = cls._export_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_z_export_csr.restypes = _ctypes.c_int
        cls._mkl_sparse_c_export_csr.argtypes = cls._export_argtypes(
            _ctypes.c_float
        )
        cls._mkl_sparse_c_export_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_csc.argtypes = cls._export_argtypes(
            _ctypes.c_float
        )
        cls._mkl_sparse_s_export_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_d_export_csc.argtypes = cls._export_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_d_export_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_c_export_csc.argtypes = cls._export_argtypes(
            _ctypes.c_float
        )
        cls._mkl_sparse_c_export_csc.restypes = _ctypes.c_int
        cls._mkl_sparse_z_export_csc.argtypes = cls._export_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_z_export_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_bsr.argtypes = cls._export_bsr_argtypes(
            _ctypes.c_float
        )
        cls._mkl_sparse_s_export_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_d_export_bsr.argtypes = cls._export_bsr_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_d_export_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_c_export_bsr.argtypes = cls._export_bsr_argtypes(
            _ctypes.c_float
        )
        cls._mkl_sparse_c_export_bsr.restypes = _ctypes.c_int
        cls._mkl_sparse_z_export_bsr.argtypes = cls._export_bsr_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_z_export_bsr.restypes = _ctypes.c_int

    @classmethod
    def _set_int_type_sparse_matmul(cls):
        """
        Set the correct argtypes for sparse (*) sparse functions and
        sparse (*) dense functions
        """
        cls._mkl_sparse_spmm.argtypes = [
            _ctypes.c_int,
            sparse_matrix_t,
            sparse_matrix_t,
            _ctypes.POINTER(sparse_matrix_t),
        ]
        cls._mkl_sparse_spmm.restypes = _ctypes.c_int

        cls._mkl_sparse_s_spmmd.argtypes = cls._spmmd_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_spmmd.restypes = _ctypes.c_int
        cls._mkl_sparse_d_spmmd.argtypes = cls._spmmd_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_d_spmmd.restypes = _ctypes.c_int
        cls._mkl_sparse_c_spmmd.argtypes = cls._spmmd_argtypes(_ctypes.c_float)
        cls._mkl_sparse_c_spmmd.restypes = _ctypes.c_int
        cls._mkl_sparse_z_spmmd.argtypes = cls._spmmd_argtypes(
            _ctypes.c_double
        )
        cls._mkl_sparse_z_spmmd.restypes = _ctypes.c_int

        cls._mkl_sparse_s_mm.argtypes = cls._mm_argtypes(
            _ctypes.c_float, _ctypes.c_float, _ctypes.c_float
        )
        cls._mkl_sparse_s_mm.restypes = _ctypes.c_int
        cls._mkl_sparse_d_mm.argtypes = cls._mm_argtypes(
            _ctypes.c_double, _ctypes.c_double, _ctypes.c_double
        )
        cls._mkl_sparse_d_mm.restypes = _ctypes.c_int
        cls._mkl_sparse_c_mm.argtypes = cls._mm_argtypes(
            MKL_Complex8, _ctypes.c_float, _np.csingle
        )
        cls._mkl_sparse_c_mm.restypes = _ctypes.c_int
        cls._mkl_sparse_z_mm.argtypes = cls._mm_argtypes(
            MKL_Complex16, _ctypes.c_double, _np.cdouble
        )
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
        cls._mkl_sparse_s_mv.argtypes = cls._mv_argtypes(
            _ctypes.c_float, _np.float32
        )
        cls._mkl_sparse_s_mv.restypes = _ctypes.c_int
        cls._mkl_sparse_d_mv.argtypes = cls._mv_argtypes(
            _ctypes.c_double, _np.float64
        )
        cls._mkl_sparse_d_mv.restypes = _ctypes.c_int
        cls._mkl_sparse_c_mv.argtypes = cls._mv_argtypes(
            MKL_Complex8, _np.csingle
        )
        cls._mkl_sparse_c_mv.restypes = _ctypes.c_int
        cls._mkl_sparse_z_mv.argtypes = cls._mv_argtypes(
            MKL_Complex16, _np.cdouble
        )
        cls._mkl_sparse_z_mv.restypes = _ctypes.c_int

    @classmethod
    def _set_int_type_misc(cls):
        cls._mkl_sparse_destroy.argtypes = [sparse_matrix_t]
        cls._mkl_sparse_destroy.restypes = _ctypes.c_int

        cls._mkl_sparse_order.argtypes = [sparse_matrix_t]
        cls._mkl_sparse_order.restypes = _ctypes.c_int

    @classmethod
    def _set_int_type_syrk(cls):
        cls._mkl_sparse_syrk.argtypes = [
            _ctypes.c_int,
            sparse_matrix_t,
            _ctypes.POINTER(sparse_matrix_t),
        ]
        cls._mkl_sparse_syrk.restypes = _ctypes.c_int

        cls._mkl_sparse_s_syrkd.argtypes = cls._syrkd_argtypes(
            _ctypes.c_float, _np.float32
        )
        cls._mkl_sparse_s_syrkd.restypes = _ctypes.c_int
        cls._mkl_sparse_d_syrkd.argtypes = cls._syrkd_argtypes(
            _ctypes.c_double, _np.float64
        )
        cls._mkl_sparse_d_syrkd.restypes = _ctypes.c_int
        cls._mkl_sparse_c_syrkd.argtypes = cls._syrkd_argtypes(
            MKL_Complex8, _np.csingle
        )
        cls._mkl_sparse_c_syrkd.restypes = _ctypes.c_int
        cls._mkl_sparse_z_syrkd.argtypes = cls._syrkd_argtypes(
            MKL_Complex16, _np.cdouble
        )
        cls._mkl_sparse_z_syrkd.restypes = _ctypes.c_int

        cls._cblas_ssyrk.argtypes = cls._cblas_syrk_argtypes(
            _ctypes.c_float, _np.float32
        )
        cls._cblas_ssyrk.restypes = None
        cls._cblas_dsyrk.argtypes = cls._cblas_syrk_argtypes(
            _ctypes.c_double, _np.float64
        )
        cls._cblas_dsyrk.restypes = None
        cls._cblas_csyrk.argtypes = cls._cblas_syrk_argtypes(
            MKL_Complex8, _np.csingle, scalar_pointers=True
        )
        cls._cblas_csyrk.restypes = None
        cls._cblas_zsyrk.argtypes = cls._cblas_syrk_argtypes(
            MKL_Complex16, _np.cdouble, scalar_pointers=True
        )
        cls._cblas_zsyrk.restypes = None

    @classmethod
    def _set_int_type_qr_solver(cls):
        """Set the correct argtypes for QR solver functions"""
        cls._mkl_sparse_qr_reorder.argtypes = [sparse_matrix_t, matrix_descr]
        cls._mkl_sparse_qr_reorder.restypes = _ctypes.c_int
        cls._mkl_sparse_d_qr_factorize.argtypes = [
            sparse_matrix_t,
            _ctypes.POINTER(_ctypes.c_double),
        ]
        cls._mkl_sparse_d_qr_factorize.restypes = _ctypes.c_int
        cls._mkl_sparse_s_qr_factorize.argtypes = [
            sparse_matrix_t,
            _ctypes.POINTER(_ctypes.c_float),
        ]
        cls._mkl_sparse_s_qr_factorize.restypes = _ctypes.c_int
        cls._mkl_sparse_d_qr_solve.argtypes = cls._qr_solve(_ctypes.c_double)
        cls._mkl_sparse_d_qr_solve.restypes = _ctypes.c_int
        cls._mkl_sparse_s_qr_solve.argtypes = cls._qr_solve(_ctypes.c_float)
        cls._mkl_sparse_s_qr_solve.restypes = _ctypes.c_int

    def __init__(self):
        raise NotImplementedError("This class is not intended to be instanced")

    """
    The following methods return the argtype lists for each MKL function
    that has s and d variants
    """

    @staticmethod
    def _create_argtypes(prec_type):
        return [
            _ctypes.POINTER(sparse_matrix_t),
            _ctypes.c_int,
            MKL.MKL_INT,
            MKL.MKL_INT,
            ndpointer(dtype=MKL.MKL_INT, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(dtype=MKL.MKL_INT, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(dtype=MKL.MKL_INT, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(dtype=prec_type, ndim=1, flags="C_CONTIGUOUS"),
        ]

    @staticmethod
    def _create_bsr_argtypes(prec_type):
        return [
            _ctypes.POINTER(sparse_matrix_t),
            _ctypes.c_int,
            _ctypes.c_int,
            MKL.MKL_INT,
            MKL.MKL_INT,
            MKL.MKL_INT,
            ndpointer(dtype=MKL.MKL_INT, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(dtype=MKL.MKL_INT, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(dtype=MKL.MKL_INT, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(dtype=prec_type, ndim=3, flags="C_CONTIGUOUS"),
        ]

    @staticmethod
    def _export_argtypes(prec_type):
        return [
            sparse_matrix_t,
            _ctypes.POINTER(_ctypes.c_int),
            _ctypes.POINTER(MKL.MKL_INT),
            _ctypes.POINTER(MKL.MKL_INT),
            _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
            _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
            _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
            _ctypes.POINTER(_ctypes.POINTER(prec_type)),
        ]

    @staticmethod
    def _export_bsr_argtypes(prec_type):
        return [
            sparse_matrix_t,
            _ctypes.POINTER(_ctypes.c_int),
            _ctypes.POINTER(_ctypes.c_int),
            _ctypes.POINTER(MKL.MKL_INT),
            _ctypes.POINTER(MKL.MKL_INT),
            _ctypes.POINTER(MKL.MKL_INT),
            _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
            _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
            _ctypes.POINTER(_ctypes.POINTER(MKL.MKL_INT)),
            _ctypes.POINTER(_ctypes.POINTER(prec_type)),
        ]

    @staticmethod
    def _cblas_gemm_argtypes(prec_type, scalar_pointers=False):
        return [
            _ctypes.c_int,
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
            MKL.MKL_INT,
        ]

    @staticmethod
    def _spmmd_argtypes(prec_type):
        return [
            _ctypes.c_int,
            sparse_matrix_t,
            sparse_matrix_t,
            _ctypes.c_int,
            _ctypes.POINTER(prec_type),
            MKL.MKL_INT,
        ]

    @staticmethod
    def _mm_argtypes(scalar_type, prec_type, _np_prec_type):
        return [
            _ctypes.c_int,
            scalar_type,
            sparse_matrix_t,
            matrix_descr,
            _ctypes.c_int,
            ndpointer(dtype=_np_prec_type, ndim=2),
            MKL.MKL_INT,
            MKL.MKL_INT,
            scalar_type,
            _ctypes.POINTER(prec_type),
            MKL.MKL_INT,
        ]

    @staticmethod
    def _mv_argtypes(prec_type, _np_type):
        return [
            _ctypes.c_int,
            prec_type,
            sparse_matrix_t,
            matrix_descr,
            ndpointer(dtype=_np_type, ndim=1),
            prec_type,
            ndpointer(dtype=_np_type),
        ]

    @staticmethod
    def _syrkd_argtypes(prec_type, _np_type):
        return [
            _ctypes.c_int,
            sparse_matrix_t,
            prec_type,
            prec_type,
            ndpointer(dtype=_np_type),
            _ctypes.c_int,
            MKL.MKL_INT,
        ]

    @staticmethod
    def _cblas_syrk_argtypes(prec_type, _np_type, scalar_pointers=False):
        return [
            _ctypes.c_int,
            _ctypes.c_int,
            _ctypes.c_int,
            MKL.MKL_INT,
            MKL.MKL_INT,
            _ctypes.POINTER(prec_type) if scalar_pointers else prec_type,
            ndpointer(dtype=_np_type, ndim=2),
            MKL.MKL_INT,
            _ctypes.POINTER(prec_type) if scalar_pointers else prec_type,
            ndpointer(dtype=_np_type, ndim=2),
            MKL.MKL_INT,
        ]

    @staticmethod
    def _qr_solve(prec_type):
        return [
            _ctypes.c_int,
            sparse_matrix_t,
            _ctypes.POINTER(prec_type),
            _ctypes.c_int,
            MKL.MKL_INT,
            _ctypes.POINTER(prec_type),
            MKL.MKL_INT,
            ndpointer(dtype=prec_type, ndim=2),
            MKL.MKL_INT,
        ]

    @staticmethod
    def _create_iss_argtypes():
        return [
            _ctypes.POINTER(MKL.MKL_INT),
            ndpointer(dtype=_ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
            ndpointer(dtype=_ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
            _ctypes.POINTER(MKL.MKL_INT),
            ndpointer(dtype=MKL.MKL_INT, shape=(128,), flags='C_CONTIGUOUS'),
            ndpointer(dtype=_ctypes.c_double, shape=(128,), flags='C_CONTIGUOUS'),
            ndpointer(dtype=_ctypes.c_double, flags='C_CONTIGUOUS')
        ]

    @staticmethod
    def _create_iss_mrhs_argtypes(add_method=False):
        _arg = MKL._create_iss_argtypes()[0:2] + [
            _ctypes.POINTER(MKL.MKL_INT)
        ]
        if add_method:
            _arg = _arg + MKL._create_iss_argtypes()[2:3] + [
                _ctypes.POINTER(MKL.MKL_INT)
            ] + MKL._create_iss_argtypes()[3:]
        else:
            _arg = _arg + MKL._create_iss_argtypes()[2:]
        
        return _arg


# Set argtypes and return types for service functions
# not interface dependent
MKL._mkl_set_interface_layer.argtypes = [_ctypes.c_int]
MKL._mkl_set_interface_layer.restypes = [_ctypes.c_int]
MKL._mkl_get_max_threads.argtypes = None
MKL._mkl_get_max_threads.restypes = [_ctypes.c_int]
MKL._mkl_set_num_threads.argtypes = [_ctypes.c_int]
MKL._mkl_set_num_threads.restypes = None
MKL._mkl_set_num_threads_local.argtypes = [_ctypes.c_int]
MKL._mkl_set_num_threads_local.restypes = [_ctypes.c_int]
MKL._mkl_get_version.argtypes = [_ctypes.POINTER(MKLVersion)]
MKL._mkl_get_version.restypes = None
MKL._mkl_get_version_string.argtypes = [
    _ctypes.POINTER(_ctypes.c_char * 256),
    _ctypes.c_int
]
MKL._mkl_get_version_string.restypes = None
MKL._mkl_free_buffers.argtypes = None
MKL._mkl_free_buffers.restype = None


def mkl_set_interface_layer(layer_code):
    if layer_code not in [0, 1, 2]:
        raise ValueError(
            f'{layer_code} invalid argument to mkl_set_interface_layer'
        )

    return MKL._mkl_set_interface_layer(layer_code)


def mkl_get_max_threads():
    return MKL._mkl_get_max_threads()


def mkl_set_num_threads(n_threads):
    MKL._mkl_set_num_threads(n_threads)


def mkl_set_num_threads_local(n_threads):
    return MKL._mkl_set_num_threads_local(n_threads)


def mkl_get_version():
    mkl_version = MKLVersion()
    MKL._mkl_get_version(mkl_version)
    return (
        mkl_version.MajorVersion,
        mkl_version.MinorVersion,
        mkl_version.UpdateVersion,
        mkl_version.ProductStatus.contents.value.decode(),
        mkl_version.Build.contents.value.decode(),
        mkl_version.Processor.contents.value.decode(),
        mkl_version.Platform.contents.value.decode()
    )


def mkl_get_version_string():
    c_str = (_ctypes.c_char * 256)()
    MKL._mkl_get_version_string(_ctypes.byref(c_str), 256)
    return c_str.value.decode()


def mkl_free_buffers():
    MKL._mkl_free_buffers()


_mkl_interface_env = os.getenv('MKL_INTERFACE_LAYER')


if MKL.MKL_INT is None and _mkl_interface_env == 'ILP64':
    mkl_set_interface_layer(MKL_INTERFACE_ILP64)
    MKL._set_int_type(_ctypes.c_longlong, _np.int64)
elif MKL.MKL_INT is None and _mkl_interface_env == 'LP64':
    mkl_set_interface_layer(MKL_INTERFACE_LP64)
    MKL._set_int_type(_ctypes.c_int, _np.int32)
