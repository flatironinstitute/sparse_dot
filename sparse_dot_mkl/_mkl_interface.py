import warnings
import ctypes as _ctypes
import ctypes.util as _ctypes_util

# Load mkl_spblas through the libmkl_rt common interface
_libmkl, _libmkl_loading_errors = None, []
try:
    so_file = _ctypes_util.find_library('mkl_rt')
    _libmkl = _ctypes.cdll.LoadLibrary(so_file)    
except (OSError, ImportError) as err:
    _libmkl_loading_errors.append(err)

if _libmkl._name is None:
    ierr_msg = "Unable to load the MKL libraries through libmkl_rt. Try setting $LD_LIBRARY_PATH."
    ierr_msg += "\n\t" + "\n\t".join(map(lambda x: str(x), _libmkl_loading_errors))
    raise ImportError(ierr_msg)

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
    msg = "Loaded version of MKL is out of date: {v}".format(v=get_version_string())
    warnings.warn(msg)

import numpy as np
import scipy.sparse as _spsparse
from numpy.ctypeslib import ndpointer, as_array

NUMPY_FLOAT_DTYPES = [np.float32, np.float64]


class MKL:
    """ This class holds shared object references to C functions with arg and returntypes that can be adjusted"""

    MKL_INT = None
    MKL_INT_NUMPY = None

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

        cls._mkl_sparse_d_export_csr.argtypes = cls._mkl_export_create_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_export_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_csr.argtypes = cls._mkl_export_create_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_export_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_d_export_csc.argtypes = cls._mkl_export_create_argtypes(_ctypes.c_double)
        cls._mkl_sparse_d_export_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_csr.argtypes = cls._mkl_export_create_argtypes(_ctypes.c_float)
        cls._mkl_sparse_s_export_csr.restypes = _ctypes.c_int

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
    def _mkl_export_create_argtypes(prec_type):
        return [sparse_matrix_t,
                _ctypes.POINTER(_ctypes.c_int),
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


def _check_scipy_index_typing(sparse_matrix):
    """
    Ensure that the sparse matrix indicies are in the correct integer type

    :param sparse_matrix: Scipy matrix in CSC or CSR format
    :type sparse_matrix: scipy.sparse.spmatrix
    """

    int_max = np.iinfo(MKL.MKL_INT_NUMPY).max
    if (sparse_matrix.nnz > int_max) or (max(sparse_matrix.shape) > int_max):
        msg = "MKL interface is {t} and cannot hold matrix {m}".format(m=repr(sparse_matrix), t=MKL.MKL_INT_NUMPY)
        raise ValueError(msg)

    # Cast indexes to MKL_INT type
    if sparse_matrix.indptr.dtype != MKL.MKL_INT_NUMPY:
        sparse_matrix.indptr = sparse_matrix.indptr.astype(MKL.MKL_INT_NUMPY)
    if sparse_matrix.indices.dtype != MKL.MKL_INT_NUMPY:
        sparse_matrix.indices = sparse_matrix.indices.astype(MKL.MKL_INT_NUMPY)


def _get_numpy_layout(numpy_arr):
    """
    Get the array layout code for a dense array in C or F order.
    Raises a ValueError if the array is not contiguous.

    :param numpy_arr: Numpy dense array
    :type numpy_arr: np.ndarray
    :return: The layout code for MKL and the leading dimension
    :rtype: int, int
    """

    if numpy_arr.flags.c_contiguous:
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

    # Figure out which dtype for data
    if matrix.dtype == np.float32:
        double_precision = False
    elif matrix.dtype == np.float64:
        double_precision = True
    else:
        raise ValueError("Only float32 or float64 dtypes are supported")

    # Figure out which matrix creation function to use
    if _spsparse.isspmatrix_csr(matrix):
        assert matrix.indptr.shape[0] == matrix.shape[0] + 1
        handle_func = MKL._mkl_sparse_d_create_csr if double_precision else MKL._mkl_sparse_s_create_csr
    elif _spsparse.isspmatrix_csc(matrix):
        assert matrix.indptr.shape[0] == matrix.shape[1] + 1
        handle_func = MKL._mkl_sparse_d_create_csc if double_precision else MKL._mkl_sparse_s_create_csc
    else:
        raise ValueError("Matrix is not CSC or CSR")

    # Make sure indices are of the correct integer type
    _check_scipy_index_typing(matrix)
    assert matrix.data.shape[0] == matrix.indices.shape[0]

    return _pass_mkl_handle(matrix, handle_func), double_precision


def _pass_mkl_handle(data, handle_func):
    """
    Create MKL internal representation

    :param data: Sparse data
    :type data: scipy.sparse.spmatrix
    :return ref: Handle for the MKL internal representation
    :rtype: sparse_matrix_t
    """

    # Create a pointer for the output matrix
    ref = sparse_matrix_t()

    # Load into a MKL data structure and check return
    ret_val = handle_func(_ctypes.byref(ref),
                          _ctypes.c_int(0),
                          MKL.MKL_INT(data.shape[0]),
                          MKL.MKL_INT(data.shape[1]),
                          data.indptr[0:-1],
                          data.indptr[1:],
                          data.indices,
                          data.data)

    # Check return
    if ret_val != 0:
        err_msg = "{fn} returned {v} ({e})".format(fn=handle_func.__name__, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err_msg)

    return ref


def _export_mkl(csr_mkl_handle, double_precision, output_type="csr"):
    """
    Export a MKL sparse handle

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

    # Create the pointers for the output data
    indptrb = _ctypes.POINTER(MKL.MKL_INT)()
    indptren = _ctypes.POINTER(MKL.MKL_INT)()
    indices = _ctypes.POINTER(MKL.MKL_INT)()

    ordering = _ctypes.c_int()
    nrows = MKL.MKL_INT()
    ncols = MKL.MKL_INT()

    output_type = output_type.lower()

    if output_type == "csr":
        out_func = MKL._mkl_sparse_d_export_csr if double_precision else MKL._mkl_sparse_s_export_csr
        sp_matrix_constructor = _spsparse.csr_matrix
    elif output_type == "csc":
        out_func = MKL._mkl_sparse_d_export_csc if double_precision else MKL._mkl_sparse_s_export_csc
        sp_matrix_constructor = _spsparse.csc_matrix
    else:
        raise ValueError("Only CSR and CSC output types are supported")

    if double_precision:
        data = _ctypes.POINTER(_ctypes.c_double)()
        final_dtype = np.float64
    else:
        data = _ctypes.POINTER(_ctypes.c_float)()
        final_dtype = np.float32

    ret_val = out_func(csr_mkl_handle,
                       _ctypes.byref(ordering),
                       _ctypes.byref(nrows),
                       _ctypes.byref(ncols),
                       _ctypes.byref(indptrb),
                       _ctypes.byref(indptren),
                       _ctypes.byref(indices),
                       _ctypes.byref(data))

    # Check return
    if ret_val != 0:
        err_msg = "{fn} returned {v} ({e})".format(fn=out_func.__name__, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err_msg)

    # Check ordering
    if ordering.value != 0:
        raise ValueError("1-indexing (F-style) is not supported")

    # Get matrix dims
    ncols = ncols.value
    nrows = nrows.value

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


def _destroy_mkl_handle(ref_handle):
    """
    Deallocate a MKL sparse handle

    :param ref_handle:
    :type ref_handle: sparse_matrix_t
    """

    ret_val = MKL._mkl_sparse_destroy(ref_handle)

    if ret_val != 0:
        raise ValueError("mkl_sparse_destroy returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))


def _order_mkl_handle(ref_handle):
    """
    Reorder indexes in a MKL sparse handle

    :param ref_handle:
    :type ref_handle: sparse_matrix_t
    """

    ret_val = MKL._mkl_sparse_order(ref_handle)

    if ret_val != 0:
        raise ValueError("mkl_sparse_order returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))


def _convert_to_csr(ref_handle, destroy_original=False):
    """
    Convert a MKL sparse handle to CSR format

    :param ref_handle:
    :type ref_handle: sparse_matrix_t
    :return:
    """

    csr_ref = sparse_matrix_t()
    ret_val = MKL._mkl_sparse_convert_csr(ref_handle, _ctypes.c_int(10), _ctypes.byref(csr_ref))

    if ret_val != 0:
        try:
            _destroy_mkl_handle(csr_ref)
        except ValueError:
            pass

        raise ValueError("mkl_sparse_convert_csr returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))

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

    invalid_ndims = not (a_2d or a_vec) or not (b_2d, b_vec)
    invalid_align = (matrix_a.shape[1] if not matrix_a.ndim == 1 else matrix_a.shape[0]) != matrix_b.shape[0]

    # Check to make sure that this multiplication can work
    if invalid_align or invalid_ndims:
        err_msg = "Matrix alignment error: {m1} * {m2} is not valid".format(m1=matrix_a.shape, m2=matrix_b.shape)
        raise ValueError(err_msg)


def _cast_to_float64(matrix):
    """ Make a copy of the array as double precision floats or return the reference if it already is"""
    return matrix.astype(np.float64) if matrix.dtype != np.float64 else matrix


def _type_check(matrix_a, matrix_b=None, cast=False, dprint=print):
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
        dprint("Recasting matrix data types {a} and {b} to np.float64".format(a=matrix_a.dtype,
                                                                              b=matrix_b.dtype))
        return _cast_to_float64(matrix_a), _cast_to_float64(matrix_b)

    elif matrix_a.dtype != np.float64 or matrix_b.dtype != np.float64:
        err_msg = "Matrix data types must be in concordance; {a} and {b} provided".format(a=matrix_a.dtype,
                                                                                          b=matrix_b.dtype)
        raise ValueError(err_msg)


def _is_dense_vector(m_or_v):
    return not _spsparse.issparse(m_or_v) and ((m_or_v.ndim == 1) or ((m_or_v.ndim == 2) and min(m_or_v.shape) == 1))


def _empty_output_check(matrix_a, matrix_b):
    """Check for trivial cases where an empty array should be produced"""

    # One dimension is zero
    if min([*matrix_a.shape, *matrix_b.shape]) == 0:
        return True

    # The sparse array is empty
    elif _spsparse.issparse(matrix_a) and min(matrix_a.data.shape[0], matrix_a.indices.shape[0]) == 0:
        return True
    elif _spsparse.issparse(matrix_b) and min(matrix_b.data.shape[0], matrix_b.indices.shape[0]) == 0:
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


# Define dtypes empirically
# Basically just try with int64s and if that doesn't work try with int32s
# There's a way to do this with intel's mkl helper package but I don't want to add the dependency
if MKL.MKL_INT is None:

    MKL._set_int_type(_ctypes.c_longlong, np.int64)

    try:
        _validate_dtype()
    except ValueError as err:

        MKL._set_int_type(_ctypes.c_int, np.int32)

        try:
            _validate_dtype()
        except ValueError:
            raise ImportError("Unable to set MKL numeric type")
