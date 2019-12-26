import ctypes as _ctypes
import numpy as np
import scipy.sparse as _spsparse
from numpy.ctypeslib import ndpointer, as_array
from numpy.testing import assert_array_almost_equal

# Load mkl_spblas.so through the common interface
_libmkl = _ctypes.cdll.LoadLibrary("libmkl_rt.so")
NUMPY_FLOAT_DTYPES = [np.float32, np.float64]


class MKL:
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

    @classmethod
    def _set_int_type(cls, c_type, np_type):
        cls.MKL_INT = c_type
        cls.MKL_INT_NUMPY = np_type

        create_argtypes = [_ctypes.POINTER(sparse_matrix_t),
                           _ctypes.c_int,
                           cls.MKL_INT,
                           cls.MKL_INT,
                           ndpointer(dtype=cls.MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                           ndpointer(dtype=cls.MKL_INT, ndim=1, flags='C_CONTIGUOUS'),
                           ndpointer(dtype=cls.MKL_INT, ndim=1, flags='C_CONTIGUOUS')]

        export_argtypes = [sparse_matrix_t,
                           _ctypes.POINTER(_ctypes.c_int),
                           _ctypes.POINTER(cls.MKL_INT),
                           _ctypes.POINTER(cls.MKL_INT),
                           _ctypes.POINTER(_ctypes.POINTER(cls.MKL_INT)),
                           _ctypes.POINTER(_ctypes.POINTER(cls.MKL_INT)),
                           _ctypes.POINTER(_ctypes.POINTER(cls.MKL_INT))]

        ndpt_double = ndpointer(dtype=_ctypes.c_double, ndim=1, flags='C_CONTIGUOUS')
        ndpt_single = ndpointer(dtype=_ctypes.c_float, ndim=1, flags='C_CONTIGUOUS')

        cls._mkl_sparse_d_create_csr.argtypes = create_argtypes + [ndpt_double]
        cls._mkl_sparse_d_create_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_s_create_csr.argtypes = create_argtypes + [ndpt_single]
        cls._mkl_sparse_s_create_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_d_create_csc.argtypes = create_argtypes + [ndpt_double]
        cls._mkl_sparse_d_create_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_s_create_csc.argtypes = create_argtypes + [ndpt_single]
        cls._mkl_sparse_s_create_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_d_export_csr.argtypes = export_argtypes + [_ctypes.POINTER(_ctypes.POINTER(_ctypes.c_double))]
        cls._mkl_sparse_d_export_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_csr.argtypes = export_argtypes + [_ctypes.POINTER(_ctypes.POINTER(_ctypes.c_float))]
        cls._mkl_sparse_s_export_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_d_export_csc.argtypes = export_argtypes + [_ctypes.POINTER(_ctypes.POINTER(_ctypes.c_double))]
        cls._mkl_sparse_d_export_csc.restypes = _ctypes.c_int

        cls._mkl_sparse_s_export_csr.argtypes = export_argtypes + [_ctypes.POINTER(_ctypes.POINTER(_ctypes.c_float))]
        cls._mkl_sparse_s_export_csr.restypes = _ctypes.c_int

        cls._mkl_sparse_spmm.argtypes = [_ctypes.c_int,
                                         sparse_matrix_t,
                                         sparse_matrix_t,
                                         _ctypes.POINTER(sparse_matrix_t)]
        cls._mkl_sparse_spmm.restypes = _ctypes.c_int

        cls._mkl_sparse_destroy.argtypes = [sparse_matrix_t]
        cls._mkl_sparse_destroy.restypes = _ctypes.c_int

        cls._mkl_sparse_order.argtypes = [sparse_matrix_t]
        cls._mkl_sparse_order.restypes = _ctypes.c_int


# Construct opaque struct & type
class _sparse_matrix(_ctypes.Structure):
    pass


sparse_matrix_t = _ctypes.POINTER(_sparse_matrix)

# Define standard return codes
RETURN_CODES = {0: "SPARSE_STATUS_SUCCESS",
                1: "SPARSE_STATUS_NOT_INITIALIZED",
                2: "SPARSE_STATUS_ALLOC_FAILED",
                3: "SPARSE_STATUS_INVALID_VALUE",
                4: "SPARSE_STATUS_EXECUTION_FAILED",
                5: "SPARSE_STATUS_INTERNAL_ERROR",
                6: "SPARSE_STATUS_NOT_SUPPORTED"}


def _create_mkl_csr(csr_data, double_precision=True, copy=False):
    """
    Create MKL internal representation in CSR format
    https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr

    :param csr_data: Sparse data in CSR format
    :type csr_data: scipy.sparse.csr_matrix
    :param double_precision: Use float64 if True, float32 if False
    :type double_precision: bool
    :param copy: If the dtype must be cast to a different format, make a copy if True or change the underlying data if
        False.
    :type copy: bool

    :return ref: Handle for the MKL internal representation
        Also returns a tuple of references to the data structures which MKL is using.
        This prevents the python garbage collector from deallocating them
    :rtype: sparse_matrix_t
    """

    ref = sparse_matrix_t()

    # Cast indexes to MKL_INT type
    if csr_data.indptr.dtype != MKL.MKL_INT_NUMPY:
        csr_data.indptr = csr_data.indptr.astype(MKL.MKL_INT_NUMPY)
    if csr_data.indices.dtype != MKL.MKL_INT_NUMPY:
        csr_data.indices = csr_data.indices.astype(MKL.MKL_INT_NUMPY)

    # Figure out which dtype for data
    if double_precision:
        csr_func = MKL._mkl_sparse_d_create_csr
        csr_dtype = np.float64
    else:
        csr_func = MKL._mkl_sparse_s_create_csr
        csr_dtype = np.float32

    # Cast data if it has to be cast
    if csr_data.data.dtype != csr_dtype and copy:
        data = csr_data.data.astype(csr_dtype)
    elif csr_data.dtype != csr_dtype:
        csr_data.data = csr_data.data.astype(csr_dtype)
        data = csr_data.data
    else:
        data = csr_data.data

    # Load into a MKL data structure and check return
    ret_val = csr_func(_ctypes.byref(ref),
                       _ctypes.c_int(0),
                       MKL.MKL_INT(csr_data.shape[0]),
                       MKL.MKL_INT(csr_data.shape[1]),
                       csr_data.indptr[0:-1],
                       csr_data.indptr[1:],
                       csr_data.indices,
                       data)

    # Check return
    if ret_val != 0:
        fname = "mkl_sparse_d_create" if double_precision else "mkl_sparse_s_create"
        err = "{fn} returned {v} ({e})".format(fn=fname, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err)

    return ref


def _create_mkl_csc(csc_data, double_precision=True, copy=False):
    """
    Create MKL internal representation in CSC format
    https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr

    :param csc_data: Sparse data in CSC format
    :type csc_data: scipy.sparse.csr_matrix
    :param double_precision: Use float64 if True, float32 if False
    :type double_precision: bool
    :param copy: If the dtype must be cast to a different format, make a copy if True or change the underlying data if
        False.
    :type copy: bool

    :return ref: Handle for the MKL internal representation
        Also returns a tuple of references to the data structures which MKL is using.
        This prevents the python garbage collector from deallocating them
    :rtype: sparse_matrix_t
    """

    ref = sparse_matrix_t()

    # Cast indexes to MKL_INT type
    if csc_data.indptr.dtype != MKL.MKL_INT_NUMPY:
        csc_data.indptr = csc_data.indptr.astype(MKL.MKL_INT_NUMPY)
    if csc_data.indices.dtype != MKL.MKL_INT_NUMPY:
        csc_data.indices = csc_data.indices.astype(MKL.MKL_INT_NUMPY)

    # Figure out which dtype for data
    if double_precision:
        csc_func = MKL._mkl_sparse_d_create_csc
        csc_dtype = np.float64
    else:
        csc_func = MKL._mkl_sparse_s_create_csc
        csc_dtype = np.float32

    # Cast data if it has to be cast
    if csc_data.data.dtype != csc_dtype and copy:
        data = csc_data.data.astype(csc_dtype)
    elif csc_data.dtype != csc_dtype:
        csc_data.data = csc_data.data.astype(csc_dtype)
        data = csc_data.data
    else:
        data = csc_data.data

    # Load into a MKL data structure and check return
    ret_val = csc_func(_ctypes.byref(ref),
                       _ctypes.c_int(0),
                       MKL.MKL_INT(csc_data.shape[0]),
                       MKL.MKL_INT(csc_data.shape[1]),
                       csc_data.indptr[0:-1],
                       csc_data.indptr[1:],
                       csc_data.indices,
                       data)

    # Check return
    if ret_val != 0:
        fname = "mkl_sparse_d_create" if double_precision else "mkl_sparse_s_create"
        err = "{fn} returned {v} ({e})".format(fn=fname, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err)

    return ref


def _export_csr_mkl(csr_mkl_handle, double_precision=True, copy=False):
    """
    Export a MKL sparse handle in CSR format
    https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-export-csr

    :param csr_mkl_handle: Handle for the MKL internal representation
    :type csr_mkl_handle: sparse_matrix_t
    :param double_precision: Use float64 if True, float32 if False. This MUST match the underlying float type - this
        defines a memory view, it does not cast.
    :type double_precision: bool

    :return:
    """

    indptrb = _ctypes.POINTER(MKL.MKL_INT)()
    indptren = _ctypes.POINTER(MKL.MKL_INT)()
    indices = _ctypes.POINTER(MKL.MKL_INT)()

    ordering = _ctypes.c_int()
    nrows = MKL.MKL_INT()
    ncols = MKL.MKL_INT()

    if double_precision:
        data = _ctypes.POINTER(_ctypes.c_double)()
        csr_func = MKL._mkl_sparse_d_export_csr
        final_dtype = np.float64
    else:
        data = _ctypes.POINTER(_ctypes.c_float)()
        csr_func = MKL._mkl_sparse_s_export_csr
        final_dtype = np.float32

    ret_val = csr_func(csr_mkl_handle,
                       _ctypes.byref(ordering),
                       _ctypes.byref(nrows),
                       _ctypes.byref(ncols),
                       _ctypes.byref(indptrb),
                       _ctypes.byref(indptren),
                       _ctypes.byref(indices),
                       _ctypes.byref(data))

    # Check return
    if ret_val != 0:
        fname = "mkl_sparse_d_export" if double_precision else "mkl_sparse_s_export"
        err = "{fn} returned {v} ({e})".format(fn=fname, v=ret_val, e=RETURN_CODES[ret_val])
        raise ValueError(err)

    # Check ordering
    if ordering.value != 0:
        raise ValueError("1-indexing (F-style) is not supported")

    # Get matrix dims
    ncols = ncols.value
    nrows = nrows.value

    # If any axis is 0 return an empty matrix
    if nrows == 0 or ncols == 0:
        return _spsparse.csr_matrix((nrows, ncols), dtype=final_dtype)

    # Construct a numpy array from row end index pointer
    # Add 0 to first position for scipy.sparse's 3-array indexing
    indptren = as_array(indptren, shape=(nrows,))
    indptren = np.insert(indptren, 0, 0)
    nnz = indptren[-1]

    # If there are no non-zeros, return an empty matrix
    # If the number of non-zeros is insane, raise a ValueError
    if nnz == 0:
        return _spsparse.csr_matrix((nrows, ncols), dtype=final_dtype)
    elif nnz < 0 or nnz > ncols * nrows:
        raise ValueError("Matrix ({m} x {n}) is attempting to index {z} elements".format(m=nrows, n=ncols, z=nnz))

    # Construct numpy arrays from data pointer and from column indicies pointer
    data = np.array(as_array(data, shape=(nnz,)), copy=copy)
    indices = np.array(as_array(indices, shape=(nnz,)), copy=copy)

    # Pack and return the CSR matrix

    return _spsparse.csr_matrix((data, indices, indptren), shape=(nrows, ncols))


def _validate_dtype():
    test_array = _spsparse.random(50, 50, density=0.5, format="csc", dtype=np.float32, random_state=50)
    test_comparison = test_array.A

    csc_ref = _create_mkl_csc(test_array, double_precision=False)
    csr_ref = sparse_matrix_t()

    if MKL._mkl_sparse_convert_csr(csc_ref, _ctypes.c_int(10), _ctypes.byref(csr_ref)) != 0:
        raise ValueError("CSC to CSR Conversion failed")

    final_array = _export_csr_mkl(csr_ref, double_precision=False)
    try:
        assert_array_almost_equal(test_comparison, final_array.A)
    except AssertionError:
        raise
    finally:
        MKL._mkl_sparse_destroy(csr_ref)

    return True


def _matmul_mkl(sp_ref_a, sp_ref_b):
    """
    Dot product two MKL objects together and return a handle to the result

    :param sp_ref_a: Sparse matrix A handle
    :type sp_ref_a: sparse_matrix_t
    :param sp_ref_b: Sparse matrix B handle
    :param sp_ref_b: sparse_matrix_t
    :return: Sparse matrix handle that is the dot product A * B
    :rtype: sparse_matrix_t
    """

    ref_handle = sparse_matrix_t()
    ret_val = MKL._mkl_sparse_spmm(_ctypes.c_int(10),
                                   sp_ref_a,
                                   sp_ref_b,
                                   _ctypes.byref(ref_handle))

    # Check return
    if ret_val != 0:
        raise ValueError("mkl_sparse_spmm returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))

    return ref_handle


def _destroy_mkl_handle(ref_handle):
    """
    Deallocate a MKL sparse handle

    :param ref_handle:
    :type ref_handle: sparse_matrix_t
    :return:
    """

    ret_val = MKL._mkl_sparse_destroy(ref_handle)

    if ret_val != 0:
        raise ValueError("mkl_sparse_destroy returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))


def _order_mkl_handle(ref_handle):
    """
    Reorder indexes in a MKL sparse handle

    :param ref_handle:
    :type ref_handle: sparse_matrix_t
    :return:
    """

    ret_val = MKL._mkl_sparse_order(ref_handle)

    if ret_val != 0:
        raise ValueError("mkl_sparse_order returned {v} ({e})".format(v=ret_val, e=RETURN_CODES[ret_val]))


# Define dtypes empirically
# Basically just try with int64s and if that doesn't work try with int32s
if MKL.MKL_INT is None:

    MKL._set_int_type(_ctypes.c_longlong, np.int64)

    try:
        _validate_dtype()
    except (AssertionError, ValueError) as err:

        MKL._set_int_type(_ctypes.c_int, np.int32)

        try:
            _validate_dtype()
        except (AssertionError, ValueError):
            raise ImportError("Unable to set MKL numeric types")
