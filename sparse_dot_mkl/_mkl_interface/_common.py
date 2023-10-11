from ._cfunctions import MKL, mkl_library_name
from ._constants import (
    LAYOUT_CODE_C,
    LAYOUT_CODE_F,
    SPARSE_INDEX_BASE_ZERO,
    RETURN_CODES,
    ILP64_MSG
)
from ._structs import (
    sparse_matrix_t,
    MKL_Complex8,
    MKL_Complex16
)

import numpy as _np
import ctypes as _ctypes
from scipy import sparse as _spsparse
import warnings
import time

from numpy.ctypeslib import as_array

# Use mkl-service to check version if it's installed
# Since it's not on PyPi I don't want to make this an actual package dependency
# So without it just create mock functions and don't do version checking
try:
    from mkl import get_version, get_version_string, get_max_threads
except ImportError:

    def get_version():
        return None

    def get_version_string():
        return None

    def get_max_threads():
        return None


if get_version() is not None and get_version()["MajorVersion"] < 2020:
    warnings.warn(
        f"Loaded version of MKL is out of date: {get_version_string()}"
    )


# Dict keyed by ('sparse_type_str', 'double_precision_bool', 'complex_bool')
_create_functions = {
    ("csr", False, False): MKL._mkl_sparse_s_create_csr,
    ("csr", True, False): MKL._mkl_sparse_d_create_csr,
    ("csr", False, True): MKL._mkl_sparse_c_create_csr,
    ("csr", True, True): MKL._mkl_sparse_z_create_csr,
    ("csc", False, False): MKL._mkl_sparse_s_create_csc,
    ("csc", True, False): MKL._mkl_sparse_d_create_csc,
    ("csc", False, True): MKL._mkl_sparse_c_create_csc,
    ("csc", True, True): MKL._mkl_sparse_z_create_csc,
    ("bsr", False, False): MKL._mkl_sparse_s_create_bsr,
    ("bsr", True, False): MKL._mkl_sparse_d_create_bsr,
    ("bsr", False, True): MKL._mkl_sparse_c_create_bsr,
    ("bsr", True, True): MKL._mkl_sparse_z_create_bsr,
}

# Dict keyed by ('sparse_type_str', 'double_precision_bool', 'complex_bool')
_export_functions = {
    ("csr", False, False): MKL._mkl_sparse_s_export_csr,
    ("csr", True, False): MKL._mkl_sparse_d_export_csr,
    ("csr", False, True): MKL._mkl_sparse_c_export_csr,
    ("csr", True, True): MKL._mkl_sparse_z_export_csr,
    ("csc", False, False): MKL._mkl_sparse_s_export_csc,
    ("csc", True, False): MKL._mkl_sparse_d_export_csc,
    ("csc", False, True): MKL._mkl_sparse_c_export_csc,
    ("csc", True, True): MKL._mkl_sparse_z_export_csc,
    ("bsr", False, False): MKL._mkl_sparse_s_export_bsr,
    ("bsr", True, False): MKL._mkl_sparse_d_export_bsr,
    ("bsr", False, True): MKL._mkl_sparse_c_export_bsr,
    ("bsr", True, True): MKL._mkl_sparse_z_export_bsr,
}

# Dict keyed by ('double_precision_bool', 'complex_bool')
_output_dtypes = {
    (False, False): _np.float32,
    (True, False): _np.float64,
    (False, True): _np.csingle,
    (True, True): _np.cdouble,
}

NUMPY_FLOAT_DTYPES = [_np.float32, _np.float64]
NUMPY_COMPLEX_DTYPES = [_np.csingle, _np.cdouble]


# Initial support for sparse_array structures in scipy
try:
    _scipy_objs = {
        'csr': (_spsparse.csr_matrix, _spsparse.csr_array),
        'csc': (_spsparse.csc_matrix, _spsparse.csc_array),
        'bsr': (_spsparse.bsr_matrix, _spsparse.bsr_array),
        'csr_matrix': _spsparse.csr_matrix,
        'csr_array': _spsparse.csr_array,
        'csc_matrix': _spsparse.csc_matrix,
        'csc_array': _spsparse.csc_array,
        'bsr_matrix': _spsparse.bsr_matrix,
        'bsr_array': _spsparse.bsr_array
    }

except AttributeError:
    _scipy_objs = {
        'csr': (_spsparse.csr_matrix, ),
        'csc': (_spsparse.csc_matrix, ),
        'bsr': (_spsparse.bsr_matrix, ),
        'csr_matrix': _spsparse.csr_matrix,
        'csc_matrix': _spsparse.csc_matrix,
        'bsr_matrix': _spsparse.bsr_matrix
    }


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
        print(f"MKL Number of Threads: {get_max_threads()}")

    print(f"MKL linked: {mkl_library_name()}")
    print(f"MKL interface {MKL.MKL_INT_NUMPY} | {MKL.MKL_INT}")
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

    int_max = _np.iinfo(MKL.MKL_INT_NUMPY).max
    if (sparse_matrix.nnz > int_max) or (max(sparse_matrix.shape) > int_max):
        raise ValueError(
            f"MKL interface is {MKL.MKL_INT_NUMPY} and cannot hold matrix "
            f"{repr(sparse_matrix)}; Try changing MKL to int64 with the "
            "environment variable MKL_INTERFACE_LAYER=ILP64"
        )

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
    :type numpy_arr: _np.ndarray
    :param second_arr: Numpy dense array; if numpy_arr is 1d
        (and therefore both C and F order), use this order
    :type second_arr: _np.ndarray, None
    :return: The layout code for MKL and the leading dimension
    :rtype: int, int
    """

    _is_c = numpy_arr.flags.c_contiguous
    _is_f = numpy_arr.flags.f_contiguous

    # Return the second array order if the first is ambiguous
    if _is_c and _is_f and second_arr is not None:
        if second_arr.flags.c_contiguous:
            return LAYOUT_CODE_C, numpy_arr.shape[1]
        elif second_arr.flags.f_contiguous:
            return LAYOUT_CODE_F, numpy_arr.shape[0]

    # Return the first order array otherwise
    elif _is_c:
        return LAYOUT_CODE_C, numpy_arr.shape[1]
    elif _is_f:
        return LAYOUT_CODE_F, numpy_arr.shape[0]
    elif not numpy_arr.flags.contiguous:
        raise ValueError("Array is not contiguous")
    else:
        raise ValueError("Array layout check has failed")


def is_csr(x):
    return isinstance(x, _scipy_objs['csr'])


def is_csc(x):
    return isinstance(x, _scipy_objs['csc'])


def is_bsr(x):
    return isinstance(x, _scipy_objs['bsr'])


def sparse_output_type(x):

    for output_type, constructor in _scipy_objs.items():

        if isinstance(constructor, tuple):
            continue

        if isinstance(x, constructor):
            return constructor, output_type

    else:
        raise ValueError(
            "Input matrices to dot_product_mkl must be CSR, CSC, or BSR; "
            "COO is not supported"
        )


def _create_mkl_sparse(matrix):
    """
    Create MKL internal representation

    :param matrix: Sparse data in CSR or CSC format
    :type matrix: scipy.sparse.spmatrix

    :return ref, double_precision: Handle for the MKL internal representation
        and boolean for double precision and for complex dtype
    :rtype: sparse_matrix_t, bool, bool
    """

    double_precision, complex_type = _is_double(matrix)

    # Figure out which matrix creation function to use
    if is_csr(matrix):

        _check_scipy_index_typing(matrix)

        assert matrix.data.shape[0] == matrix.indices.shape[0]
        assert matrix.indptr.shape[0] == matrix.shape[0] + 1

        ref = _pass_mkl_handle_csr_csc(
            matrix,
            _create_functions[("csr", double_precision, complex_type)]
        )

    elif is_csc(matrix):

        _check_scipy_index_typing(matrix)

        assert matrix.data.shape[0] == matrix.indices.shape[0]
        assert matrix.indptr.shape[0] == matrix.shape[1] + 1

        ref = _pass_mkl_handle_csr_csc(
            matrix,
            _create_functions[("csc", double_precision, complex_type)]
        )

    elif is_bsr(matrix):

        _check_scipy_index_typing(matrix)

        ref = _create_mkl_sparse_bsr(matrix)

    else:
        raise ValueError("Matrix is not CSC, CSR, or BSR")

    return ref, double_precision, complex_type


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
    ret_val = handle_func(
        _ctypes.byref(ref),
        _ctypes.c_int(SPARSE_INDEX_BASE_ZERO),
        MKL.MKL_INT(data.shape[0]),
        MKL.MKL_INT(data.shape[1]),
        data.indptr[0:-1],
        data.indptr[1:],
        data.indices,
        data.data,
    )

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

    double_precision, complex_type = _is_double(matrix)
    handle_func = _create_functions[("bsr", double_precision, complex_type)]

    # Get the blocksize and check that the blocks are square
    _blocksize = matrix.blocksize[0]

    if _blocksize != matrix.blocksize[1]:
        raise ValueError(
            "MKL BSR representation requires square blocks; "
            "{matrix.blocksize} blocks provided"
        )

    if (
        (matrix.shape[0] % _blocksize != 0) or
        (matrix.shape[1] % _blocksize != 0)
    ):
        raise ValueError(
            f"BSR blocks {matrix.blocksize} do not align with dims "
            f"{matrix.shape}"
        )

    _block_rows = int(matrix.shape[0] / _blocksize)
    _block_cols = int(matrix.shape[1] / _blocksize)

    # Get the data block array structure
    _layout, _ = _get_numpy_layout(matrix.data)

    # Create a pointer for the output matrix
    ref = sparse_matrix_t()

    # Load into a MKL data structure and check return
    ret_val = handle_func(
        _ctypes.byref(ref),
        _ctypes.c_int(SPARSE_INDEX_BASE_ZERO),
        _ctypes.c_int(_layout),
        MKL.MKL_INT(_block_rows),
        MKL.MKL_INT(_block_cols),
        MKL.MKL_INT(_blocksize),
        matrix.indptr[0:-1],
        matrix.indptr[1:],
        matrix.indices,
        matrix.data,
    )

    # Check return
    _check_return_value(ret_val, handle_func.__name__)

    return ref


def _export_mkl(
    csr_mkl_handle,
    double_precision,
    complex_type=False,
    output_type="csr_matrix"
):
    """
    Export a MKL sparse handle of CSR or CSC type

    :param csr_mkl_handle: Handle for the MKL internal representation
    :type csr_mkl_handle: sparse_matrix_t
    :param double_precision: Use float64 if True, float32 if False.
        This MUST match the underlying float type - this
        defines a memory view, it does not cast.
    :type double_precision: bool
    :param output_type: The structure of the MKL handle
        (and therefore the type of scipy sparse to create)
    :type output_type: str

    :return: Sparse matrix in scipy format
    :rtype: scipy.spmatrix
    """

    output_type = output_type.lower()

    if output_type.startswith("bsr"):
        return _export_mkl_sparse_bsr(
            csr_mkl_handle,
            double_precision,
            complex_type=complex_type,
            output_type=output_type
        )

    elif output_type.startswith("csr") or output_type.startswith("csc"):
        out_func = _export_functions[
            (output_type[0:3], double_precision, complex_type)
        ]
        sp_matrix_constructor = _scipy_objs[output_type]

    else:
        raise ValueError("Only CSR, CSC, and BSR output types are supported")

    # Allocate for output
    (
        ordering,
        nrows,
        ncols,
        indptrb,
        indptren,
        indices,
        data
    ) = _allocate_for_export(double_precision)

    final_dtype = _output_dtypes[(double_precision, complex_type)]

    ret_val = out_func(
        csr_mkl_handle,
        _ctypes.byref(ordering),
        _ctypes.byref(nrows),
        _ctypes.byref(ncols),
        _ctypes.byref(indptrb),
        _ctypes.byref(indptren),
        _ctypes.byref(indices),
        _ctypes.byref(data),
    )

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
    index_dim = nrows if output_type.startswith("csr") else ncols

    # Construct a numpy array and add 0 to first position for
    # scipy.sparse's 3-array indexing
    indptrb = as_array(indptrb, shape=(index_dim,))
    indptren = as_array(indptren, shape=(index_dim,))

    indptren = _np.insert(indptren, 0, indptrb[0])
    nnz = indptren[-1] - indptrb[0]

    # If there are no non-zeros, return an empty matrix
    # If the number of non-zeros is insane, raise a ValueError
    if nnz == 0:
        return sp_matrix_constructor((nrows, ncols), dtype=final_dtype)
    elif nnz < 0 or nnz > ncols * nrows:
        raise ValueError(
            f"Matrix ({nrows} x {ncols}) is attempting to index {nnz} elements"
        )

    # Construct numpy arrays from data pointer and from indicies pointer
    data = _np.array(
        as_array(data, shape=(nnz * 2 if complex_type else nnz,)), copy=True
    )
    indices = _np.array(as_array(indices, shape=(nnz,)), copy=True)

    # Pack and return the matrix
    if complex_type:
        data = data.view(final_dtype)

    return sp_matrix_constructor(
        (data, indices, indptren),
        shape=(nrows, ncols),
    )


def _export_mkl_sparse_bsr(
    bsr_mkl_handle,
    double_precision,
    complex_type=False,
    output_type="bsr"
):
    """
    Export a BSR matrix from MKL's internal representation to scipy

    :param bsr_mkl_handle: MKL internal representation
    :type bsr_mkl_handle: sparse_matrix_t
    :param double_precision: Use float64 if True, float32 if False.
        This MUST match the underlying float type - this
        defines a memory view, it does not cast.
    :type double_precision: bool
    :return: Sparse BSR matrix
    :rtype:
    """

    # Allocate for output
    (
        ordering,
        nrows,
        ncols,
        indptrb,
        indptren,
        indices,
        data
    ) = _allocate_for_export(double_precision)
    block_layout = _ctypes.c_int()
    block_size = MKL.MKL_INT()

    # Set output
    out_func = _export_functions[
        (output_type[0:3], double_precision, complex_type)
    ]
    final_dtype = _output_dtypes[(double_precision, complex_type)]
    sp_matrix_constructor = _scipy_objs[output_type]

    ret_val = out_func(
        bsr_mkl_handle,
        _ctypes.byref(ordering),
        _ctypes.byref(block_layout),
        _ctypes.byref(nrows),
        _ctypes.byref(ncols),
        _ctypes.byref(block_size),
        _ctypes.byref(indptrb),
        _ctypes.byref(indptren),
        _ctypes.byref(indices),
        _ctypes.byref(data),
    )

    # Check return
    _check_return_value(ret_val, out_func.__name__)

    # Get matrix dims
    ncols, nrows, block_size = ncols.value, nrows.value, block_size.value
    index_dim, block_dims = nrows, (block_size, block_size)
    ncols, nrows = ncols * block_size, nrows * block_size

    # If any axis is 0 return an empty matrix
    if nrows == 0 or ncols == 0:
        return sp_matrix_constructor(
            (nrows, ncols), dtype=final_dtype, blocksize=block_dims
        )

    ordering = "F" if ordering.value == LAYOUT_CODE_F else "C"

    # Construct a numpy array and add 0 to first position for
    # scipy.sparse's 3-array indexing
    indptrb = as_array(indptrb, shape=(index_dim,))
    indptren = as_array(indptren, shape=(index_dim,))

    indptren = _np.insert(indptren, 0, indptrb[0])

    nnz_blocks = indptren[-1] - indptrb[0]
    nnz = nnz_blocks * (block_size**2)

    # If there's no non-zero data, return an empty matrix
    if nnz_blocks == 0:
        return sp_matrix_constructor(
            (nrows, ncols), dtype=final_dtype, blocksize=block_dims
        )

    elif nnz_blocks < 0 or nnz > ncols * nrows:

        raise ValueError(
            f"Matrix ({nrows} x {ncols}) is attempting to index {nnz} "
            f"elements as {block_size} {(block_size, block_size)} blocks"
        )

    nnz_row_block = block_size if not complex_type else block_size * 2
    data = _np.array(
        as_array(data, shape=(nnz_blocks, block_size, nnz_row_block)),
        copy=True,
        order=ordering,
    )
    indices = _np.array(as_array(indices, shape=(nnz_blocks,)), copy=True)

    if complex_type:
        data = data.view(final_dtype)

    return sp_matrix_constructor(
        (data, indices, indptren),
        shape=(nrows, ncols),
        blocksize=block_dims,
    )


def _allocate_for_export(double_precision):
    """
    Get pointers for output from MKL internal representation
    :param double_precision: Allocate an output pointer of doubles
    :type double_precision: bool
    :return: ordering, nrows, ncols, indptrb, indptren, indices, data
    :rtype:
        c_int,
        MKL_INT,
        MKL_INT,
        MKL_INT*,
        MKL_INT*,
        MKL_INT*,
        c_float|c_double*
    """
    # Create the pointers for the output data
    indptrb = _ctypes.POINTER(MKL.MKL_INT)()
    indptren = _ctypes.POINTER(MKL.MKL_INT)()
    indices = _ctypes.POINTER(MKL.MKL_INT)()

    ordering = _ctypes.c_int()
    nrows = MKL.MKL_INT()
    ncols = MKL.MKL_INT()

    data = (
        _ctypes.POINTER(_ctypes.c_double)()
        if double_precision
        else _ctypes.POINTER(_ctypes.c_float)()
    )

    return ordering, nrows, ncols, indptrb, indptren, indices, data


def _check_return_value(ret_val, func_name):
    """
    Check the return value from a sparse function

    :param ret_val:
    :param func_name:
    :return:
    """

    if ret_val != 0:
        err_msg = "{fn} returned {v} ({e})".format(
            fn=func_name, v=ret_val, e=RETURN_CODES[ret_val]
        )
        if ret_val == 2:
            err_msg += "; " + ILP64_MSG
        raise ValueError(err_msg)
    elif MKL.MKL_DEBUG:
        print(
            "{fn} returned {v} ({e})".format(
                fn=func_name, v=ret_val, e=RETURN_CODES[ret_val]
            )
        )
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
    ret_val = MKL._mkl_sparse_convert_csr(
        ref_handle, _ctypes.c_int(10), _ctypes.byref(csr_ref)
    )

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
        raise ValueError(
            f"Matrices must be 2d: {matrix_a.shape} * {matrix_b.shape} "
            f"is not valid"
        )

    invalid_ndims = not (a_2d or a_vec) or not (b_2d or b_vec)
    invalid_align = (
        matrix_a.shape[1] if not matrix_a.ndim == 1 else matrix_a.shape[0]
    ) != matrix_b.shape[0]

    # Check to make sure that this multiplication can work
    if invalid_align or invalid_ndims:
        raise ValueError(
            f"Matrix alignment error: {matrix_a.shape} * {matrix_b.shape} "
            "is not valid"
        )


def _cast_to(matrix, dtype):
    """
    Make a copy of the array as double precision floats
    or return the reference if it already is
    """
    return matrix.astype(dtype) if matrix.dtype != dtype else matrix


def _is_valid_dtype(matrix, complex_dtype=False, all_dtype=False):
    """Check to see if it's a usable float dtype"""
    if all_dtype:
        return matrix.dtype in NUMPY_FLOAT_DTYPES + NUMPY_COMPLEX_DTYPES
    elif complex_dtype:
        return matrix.dtype in NUMPY_COMPLEX_DTYPES
    else:
        return matrix.dtype in NUMPY_FLOAT_DTYPES


def _type_check(matrix_a, matrix_b=None, cast=False, allow_complex=True):
    """
    Make sure that both matrices are single precision floats or both are
    double precision floats.
    If not, convert to double precision floats if cast is True,
    or raise an error if cast is False
    """

    _n_complex = _np.iscomplexobj(matrix_a) + _np.iscomplexobj(matrix_b)

    if not allow_complex and _n_complex > 0:
        raise ValueError("Complex datatypes are not supported")

    # If there's no matrix B and matrix A is valid dtype, return it
    if matrix_b is None and _is_valid_dtype(matrix_a, all_dtype=True):
        return matrix_a

    # If matrix A is complex but not csingle or cdouble, and cast is True,
    # convert it to a cdouble
    elif matrix_b is None and cast and _n_complex == 1:
        return _cast_to(matrix_a, _np.cdouble)

    # If matrix A is real but not float32 or float64, and cast is True,
    # convert it to a float64
    elif matrix_b is None and cast:
        return _cast_to(matrix_a, _np.float64)

    # Raise an error - the dtype is invalid and cast is False
    elif matrix_b is None:
        raise ValueError(
            "Matrix data type must be float32, float64, csingle, or cdouble; "
            f"{matrix_a.dtype} provided"
        )

    # If Matrix A & B have the same valid dtype, return them
    if (
        _is_valid_dtype(matrix_a, all_dtype=True) and
        matrix_a.dtype == matrix_b.dtype
    ):
        return matrix_a, matrix_b

    # If neither matrix is complex and cast is True, convert to float64s
    # and return them
    elif cast and _n_complex == 0:
        debug_print(
            f"Recasting matrix data types {matrix_a.dtype} and "
            f"{matrix_b.dtype} to np.float64"
        )
        return _cast_to(matrix_a, _np.float64), _cast_to(matrix_b, _np.float64)

    # If both matrices are complex and cast is True, convert to cdoubles
    # and return them
    elif cast and _n_complex == 2:
        debug_print(
            f"Recasting matrix data types {matrix_a.dtype} and "
            f"{matrix_b.dtype} to _np.cdouble"
        )
        return _cast_to(matrix_a, _np.cdouble), _cast_to(matrix_b, _np.cdouble)

    # Cast reals and complex matrices together
    elif (
        cast and
        _n_complex == 1 and
        _is_valid_dtype(matrix_a, complex_dtype=True)
    ):
        debug_print(
            f"Recasting matrix data type {matrix_b.dtype} to {matrix_a.dtype}"
        )
        return matrix_a, _cast_to(matrix_b, matrix_a.dtype)

    elif (
        cast and
        _n_complex == 1 and
        _is_valid_dtype(matrix_b, complex_dtype=True)
    ):
        debug_print(
            f"Recasting matrix data type {matrix_a.dtype} to {matrix_b.dtype}"
        )
        return _cast_to(matrix_a, matrix_b.dtype), matrix_b

    elif cast and _n_complex == 1:
        debug_print(
            f"Recasting matrix data type {matrix_a.dtype} and {matrix_b.dtype}"
            f" to np.cdouble"
        )
        return _cast_to(matrix_a, _np.cdouble), _cast_to(matrix_b, _np.cdouble)

    # If cast is False, can't cast anything together
    elif not cast:
        raise ValueError(
            "Matrix data type must be float32, float64, csingle, or cdouble, "
            + "and must be the same if cast=False; "
            + f"{matrix_a.dtype} & {matrix_b.dtype} provided"
        )


def _mkl_scalar(scalar, complex_type, double_precision):
    """
    Turn complex scalars into appropriate precision MKL scalars
    or leave floats as floats
    """

    scalar = 1.0 if scalar is None else scalar

    if complex_type and double_precision:
        return MKL_Complex16(complex(scalar))
    elif complex_type:
        return MKL_Complex8(complex(scalar))
    else:
        return float(scalar)


def _out_matrix(shape, dtype, order="C", out_arr=None, out_t=False):
    """
    Create an all-zero matrix or check to make sure that
    the provided output array matches

    :param shape: Required output shape
    :type shape: tuple(int)
    :param dtype: Required output data type
    :type dtype: _np.dtype
    :param order: Array order (row or column-major)
    :type order: str
    :param out_arr: Provided output array
    :type out_arr: _np.ndarray
    :param out_t: Out array has been transposed
    :type out_t: bool
    :return: Array
    :rtype: _np.ndarray
    """

    out_t = False if out_t is None else out_t

    # If there's no output array allocate a new array and return it
    if out_arr is None:
        return _np.zeros(shape, dtype=dtype, order=order)

    # Check and make sure the order is correct
    # Note 1d arrays have both flags set
    if order == "C":
        _order_match = out_arr.flags["C_CONTIGUOUS"]
    else:
        _order_match = out_arr.flags["F_CONTIGUOUS"]

    # If there are any incompatible parameters, raise an error
    # with the provided and required array parameters
    # Flip them if out_T is set so that the original values and
    # the values which would have to be provided are correct
    if (
        shape != out_arr.shape
        or dtype != out_arr.dtype
        or not _order_match
        or not out_arr.data.contiguous
    ):
        _c_contig = out_arr.flags["C_CONTIGUOUS"]
        _f_contig = out_arr.flags["F_CONTIGUOUS"]

        if not out_t or out_arr.ndim == 1:
            _err_shape = out_arr.shape
            _req_shape = shape
            _err_order = "C" if _c_contig else "F"
            _req_order = order
        else:
            _err_shape = out_arr.shape[::-1]
            _req_shape = shape[::-1]
            _err_order = "F" if _c_contig and not _f_contig else "C"
            _req_order = "C" if order == "F" else "F"

        try:
            _req_dtype = dtype.__name__
        except AttributeError:
            _req_dtype = dtype.name

        raise ValueError(
            "Provided out array is "
            f"{_err_shape} {out_arr.dtype} [{_err_order}"
            f"_{'CONTIGUOUS' if out_arr.data.contiguous else 'NONCONTIGUOUS'}]"
            f" and product requires "
            f"{_req_shape} {_req_dtype} [{_req_order}_CONTIGUOUS]"
        )

    else:
        return out_arr


def _is_dense_vector(m_or_v):
    return not _spsparse.issparse(m_or_v) and (
        (m_or_v.ndim == 1) or ((m_or_v.ndim == 2) and min(m_or_v.shape) == 1)
    )


def _is_double(arr):
    """
    Return bools corresponding to float precision & real/complex.

    :param arr: 2d array
    :type arr: np.ndarray, scipy.sparse.spmatrix
    :return: False(single) or True(double), False(real) or True(complex)
    :rtype: bool, bool
    """

    # Figure out which dtype for data
    if arr.dtype == _np.float32:
        return False, False
    elif arr.dtype == _np.float64:
        return True, False
    elif arr.dtype == _np.csingle:
        return False, True
    elif arr.dtype == _np.cdouble:
        return True, True
    else:
        raise ValueError(
            "Only float32, float64, csingle, and cdouble dtypes are supported"
        )


def _is_allowed_sparse_format(matrix):
    """
    Return True if the matrix is dense or a sparse format we can
    turn into an MKL object. False otherwise.
    :param matrix:
    :return:
    :rtype: bool
    """
    if _spsparse.issparse(matrix):
        return is_csr(matrix) or is_csc(matrix) or is_bsr(matrix)
    else:
        return True


def _empty_output_check(matrix_a, matrix_b):
    """Check for trivial cases where an empty array should be produced"""

    # One dimension is zero
    if min([*matrix_a.shape, *matrix_b.shape]) == 0:
        return True

    # The sparse array is empty
    elif (
        _spsparse.issparse(matrix_a)
        and min(matrix_a.data.size, matrix_a.indices.size) == 0
    ):
        return True
    elif (
        _spsparse.issparse(matrix_b)
        and min(matrix_b.data.size, matrix_b.indices.size) == 0
    ):
        return True

    # Neither trivial condition
    else:
        return False
