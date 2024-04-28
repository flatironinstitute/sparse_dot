import os

# Workaround for this stupid sklearn thing
if "KMP_INIT_AT_FORK" in os.environ:
    _sklearn_env = os.environ["KMP_INIT_AT_FORK"]
    del os.environ["KMP_INIT_AT_FORK"]
else:
    _sklearn_env = None

from scipy import sparse as _spsparse
import numpy as _np
import ctypes as _ctypes
import warnings as _warnings
from ._constants import *
from ._structs import (
    MKL_Complex8,
    MKL_Complex16,
    matrix_descr,
    sparse_matrix_t
)
from ._cfunctions import (
    MKL,
    mkl_set_interface_layer
)
from ._common import (
    _create_mkl_sparse,
    _export_mkl,
    _check_return_value,
    _convert_to_csr,
    _is_allowed_sparse_format,
    _is_double,
    _destroy_mkl_handle,
    _order_mkl_handle,
    debug_print,
    debug_timer,
    _type_check,
    _empty_output_check,
    _sanity_check,
    _output_dtypes,
    _get_numpy_layout,
    _out_matrix,
    _mkl_scalar,
    _is_dense_vector,
    print_mkl_debug,
    set_debug_mode,
    get_version_string,
    is_csr,
    is_csc,
    is_bsr,
    sparse_output_type
)


def _validate_dtype():
    """
    Test to make sure that this library works by creating arandom sparse array
    in CSC format, then converting it to CSR format and making sure is has not
    raised an exception.
    """

    test_array = _spsparse.random(
        5, 5, density=0.5, format="csc", dtype=_np.float32, random_state=50
    )
    test_comparison = test_array.A

    csc_ref, precision_flag, _ = _create_mkl_sparse(test_array)

    try:
        csr_ref = _convert_to_csr(csc_ref)

        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=RuntimeWarning)
            final_array = _export_mkl(
                csr_ref,
                precision_flag,
                output_type='csr_matrix'
            )

        if not _np.allclose(test_comparison, final_array.A):
            raise ValueError("Match failed after matrix conversion")
        _destroy_mkl_handle(csr_ref)
    finally:
        _destroy_mkl_handle(csc_ref)


def _empirical_set_dtype():
    """
    Define dtypes empirically
    Basically just try with int64s and if that doesn't work try with int32s
    There's a way to do this with intel's mkl helper package but I don't want
    to add the dependency
    """
    MKL._set_int_type(_ctypes.c_longlong, _np.int64)

    try:
        _validate_dtype()
    except ValueError:
        MKL._set_int_type(_ctypes.c_int, _np.int32)

        try:
            _validate_dtype()
        except ValueError:
            raise ImportError("Unable to set MKL numeric type")


_mkl_interface_env = os.getenv('MKL_INTERFACE_LAYER')


# Check to make sure that the MKL_Set_Interface_Layer call was correct
# And fail back to 32bit if it wasn't
if _mkl_interface_env == 'ILP64':
    try:
        _validate_dtype()
    except (ValueError, RuntimeError):
        _warnings.warn(
            "MKL_INTERFACE_LAYER=ILP64 failed to set MKL interface; "
            "64-bit integer support unavailable",
            RuntimeWarning
        )
        MKL._set_int_type(_ctypes.c_int, _np.int32)
        _validate_dtype()
elif _mkl_interface_env == 'LP64':
    _validate_dtype()
elif MKL.MKL_INT is None and _mkl_interface_env is not None:
    _warnings.warn(
        f"MKL_INTERFACE_LAYER value {_mkl_interface_env} invalid; "
        "set 'ILP64' or 'LP64'",
        RuntimeWarning
    )
    _empirical_set_dtype()
elif MKL.MKL_INT is None:
    _empirical_set_dtype()


if _sklearn_env is not None:
    os.environ["KMP_INIT_AT_FORK"] = _sklearn_env
