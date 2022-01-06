import os

# Workaround for this stupid sklearn thing
if 'KMP_INIT_AT_FORK' in os.environ:
    _sklearn_env = os.environ['KMP_INIT_AT_FORK']
    del os.environ['KMP_INIT_AT_FORK']
else:
    _sklearn_env = None

from scipy import sparse as _spsparse
import numpy as _np
import ctypes as _ctypes
from ._constants import *
from ._structs import *
from ._cfunctions import MKL
from ._common import (_create_mkl_sparse, _export_mkl, _check_return_value, _convert_to_csr, _is_allowed_sparse_format,
                      _is_double, _destroy_mkl_handle, _order_mkl_handle, debug_print, debug_timer, _type_check,
                      _empty_output_check, _sanity_check, _output_dtypes, _get_numpy_layout, _out_matrix, _mkl_scalar,
                      _is_dense_vector, print_mkl_debug, set_debug_mode, get_version_string)

def _validate_dtype():
    """
    Test to make sure that this library works by creating a random sparse array in CSC format,
    then converting it to CSR format and making sure is has not raised an exception.

    """

    test_array = _spsparse.random(5, 5, density=0.5, format="csc", dtype=_np.float32, random_state=50)
    test_comparison = test_array.A

    csc_ref, precision_flag, _ = _create_mkl_sparse(test_array)

    try:
        csr_ref = _convert_to_csr(csc_ref)
        final_array = _export_mkl(csr_ref, precision_flag)
        if not _np.allclose(test_comparison, final_array.A):
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
    MKL._set_int_type(_ctypes.c_longlong, _np.int64)

    try:
        _validate_dtype()
    except ValueError as err:

        MKL._set_int_type(_ctypes.c_int, _np.int32)

        try:
            _validate_dtype()
        except ValueError:
            raise ImportError("Unable to set MKL numeric type")


if MKL.MKL_INT is None:
    _empirical_set_dtype()

if _sklearn_env is not None:
    os.environ['KMP_INIT_AT_FORK'] = _sklearn_env
