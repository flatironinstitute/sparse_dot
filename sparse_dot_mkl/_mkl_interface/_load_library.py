import ctypes as _ctypes
import ctypes.util as _ctypes_util
import os

IMPORT_ERRORS = (OSError, ImportError)


def _try_load_mkl_rt(path=None):
    # Check each of these library names
    # Also include derivatives because windows find_library implementation
    # won't match partials
    for so_file in [
        "libmkl_rt.so",
        "libmkl_rt.dylib",
        "mkl_rt.dll"
    ] + [
        f"mkl_rt.{i}.dll" for i in range(5, 0, -1)
    ] + [
        f"libmkl_rt.so.{i}" for i in range(5, 0, -1)
    ]:
        try:
            # If this finds anything, break out of the loop
            return _ctypes.cdll.LoadLibrary(os.path.join(path, so_file))

        except IMPORT_ERRORS:
            pass

    return None


def mkl_library():

    # Load mkl_spblas through the libmkl_rt common interface
    _libmkl = None

    # Use MKL_RT env (useful if there are multiple MKL binaries in path)
    if 'MKL_RT' in os.environ:
        try:
            _libmkl = _ctypes.cdll.LoadLibrary(os.environ['MKL_RT'])
            return _libmkl
        except IMPORT_ERRORS:
            pass

    try:
        _so_file = _ctypes_util.find_library("mkl_rt")

        if _so_file is None:
            # Apparently this is gonna be an iterative thing
            # that the MKL library does
            for i in range(5, 0, -1):
                _so_file = _ctypes_util.find_library(f"mkl_rt.{i}")
                if _so_file is not None:
                    break

        if _so_file is None:
            # For some reason, find_library is not checking
            # LD_LIBRARY_PATH
            # If the ctypes.util approach doesn't work,
            # try this (crude) approach
            _libmkl = _try_load_mkl_rt("")

        else:
            _libmkl = _ctypes.cdll.LoadLibrary(_so_file)

        if _libmkl is None:
            oneapi_root = os.getenv("ONEAPI_ROOT")

            if oneapi_root is not None:
                for mkl_arch in ["intel64", "ia32"]:
                    _libmkl = _try_load_mkl_rt(
                        os.path.join(
                            oneapi_root,
                            "mkl",
                            "latest",
                            "redist",
                            mkl_arch
                        )
                    )

                    if _libmkl is not None:
                        break

        if _libmkl is None:
            raise ImportError("mkl_rt not found.")

    # Couldn't find anything to import
    # Raise the ImportError
    except IMPORT_ERRORS as err:
        raise ImportError(
            "Unable to load the MKL libraries through "
            "libmkl_rt. Try setting $LD_LIBRARY_PATH to the "
            "LD library path or $MKL_RT to the libmkl_rt.so library "
            "file directly. " + str(err)
        )

    return _libmkl
