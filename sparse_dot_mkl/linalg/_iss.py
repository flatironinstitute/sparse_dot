from sparse_dot_mkl._mkl_interface import (MKL, _libmkl, sparse_matrix_t, _is_double, _create_mkl_sparse,
                                           _destroy_mkl_handle, matrix_descr)

import warnings
import numpy as _np
import scipy.sparse as _sps
import ctypes as _ctypes
from numpy.ctypeslib import ndpointer

DEFAULT_ATOL = 0.0
DEFAULT_RTOL = 1e-6
DEFAULT_MAX_ITER = 1000

class _MKL_ISS:

    # https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/iterative-sparse-solvers-based-on-reverse-communication-interface-rci-iss.html

    # CG Solver
    _dcg_init = _libmkl.dcg_init
    _dcg_check = _libmkl.dcg_check
    _dcg = _libmkl.dcg
    _dcg_get = _libmkl.dcg_get

    # FGMRES Solver
    _dfgmres_init = _libmkl.dfgmres_init
    _dfgmres_check = _libmkl.dfgmres_check
    _dfgmres = _libmkl.dfgmres
    _dfgmres_get = _libmkl.dfgmres_get

    @classmethod
    def _set_int_type(cls):
        
        cls._dcg_init.argtypes = cls._iss_argtypes()
        cls._dcg_init.restype = None

        cls._dcg_check.argtypes = cls._iss_argtypes()
        cls._dcg_check.restype = None

        cls._dcg.argtypes = cls._iss_argtypes()
        cls._dcg.restype = None

        cls._dcg_get.argtypes = cls._iss_argtypes() + [_ctypes.POINTER(MKL.MKL_INT)]
        cls._dcg_get.restype = None

        cls._dfgmres_init.argtypes = cls._iss_argtypes()
        cls._dfgmres_init.restype = None

        cls._dfgmres_check.argtypes = cls._iss_argtypes()
        cls._dfgmres_check.restype = None

        cls._dfgmres.argtypes = cls._iss_argtypes()
        cls._dfgmres.restype = None

        cls._dfgmres_get.argtypes = cls._iss_argtypes() + [_ctypes.POINTER(MKL.MKL_INT)]
        cls._dfgmres_get.restype = None        

    @staticmethod
    def _iss_argtypes():

        return [_ctypes.POINTER(MKL.MKL_INT),
                ndpointer(dtype=_ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                ndpointer(dtype=_ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                _ctypes.POINTER(MKL.MKL_INT),
                ndpointer(dtype=MKL.MKL_INT, shape=(128,), flags='C_CONTIGUOUS'),
                ndpointer(dtype=_ctypes.c_double, shape=(128,), flags='C_CONTIGUOUS'),
                ndpointer(dtype=_ctypes.c_double, flags='C_CONTIGUOUS')]


# Set argtypes based on MKL interface type
_MKL_ISS._set_int_type()

class MKLIterativeSparseSolver:
    """
    Iterative solver object.

    Specific implementations must implement `set_initial_parameters`, `initialize_solver`,
        `check_solver`, `solve_iteration`, `get_solution`, and `update_tmp`.

    Specific implementations may implement `solve` and `update_preconditioner`.
    """

    A = None
    x = None
    b = None
    n = None

    ipar = None
    dpar = None
    tmp = None

    destroy_A_handle = False
    matrix_A_descr = None

    max_iter = None
    current_iter = None
    verbose = False
    
    a_tol = DEFAULT_ATOL
    r_tol = DEFAULT_RTOL

    def __init__(self, A, b, x=None, ipar=None, dpar=None, tmp=None, max_iter=None, a_tol=None, r_tol=None,
                 verbose=False, n=None):

        # Set defaults if Nones passed
        max_iter = DEFAULT_MAX_ITER if max_iter is None else max_iter
        a_tol = DEFAULT_ATOL if a_tol is None else a_tol
        r_tol = DEFAULT_RTOL if r_tol is None else r_tol

        # Set solver parameters into this object
        self.current_iter, self.max_iter, self.verbose = 0, max_iter, verbose
        self.a_tol, r_tol = a_tol, r_tol
        self.set_sparse_matrix_descr()

        # Set the parameter arrays into this object
        self.ipar, self.dpar, self.tmp = ipar, dpar, tmp

        # If passing in a sparse matrix handle, check that n is also passed
        if isinstance(A, sparse_matrix_t) and n is None:
            raise ValueError("If A is a MKL sparse handle, n must be passed as well")

        # If A is already a sparse matrix handle, just use it as is (and don't deallocate it at the end)
        elif isinstance(A, sparse_matrix_t):
            self.A = A
            self.destroy_A_handle = False

        # If A is a scipy sparse matrix, convert it to a sparse matrix handle and flag it for cleanup
        elif _sps.isspmatrix_csr(A):

            # Check dtype
            try:
                if not _is_double(A):
                    raise ValueError
            except ValueError:
                raise ValueError("Matrix A must be a double-precision scipy CSR matrix or a MKL sparse handle")

            if n is not None and A.shape[1] != n:
                _msg = "n = {n} does not align with matrix A ({a_sh})".format(n=n, a_sh=A.shape)
                raise ValueError(_msg)
            
            elif n is None:
                n = A.shape[1]

            # Make handle
            self.A, _ = _create_mkl_sparse(A)
            self.destroy_A_handle = True

        else:
            raise ValueError("Matrix A must be a double-precision scipy CSR matrix or a MKL sparse handle")

        self.n = n

        # If an initial value vector is passed, make sure the array is 1d
        # Also explicitly copy X (with flatten) so it does not change in place
        if x is not None and ((x.ndim == 2 and x.shape[1] == 1) or (x.ndim == 1)):
            x = x.flatten()
        elif x is not None:
            _msg = "x must be 1d or a single column 2d array; {sh} passed".format(sh=x.shape)
            raise ValueError(_msg)

        # If there's no solution vector passed, create one of zeros
        elif x is None:
            x = _np.zeros(self.n, dtype=_ctypes.c_double)

        # Also check b but don't copy it unless absolutely necessary
        self.b = _check_vector(b)

        # Run the initialization
        self.initialize_solver()
        self.set_initial_parameters()
        self.check_solver()

    # Magic functions for context manager
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):

        # Dereference the parameter and temp arrays in python
        del self.ipar
        del self.dpar
        del self.tmp

        # Destroy the matrix handle if necessary
        if self.destroy_A_handle:
            _destroy_mkl_handle(self.A)

        # Free internal MKL buffers
        MKL._mkl_free_buffers()

    def set_sparse_matrix_descr(self, sparse_matrix_type_t=20, sparse_fill_mode_t=0, sparse_diag_type_t=0):
        self.matrix_A_descr = matrix_descr(sparse_matrix_type_t, sparse_fill_mode_t, sparse_diag_type_t)

    def initialize_solver(self):
        raise NotImplementedError

    def check_solver(self):
        raise NotImplementedError

    def solve_iteration(self):
        raise NotImplementedError

    def get_solution(self):
        raise NotImplementedError

    def set_initial_parameters(self):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):

            solve_status = self.solve_iteration()

            # Update the tmp if the solver returns 1; otherwise it's the user's problem
            if solve_status == 1:
                self.update_tmp()
               
            self.current_iter += 1

            return solve_status

    def update_tmp(self):
        raise NotImplementedError

    def update_preconditioner(self):
        raise NotImplementedError

    def user_test(self):
        pass


def _check_vector(vec, allow_none=False):
    """
    
    Make sure vector is a 1d contiguous vector.
    Reshape (N, 1) vectors if needed.

    :param vec: Numpy vector
    :type vec: np.ndarray
    :param allow_none: Allow vec to be None (and return None), defaults to False
    :type allow_none: bool, optional
    :raises ValueError: Raise a ValueError if anything other then a (N, ) or (N, 1) vector is passed
    :return: Vector in 1d
    :rtype: np.ndarray, None
    """

    if vec is None and allow_none: 
        return None
    elif vec is None:
        raise ValueError("Vector cannot be None")
    elif vec.ndim == 2 and vec.shape[1] == 1:
        return vec.ravel()
    elif vec.ndim != 1:
        _msg = "Vector must be 1d or a single column 2d array; {sh} passed".format(sh=vec.shape)
        raise ValueError(_msg)
    elif not vec.data.contiguous:
        return vec.ravel()
    else:
        return vec


def _create_empty_temp_arrays(ipar=None, dpar=None, tmp=None, tmp_size=None):
    """
    Create and return arrays for solver internals

    :param x: Solution vector x, defaults to None
    :type x: np.ndarray, optional
    :param ipar: Integer parameter vector with settings for solver, defaults to None
    :type ipar: np.ndarray, optional
    :param dpar: Double parameter vector with settings for solver, defaults to None
    :type dpar: np.ndarray, optional
    :param tmp: Temp vector for solver computations, defaults to None
    :type tmp: np.ndarray, optional
    :param tmp_size: Size of temp vector to alloate, defaults to None. Must be passed if tmp is None.
    :type tmp_size: int, optional

    :return ipar: Integer parameter vector with settings for solver
    :type ipar: np.ndarray
    :return dpar: Double parameter vector with settings for solver
    :type dpar: np.ndarray
    :return tmp: Temp vector for solver computations
    :type tmp: np.ndarray
    """

    ipar = _np.zeros(128, dtype=MKL.MKL_INT_NUMPY) if ipar is None else ipar
    dpar = _np.zeros(128, dtype=_ctypes.c_double) if dpar is None else dpar

    if tmp is not None and tmp.size != tmp_size:
        _msg = "TMP vector size must be {t}; {t_1} passed".format(t=tmp_size, t_1=tmp.size)
        raise ValueError(_msg)
    elif tmp is None and tmp_size is None:
        raise ValueError("tmp or tmp_size must be provided")
    else:
        tmp = _np.zeros(tmp_size, dtype=_ctypes.c_double)

    return ipar, dpar, tmp
