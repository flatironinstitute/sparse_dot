from sparse_dot_mkl._mkl_interface import (
    MKL,
    sparse_matrix_t,
    _is_double,
    _is_dense_vector,
    _create_mkl_sparse,
    _destroy_mkl_handle,
    matrix_descr,
    is_csr
)

import numpy as _np
import scipy.sparse as _sps
import ctypes as _ctypes

DEFAULT_ATOL = 0.0
DEFAULT_RTOL = 1e-6
DEFAULT_MAX_ITER = 1000

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
    verbose = False

    a_tol = DEFAULT_ATOL
    r_tol = DEFAULT_RTOL

    final_code = None

    def __init__(
        self,
        A,
        b,
        x=None,
        ipar=None,
        dpar=None,
        tmp=None,
        max_iter=DEFAULT_MAX_ITER,
        a_tol=DEFAULT_ATOL,
        r_tol=DEFAULT_RTOL,
        verbose=False,
        n=None,
    ):

        # Set solver parameters into this object
        self.current_iter, self.max_iter = 0, max_iter
        self.a_tol, r_tol = a_tol, r_tol
        self.set_sparse_matrix_descr()

        # Set the parameter arrays into this object
        self.ipar, self.dpar, self.tmp = ipar, dpar, tmp

        # If passing in a sparse matrix handle, check that n is also passed
        if isinstance(A, sparse_matrix_t) and n is None:
            raise ValueError(
                "If A is a MKL sparse handle, n must be passed as well"
            )

        # If A is already a sparse matrix handle, just use it
        # as is (and don't deallocate it at the end)
        elif isinstance(A, sparse_matrix_t):
            self.A = A
            self.destroy_A_handle = False

        # If A is a scipy sparse matrix, convert it to a sparse matrix
        # handle and flag it for cleanup
        elif is_csr(A):

            # Check dtype
            try:
                if not _is_double(A):
                    raise ValueError
            except ValueError:
                raise ValueError(
                    "Matrix A must be a double-precision scipy CSR matrix "
                    "or a MKL sparse handle"
                )

            if n is not None and A.shape[1] != n:
                raise ValueError(
                    f"n = {n} does not align with matrix A ({A.shape})"
                )

            elif n is None:
                n = A.shape[1]

            # Make handle
            self.A, _, _ = _create_mkl_sparse(A)
            self.destroy_A_handle = True

        else:
            raise ValueError(
                "Matrix A must be a double-precision scipy CSR matrix "
                "or a MKL sparse handle"
            )

        self.n = n

        # If an initial value vector is passed, make sure the array is 1d
        # Also explicitly copy X (with flatten) so it does not change in place
        if x is not None and ((x.ndim == 2 and x.shape[1] == 1) or (x.ndim == 1)):
            x = x.flatten()

        elif x is not None:
            raise ValueError(
                "x must be 1d or a single column 2d array; "
                f"{x.shape} passed"
            )

        # If there's no solution vector passed, create one of zeros
        elif x is None:
            x = _np.zeros(self.n, dtype=_ctypes.c_double)

        # Also check b but don't copy it unless necessary
        if _is_dense_vector(b):

            if b.ndim == 2 and b.shape[1] == 1:
                self.b = b.ravel()
            elif b.ndim != 1:
                raise ValueError(
                    "B must be 1d or a single column 2d array; "
                    f"{b.shape} provided"
                )
            elif not b.data.contiguous:
                self.b = b.ravel()
            else:
                self.b = b
        else:
            raise ValueError(
                "B must be a dense 1d or a single column 2d array; "
                f"{type(b)} {b.shape} provided"
            )

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

    def set_sparse_matrix_descr(
        self,
        sparse_matrix_type_t=20,
        sparse_fill_mode_t=0,
        sparse_diag_type_t=0
    ):
        self.matrix_A_descr = matrix_descr(
            sparse_matrix_type_t,
            sparse_fill_mode_t,
            sparse_diag_type_t
        )

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


def _create_empty_temp_arrays(
    ipar=None,
    dpar=None,
    tmp=None,
    tmp_size=None
):
    """
    Create and return arrays for solver internals

    :param x: Solution vector x, defaults to None
    :type x: np.ndarray, optional
    :param ipar: Integer parameter vector with settings for solver,
        allocates new if None, defaults to None
    :type ipar: np.ndarray, optional
    :param dpar: Double parameter vector with settings for solver,
        allocates new if None, defaults to None
    :type dpar: np.ndarray, optional
    :param tmp: Temp vector for solver computations,
        allocates new if None, defaults to None
    :type tmp: np.ndarray, optional
    :param tmp_size: Size of temp vector to alloate, defaults to None.
        Must be passed if tmp is None.
    :type tmp_size: int, optional
    :return ipar: Integer parameter vector with settings for solver
    :rtype ipar: np.ndarray
    :return dpar: Double parameter vector with settings for solver
    :rtype dpar: np.ndarray
    :return tmp: Temp vector for solver computations
    :rtype tmp: np.ndarray
    """

    if ipar is None:
        ipar = _np.zeros(128, dtype=MKL.MKL_INT_NUMPY)
    
    if dpar is None:
        dpar = _np.zeros(128, dtype=_ctypes.c_double)

    if tmp is not None and tmp.size != tmp_size:
        raise ValueError(
            f"TMP vector size must be {tmp_size}; {tmp.size} passed"
        )
    elif tmp is None and tmp_size is None:
        raise ValueError("tmp or tmp_size must be provided")
    else:
        tmp = _np.zeros(tmp_size, dtype=_ctypes.c_double)

    return ipar, dpar, tmp


class ConvergenceWarning(UserWarning):
    pass
