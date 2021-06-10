from sparse_dot_mkl._mkl_interface import MKL
from sparse_dot_mkl._sparse_vector import _sparse_dense_vector_mult
from sparse_dot_mkl.iterative_sparse_solver._iss import _MKL_ISS, MKLIterativeSparseSolver, _create_empty_temp_arrays

import numpy as _np
import ctypes as _ctypes
import warnings

def dcg_init(n, b, x=None, ipar=None, dpar=None, tmp=None):
    """
    CG solver initialization. 
    If x, ipar, dpar, and tmp are not passed, will be initalized with standard values.
    
    :param n: Number of predictor variables to model
    :type n: int
    :param b: Right-hand vector b
    :type b: np.ndarray

    :param x: Solution vector x, defaults to None
    :type x: np.ndarray, optional
    :param ipar: Integer parameter vector with settings for solver, defaults to None
    :type ipar: np.ndarray, optional
    :param dpar: Double parameter vector with settings for solver, defaults to None
    :type dpar: np.ndarray, optional
    :param tmp: Temp vector for solver computations, defaults to None
    :type tmp: np.ndarray, optional

    :return x: Solution vector x
    :rtype x: np.ndarray
    :return ipar: Integer parameter vector with settings for solver
    :type ipar: np.ndarray
    :return dpar: Double parameter vector with settings for solver
    :type dpar: np.ndarray
    :return tmp: Temp vector for solver computations
    :type tmp: np.ndarray
    """
  
    x = _np.zeros(n, dtype=_ctypes.c_double) if x is None else x
    ipar, dpar, tmp = _create_empty_temp_arrays(ipar, dpar, tmp, 4 * n)
    tmp = tmp.reshape(4, n)

    ret_val = MKL.MKL_INT()
    
    _MKL_ISS._dcg_init(_ctypes.byref(MKL.MKL_INT(n)), x, b, _ctypes.byref(ret_val), ipar, dpar, tmp)

    status = ret_val.value

    if status == -10000:
        _msg = "dcg_init returned -10000 (Failed to complete task))"
        raise RuntimeError(_msg)
    elif status != 0 :
        _msg = "dcg_init returned {s} (Unknown error))".format(s=status)
        raise RuntimeError(_msg)

    return x, ipar, dpar, tmp


def dcg_check(n, b, x, ipar, dpar, tmp):
    """
    CG solver checker. 
    Verifies that the initialized parameters are valid.

    :param n: Number of predictor variables to model
    :type n: int
    :param b: Right-hand vector b
    :type b: np.ndarray
    :param x: Solution vector x
    :type x: np.ndarray
    :param ipar: Integer parameter vector with settings for solver
    :type ipar: np.ndarray
    :param dpar: Double parameter vector with settings for solver
    :type dpar: np.ndarray
    :param tmp: Temp vector for solver computations
    :type tmp: np.ndarray

    :return ret_val: Returned status of the check
    :rtype ret_val: int
    """

    ret_val = MKL.MKL_INT()

    _MKL_ISS._dcg_check(_ctypes.byref(MKL.MKL_INT(n)),
                        x,
                        b,
                        _ctypes.byref(ret_val),
                        ipar,
                        dpar,
                        tmp)

    status = ret_val.value

    if status == -1100:
        raise RuntimeError("dcg_check returned -1100 (operation failed)")
    elif status == -1001 or status == -1011:
        warnings.warn("dcg_check raised warnings")

    return status

def dcg(n, b, x, ipar, dpar, tmp):
    """
    Execute CG solver iteration

    :param n: Number of predictor variables to model
    :type n: int
    :param b: Right-hand vector b
    :type b: np.ndarray
    :param x: Solution vector x
    :type x: np.ndarray
    :param ipar: Integer parameter vector with settings for solver
    :type ipar: np.ndarray
    :param dpar: Double parameter vector with settings for solver
    :type dpar: np.ndarray
    :param tmp: Temp vector for solver computations
    :type tmp: np.ndarray

    :return ret_val: Returned status of the check
    :rtype ret_val: int
    """
    
    ret_val = MKL.MKL_INT()

    _MKL_ISS._dcg(_ctypes.byref(MKL.MKL_INT(n)), x, b, _ctypes.byref(ret_val), ipar, dpar, tmp)

    status = ret_val.value

    if status == -2:
        raise RuntimeError("dcg returned -2 (Divide by zero. This situation happens if the matrix is non-positive definite or almost non-positive definite)")
    elif status == -10:
        raise RuntimeError("dcg returned -10 (The residual norm is invalid. This usually happens because the value dpar(6) was altered outside of the routine, or the dcg_check routine was not called.)")
    elif status == -11:
        raise RuntimeError("dcg returned -11 (Infinite cycle. This usually happens because the values ipar(8), ipar(9), ipar(10) were altered outside of the routine, or the dcg_check routine was not called.)")
    elif status < 0:
        _msg = "dgc returned {s} (Unknown error)".format(s=status)
        raise RuntimeError(_msg)

    return status


class CGIterativeSparseSolver(MKLIterativeSparseSolver):

    def initialize_solver(self):
        self.x, self.ipar, self.dpar, self.tmp = dcg_init(self.n, self.b, self.x, self.ipar, self.dpar, self.tmp)

    def check_solver(self):
        return dcg_check(self.n, self.b, self.x, self.ipar, self.dpar, self.tmp)

    def solve_iteration(self):
        return dcg(self.n, self.b, self.x, self.ipar, self.dpar, self.tmp)

    def get_solution(self):
        return self.x

    def solve(self):
        """
        Run solver until complete (tolerances are reached) or until max_iter is reached
        Raises a warning if the solution did not converge

        :return: Solution vector x
        :rtype: np.ndarray
        """

        for return_value in self:

            if return_value == 1:
                pass
            elif return_value == 2:
                pass
            elif return_value == 3:
                pass
            elif return_value == 0:
                break

            if self.current_iter > self.max_iter:
                _msg = "Solution did not converge in {n} iterations".format(n=self.current_iter)
                warnings.warn(_msg, RuntimeWarning)
                break

        return self.x

    def set_initial_parameters(self):

        # Set max iterations in ipar[5]
        self.ipar[4] = self.max_iter

        # Suppress messages unless MKL debug is on or verbose is True
        self.ipar[5] = 1 if MKL.MKL_DEBUG or self.verbose else 0
        self.ipar[6] = 1 if MKL.MKL_DEBUG or self.verbose else 0

        # Set automatic stopping tests
        self.ipar[7] = 1
        self.ipar[8] = 1
        self.ipar[9] = 0

        # Set tolerances
        self.dpar[0] = self.r_tol
        self.dpar[1] = self.a_tol

    def update_tmp(self):
        _sparse_dense_vector_mult(self.A, self.tmp[0, :], out=self.tmp[1, :])
