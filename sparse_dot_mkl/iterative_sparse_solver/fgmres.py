from sparse_dot_mkl._mkl_interface import MKL
from sparse_dot_mkl._sparse_vector import _sparse_dense_vector_mult
from sparse_dot_mkl.iterative_sparse_solver._iss import _MKL_ISS, MKLIterativeSparseSolver, _create_empty_temp_arrays

import numpy as _np
import ctypes as _ctypes
import warnings

def dfgmres_init(n, b, x=None, ipar=None, dpar=None, tmp=None):
    """
    FGMRES solver initialization. 
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

    tmp_size = int(n * (2 * n + 1) + (n * (n + 9)) / 2 + 1)
    ipar, dpar, tmp = _create_empty_temp_arrays(ipar, dpar, tmp, tmp_size)

    ret_val = MKL.MKL_INT()
    
    _MKL_ISS._dfgmres_init(_ctypes.byref(MKL.MKL_INT(n)), x, b, _ctypes.byref(ret_val), ipar, dpar, tmp)

    status = ret_val.value

    if status == -10000:
        _msg = "dfgmres_init returned -10000 (Failed to complete task))"
        raise RuntimeError(_msg)
    elif status != 0 :
        _msg = "dfgmres_init returned {s} (Unknown error))".format(s=status)
        raise RuntimeError(_msg)

    return x, ipar, dpar, tmp

def dfgmres_check(n, b, x, ipar, dpar, tmp):
    """
    FGMRES solver checker. 
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

    _MKL_ISS._dfgmres_check(_ctypes.byref(MKL.MKL_INT(n)),
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

def dfgmres(n, b, x, ipar, dpar, tmp):
    """
    Execute FGMRES solver iteration

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

    _MKL_ISS._dfgmres(_ctypes.byref(MKL.MKL_INT(n)), x, b, _ctypes.byref(ret_val), ipar, dpar, tmp)

    status = ret_val.value

    if status == -1:
        # The routine was interrupted because the maximum number of iterations was reached, but the relative stopping criterion was not met.
        # This should be handled by the user
        pass 
    elif status == -10:
        raise RuntimeError("dfgmres returned -10 (The routine was interrupted because of an attempt to divide by zero. Usually this happens if the matrix is degenerate or almost degenerate.)")
    elif status == -11:
        raise RuntimeError("dfgmres returned -11 (Infinite cycle. This usually happens because the values ipar(7), ipar(8), ipar(9) were altered outside of the routine, or the dfgmres_check routine was not called.)")
    elif status == -12:
        raise RuntimeError("dfgmres returned -12 (The routine was interrupted because errors were found in the method parameters. Usually this happens if the parameters ipar and dpar were altered by mistake outside the routine.)")
    elif status < 0:
        _msg = "dgc returned {s} (Unknown error)".format(s=status)
        raise RuntimeError(_msg)

    return status

def dfgmres_get(n, b, x, ipar, dpar, tmp):
    """
    Get final results for FGMRES solve

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
    
    ret_val, iter_count = MKL.MKL_INT(), MKL.MKL_INT()

    _MKL_ISS._dfgmres_get(_ctypes.byref(MKL.MKL_INT(n)), x, b, _ctypes.byref(ret_val), ipar, dpar, tmp,
                          _ctypes.byref(iter_count))

    status = ret_val.value
    num_iteration = iter_count.value

    if status == -10000:
        _msg = "dfgmres_get returned -10000 (Failed to complete task))"
        raise RuntimeError(_msg)
    elif status == -12:
        raise RuntimeError("dfgmres_get returned -12 (The routine was interrupted because errors were found in the method parameters. Usually this happens if the parameters ipar and dpar were altered by mistake outside the routine.)")
    elif status != 0 :
        _msg = "dfgmres_get returned {s} (Unknown error))".format(s=status)
        raise RuntimeError(_msg)

    return status

class FGMRESIterativeSparseSolver(MKLIterativeSparseSolver):

    def initialize_solver(self):
        self.x, self.ipar, self.dpar, self.tmp = dfgmres_init(self.n, self.b, self.x, self.ipar, self.dpar, self.tmp)

    def check_solver(self):
        return dfgmres_check(self.n, self.b, self.x, self.ipar, self.dpar, self.tmp)

    def solve_iteration(self):
        return dfgmres(self.n, self.b, self.x, self.ipar, self.dpar, self.tmp)

    def get_solution(self):
        dfgmres_get(self.n, self.b, self.x, self.ipar, self.dpar, self.tmp)
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
        self.ipar[11] = 1

        # Set tolerances
        self.dpar[0] = self.r_tol
        self.dpar[1] = self.a_tol

    def update_tmp(self):

        input_matmul_offset = self.ipar[21]
        output_matmul_offset = self.ipar[22]

        _sparse_dense_vector_mult(self.A,
                                  self.tmp[input_matmul_offset - 1:input_matmul_offset + self.n - 2],
                                  out=self.tmp[output_matmul_offset - 1:output_matmul_offset + self.n - 2])
