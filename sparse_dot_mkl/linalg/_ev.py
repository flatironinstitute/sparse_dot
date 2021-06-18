import numpy as np
import ctypes as _ctypes
import scipy.sparse as _sps
import warnings
import math

from sparse_dot_mkl._mkl_interface import _create_mkl_sparse, _destroy_mkl_handle, MKL, matrix_descr, _check_return_value, _order_mkl_handle
from sparse_dot_mkl.linalg._eigen import _MKL_Eigen
from sparse_dot_mkl.linalg._svd import KS_MESSAGES


def _mkl_ev(A, k, which="L", solver=None, max_iter=None, E=None, ncv=None, tol=None):

    if not (_sps.isspmatrix_csr(A) or _sps.isspmatrix_bsr(A)):
        _msg = "A must be CSR or BSR format; {t} provided".format(t=type(A))
        raise ValueError(_msg)

    # Initialize parameters
    pm = np.zeros((128, ), dtype=MKL.MKL_INT_NUMPY)
    _MKL_Eigen._mkl_sparse_ee_init(pm)

    if tol is not None:
        pm[1] = int(tol)
    
    # Set solver algorithm
    if solver == "KS":
        pm[2] = 1

    elif solver == "FEAST":
        pm[2] = 2

    elif solver is None:
        pm[2] = 0

    else:
        raise ValueError("Solver must be KS, FEAST, or None")

    if ncv is not None:
        pm[3] = int(ncv)

    if max_iter is not None:
        pm[4] = int(ncv)

    mkl_A, is_double = _create_mkl_sparse(A)
    mkl_A_desc = matrix_descr()
   
    # Convert char args to ctypes
    which = _ctypes.c_char(which.encode('utf-8'))

    mkl_func = _MKL_Eigen._mkl_sparse_d_ev if is_double else _MKL_Eigen._mkl_sparse_s_ev
    output_dtype = np.float64 if is_double else np.float32

    # Allocate output arrays
    E = np.zeros((k, ), dtype=output_dtype) if E is None else E
    X = np.zeros((A.shape[1], k), dtype=output_dtype, order="F")
    Res = np.zeros((k, ), dtype=output_dtype)

    k_found = MKL.MKL_INT(0)

    return_status = mkl_func(_ctypes.byref(which),
                             pm,
                             mkl_A,
                             mkl_A_desc,
                             k,
                             _ctypes.byref(k_found),
                             E,
                             X,
                             Res)

    _check_return_value(return_status, mkl_func.__name__)

    _destroy_mkl_handle(mkl_A)

    if pm[9] != 0:
        _msg = "Krylov-Schur returned {v}: {mg}".format(v=pm[9], mg=KS_MESSAGES[pm[9]])
        warnings.warn(_msg, RuntimeWarning)

    return E, X, Res, k_found.value


def eigs(A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
         Minv=None, OPinv=None, OPpart=None):
    """
    Compute the largest or smallest k eigenvalues & eigenvectors

    :param A: Sparse matrix in CSR or BSR format
    :type A: scipy.sprse.spmatrix
    :param k: Number of singular values and vectors to compute, defaults to 6
    :type k: int, optional
    :param ncv: The number of Lanczos vectors generated, defaults to None. Only used with Krylov-Schur Method.
    :type ncv: int, optional
    :param tol: Tolerance for singular values, defaults to 0
    :type tol: int, optional
    :param which: Which k singular values to find, defaults to 'LM': 
        ‘LM’ : largest singular values,
        ‘SM’ : smallest singular values
    :type which: str, optional
    :param v0: Starting vector for iteration, defaults to None
    :type v0: np.ndarray, optional
    :param maxiter: Maximum number of iterations, only used for Krylov-Schur, defaults to None
    :type maxiter: int, optional
    :param return_singular_vectors: Return singular vectors in addition to singular values, defaults to "u"
        “u”: only return the u matrix,
        “vh”: only return the vh matrix
    :type return_singular_vectors: bool, str, optional
    :param solver: Unused, defaults to None.
    :type solver: str, optional
    :param sigma: Unused
    :type sigma: float, optional
    :param Minv: Unused
    :type Minv: ndarray, sparse matrix or LinearOperator, optional
    :param OPinv: Unused
    :type OPinv: ndarray, sparse matrix or LinearOperator, optional
    :param OPpart: Unused
    :type OPpart: str, optional

    """

    # Switch for which flag
    if which == "LM":
        which = "L"
    elif which == "SM":
        which = "S"
    else:
        _msg = "which argument must be LM or SM; {v} passed".format(v=which)
        raise ValueError(_msg)

    if tol == 0:
        tol = 10
    else:
        tol = abs(round(math.frexp(tol)[1] / math.log(10, 2)))

    E, X, Res, k_found = _mkl_ev(A, k, which=which, ncv=ncv, max_iter=maxiter, E=v0, tol=tol)

    if return_eigenvectors:
        return E, X
    else:
        return E
