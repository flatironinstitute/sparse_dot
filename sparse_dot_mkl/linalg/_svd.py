import numpy as np
import ctypes as _ctypes
import scipy.sparse as _sps
import warnings
import math

from sparse_dot_mkl._mkl_interface import _create_mkl_sparse, _destroy_mkl_handle, MKL, matrix_descr, _check_return_value
from sparse_dot_mkl.linalg._eigen import _MKL_Eigen

KS_MESSAGES = {
    -1: "maximum number of iterations has been reached and even the residual norm estimates have not converged",
    -2: "maximum number of iterations has been reached despite the residual norm estimates have converged (but the true residuals for eigenpairs have not)",
    -3: "the iterations stagnated and even the residual norm estimates have not converged",
    -4: "the iterations stagnated while the eigenvalues have converged (but the true residuals for eigenpairs do not)."
}


def _mkl_svd(A, k, whichS="L", whichV="L", solver=None, max_iter=None, E=None, ncv=None, tol=None):

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

    # If whichV is None, turn off calculating vectors
    if whichV is None:
        pm[6] = 0
        whichV = "L"

    # Change testing from normed to true residuals if looking for small vectors
    if whichS == "S":
        pm[7], pm[8] = 1, 1


    mkl_A, is_double = _create_mkl_sparse(A)
    mkl_A_desc = matrix_descr()

    # Convert char args to ctypes
    whichS, whichV = _ctypes.c_char(whichS.encode('utf-8')), _ctypes.c_char(whichV.encode('utf-8'))

    mkl_func = _MKL_Eigen._mkl_sparse_d_svd if is_double else _MKL_Eigen._mkl_sparse_s_svd
    output_dtype = np.float64 if is_double else np.float32

    # Allocate output arrays
    E = np.zeros((k, ), dtype=output_dtype) if E is None else E
    XL = np.zeros((A.shape[0], k), dtype=output_dtype, order="F")
    XR = np.zeros((k, A.shape[1]), dtype=output_dtype)
    Res = np.zeros((k, ), dtype=output_dtype)
    k_found = np.zeros((1, ), dtype=MKL.MKL_INT_NUMPY)

    return_status = mkl_func(_ctypes.byref(whichS),
                             _ctypes.byref(whichV),
                             pm,
                             mkl_A,
                             mkl_A_desc,
                             k,
                             k_found,
                             E,
                             XL,
                             XR,
                             Res)

    _check_return_value(return_status, mkl_func.__name__)

    _destroy_mkl_handle(mkl_A)

    if pm[9] != 0:
        _msg = "Krylov-Schur returned {v}: {mg}".format(v=pm[9], mg=KS_MESSAGES[pm[9]])
        warnings.warn(_msg, RuntimeWarning)

    return E, XL, XR, Res, k_found[0]


def svds(A, k=6, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors="u", solver=None):
    """
    Compute the largest or smallest k singular values/vectors for a sparse matrix

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
    """

    # Switch for which flag
    if which == "LM":
        whichS = "L"
    elif which == "SM":
        whichS = "S"
    else:
        _msg = "which argument must be LM or SM; {v} passed".format(v=which)
        raise ValueError(_msg)

    if return_singular_vectors == "u":
        whichV = "L"
    elif return_singular_vectors == "vh":
        whichV = "R"
    elif return_singular_vectors is True:
        whichV = "L"
        warnings.warn("svds does not support returning u and vh; only u will be returned", RuntimeWarning)
    elif return_singular_vectors is False:
        whichV = None
    else:
        _msg = "return_singular_vectors argument must be u or vh; {v} passed".format(v=return_singular_vectors)
        raise ValueError(_msg)

    if tol == 0:
        tol = 10
    else:
        tol = abs(round(math.frexp(tol)[1] / math.log(10, 2)))

    E, XL, XR, Res, k_found = _mkl_svd(A, k, whichS=whichS, whichV = whichV, ncv=ncv, max_iter=maxiter, E=v0, tol=tol)

    # Matching undocumented scipy behavior
    if whichV is None and not return_singular_vectors:
        return E

    # Matching documented scipy behavior
    elif whichV == "L":
        return XL, E, None
    elif whichV == "R":
        return None, E, XR
