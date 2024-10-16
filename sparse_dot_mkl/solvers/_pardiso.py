import numpy as np
import ctypes as _ctypes
import scipy.sparse as sps

import warnings

from sparse_dot_mkl._mkl_interface._cfunctions import (
    MKL
)
from sparse_dot_mkl._mkl_interface._common import (
    is_csr
)

PARDISO_ERRORS = {
    0: None,
    -1: "input inconsistent",
    -2: "not enough memory",
    -3: "reordering problem",
    -4: "Zero pivot, numerical factorization or iterative refinement problem",
    -5: "unclassified (internal) error",
    -6: "reordering failed (matrix types 11 and 13 only)",
    -7: "diagonal matrix is singular",
    -8: "32-bit integer overflow problem",
    -9: "not enough memory for OOC",
    -10: "error opening OOC files",
    -11: "read/write error with OOC files",
    -12: "(pardiso_64 only) pardiso_64 called from 32-bit library",
    -13: "interrupted by the (user-defined) mkl_progress function",
    -15: "internal error which can appear for iparm[23]=10 and iparm[12]=1"
}

def pardiso(
    A,
    B,
    pt,
    mtype,
    iparm,
    phase=13,
    maxfct=1,
    mnum=1,
    perm=None,
    msglvl=0,
    X=None,
    quiet=False
):
    """
    Run pardiso solver for AX = B

    :param A: Matrix A in CSR format
    :type A: sp.sparse.csr_array, sp.sparse.csr_matrix
    :param B: Matrix B in dense format
    :type B: np.ndarray
    :param pt: Pointer array, shape=(64,) dtype=int64
    :type pt: np.ndarray
    :param mtype: Matrix type:
        1   Real and structurally symmetric
        2	Real and symmetric positive definite
        -2  Real and symmetric indefinite
        3   Complex and structurally symmetric
        4	Complex and Hermitian positive definite
        -4	Complex and Hermitian indefinite
        6   Complex and symmetric matrix
        11	Real and nonsymmetric matrix
        13	Complex and nonsymmetric matrix
    :type mtype: int
    :param iparm: Solver parameters array, shape=(64,)
    :type iparm: np.ndarray
    :param phase: Solver phase, defaults to 13
        (Analysis, numerical factorization, solve, iterative refinement)
    :type phase: int
    :param maxfct: Pardiso maxfct, defaults to 1
    :type maxfct: int, optional
    :param mnum: Pardiso mnum, defaults to 1
    :type mnum: int, optional
    :param perm: Permutation vector array, new allocation if None,
        defaults to None
    :type perm: np.ndarray, optional
    :param msglvl: Pardiso message level, defaults to 0
    :type msglvl: int, optional
    :param X: Solved array X,  new allocation if None,
        defaults to None
    :type X: np.ndarray, optional
    :param quiet: Don't issue runtime warnings if pardiso
        returnvalue != 0, defaults to False
    :type quiet: bool, optional
    :return:
        Solved array X,
        Pointer array pt,
        Permutation array perm,
        Return value error
    :rtype: np.ndarray, np.ndarray, np.ndarray, int
    """
    
    if not is_csr(A):
        raise ValueError(
            f'A must be a CSR matrix; {type(A)} passed'
        )
    
    if sps.issparse(B):
        raise ValueError(
            f'B must be a dense array; {type(B)} passed'
        )
    
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"Bad matrix shapes for AX=B solver: "
            f"A {A.shape} & B {B.shape}"
        )
    else:
        N = A.shape[0]
    
    if perm is None:
        perm = np.zeros(N, dtype=MKL.MKL_INT_NUMPY)

    if B.ndim == 1:
        nrhs = 1
    elif B.ndim > 2:
        raise ValueError('B must be 1- or 2-d')
    else:
        nrhs = B.shape[1]

    if X is None:
        X = np.zeros_like(B)

    error = MKL.MKL_INT(0)
    
    MKL._pardiso(
        pt,
        _ctypes.byref(MKL.MKL_INT(maxfct)),
        _ctypes.byref(MKL.MKL_INT(mnum)),
        _ctypes.byref(MKL.MKL_INT(mtype)),
        _ctypes.byref(MKL.MKL_INT(phase)),
        _ctypes.byref(MKL.MKL_INT(N)),
        A.data,
        A.indptr.astype(MKL.MKL_INT_NUMPY),
        A.indices.astype(MKL.MKL_INT_NUMPY),
        perm,
        _ctypes.byref(MKL.MKL_INT(nrhs)),
        iparm,
        _ctypes.byref(MKL.MKL_INT(msglvl)),
        B,
        X,
        _ctypes.byref(error)
    )

    error = error.value

    if error != 0 and not quiet:
        warnings.warn(
            f"MKL pardiso error {error}: " +
            PARDISO_ERRORS[error],
            RuntimeWarning
        )

    return X, pt, perm, error


def pardisoinit(
    mtype,
    pt=None,
    iparm=None,
    single_precision=None,
    zero_indexing=True
):
    """
    Run pardisoinit to initialize pt and iparm for
    a given matrix type

    :param mtype: Matrix type:
        1   Real and structurally symmetric
        2	Real and symmetric positive definite
        -2  Real and symmetric indefinite
        3   Complex and structurally symmetric
        4	Complex and Hermitian positive definite
        -4	Complex and Hermitian indefinite
        6   Complex and symmetric matrix
        11	Real and nonsymmetric matrix
        13	Complex and nonsymmetric matrix
    :type mtype: int
    :param pt: Pointer array (int64), new allocation if None,
        defaults to None
    :type pt: np.ndarray, optional
    :param iparm: Solver parameters array, new allocation if None,
        defaults to None
    :type iparm: np.ndarray, optional
    :param single_precision: Set iparm flag for single precision if True,
        set flag for double precision if False, do not change flag value in
        iparm if None, defaults to None
    :type single_precision: bool, optional
    :param zero_indexing: Set iparm flag for zero indexing (C & python)
        if True, set flag for one indexing (F) if False, do not change flag
        value in iparm if None, defaults to True
    :type zero_indexing: bool, optional
    :return: pt (pointer) and iparm (parameter) arrays for pardiso
    :rtype: np.ndarray, np.ndarray
    """

    if pt is None:
        pt = np.empty(64, np.int64)

    if iparm is None:
        iparm = np.zeros(64, dtype=MKL.MKL_INT_NUMPY)

    MKL._pardisoinit(
        pt,
        _ctypes.byref(MKL.MKL_INT(mtype)),
        iparm
    )

    # Set zero indexing flag in iparm[34]
    if zero_indexing is None:
        pass
    elif zero_indexing:
        iparm[34] = 1
    else:
        iparm[34] = 0

    if single_precision is None:
        pass
    elif single_precision:
        iparm[27] = 1
    else:
        iparm[27] = 0

    return pt, iparm
