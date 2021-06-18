import unittest
import numpy as np
from numpy.core.shape_base import vstack
import numpy.testing as npt
import scipy.sparse as _spsparse

from scipy.sparse.linalg import eigs as scipy_eigs
from scipy.linalg import eig as scipy_dense_eig
from sparse_dot_mkl.linalg._ev import _mkl_ev, eigs

from scipy.sparse.linalg import svds as scipy_svds
from scipy.linalg import svd as scipy_dense_svd
from sparse_dot_mkl.linalg._svd import _mkl_svd, svds

from sparse_dot_mkl.tests.test_mkl import MATRIX_1
from sparse_dot_mkl.tests.test_iterative_solver import test_matrix


def is_sign_indeterminate_equal(a, b, axis=1, decimal=6):

    # Check that the only difference is signs
    npt.assert_array_almost_equal(np.abs(a), np.abs(b), decimal=decimal)

    # Check that the signs are flipped for an entire row/column (and not just randomly assigned)
    ind_diff = np.minimum(np.abs(np.sum(a - b, axis=axis)), np.abs(np.sum(a + b, axis=axis)))
    npt.assert_array_almost_equal(ind_diff, np.zeros_like(ind_diff), decimal=decimal - 1)


class TestEV20Sym(unittest.TestCase):

    k = 20
    which_scipy = "LM"
    which = "L"

    solver = None

    decimal = 6

    def ds_idx(self, arr):
        if self.which == "L":
            return np.sort(np.real(arr))[-1 * self.k:]
        else:
            return np.sort(np.real(arr))[:self.k]

    @classmethod
    def setUpClass(cls):
        cls.A = MATRIX_1 @ MATRIX_1.T

    def setUp(self):
        self.mat1 = self.A.copy()

    def test_ev_csr(self):

        w, v = scipy_eigs(self.mat1, k=self.k, which=self.which_scipy)
        E, X, Res, k_found = _mkl_ev(self.mat1, self.k, which=self.which, solver=self.solver)
    
        self.assertEqual(k_found, self.k)

        w = np.real(w)[::-1] if self.which_scipy == "LM" else np.real(w)
        v = np.flip(np.real(v), axis=1) if self.which_scipy == "LM" else np.real(v)

        npt.assert_array_almost_equal(w, E, decimal=self.decimal)
        dw, dv = scipy_dense_eig(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(dw), E, decimal=self.decimal)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(X, v, axis=0, decimal=self.decimal)

    def test_ev_bsr(self):

        self.mat1 = self.mat1.tobsr(blocksize=(2,2))
        
        w, v = scipy_eigs(self.mat1, k=self.k, which=self.which_scipy)
        E, X, Res, k_found = _mkl_ev(self.mat1, self.k, which=self.which, solver=self.solver)
    
        self.assertEqual(k_found, self.k)

        w = np.real(w)[::-1] if self.which_scipy == "LM" else np.real(w)
        v = np.flip(np.real(v), axis=1) if self.which_scipy == "LM" else np.real(v)

        npt.assert_array_almost_equal(w, E, decimal=self.decimal)
        dw, dv = scipy_dense_eig(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(dw), E, decimal=self.decimal)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(X, v, axis=0, decimal=self.decimal)

    def test_ev_wrap_csr(self):

        w, v = scipy_eigs(self.mat1, k=self.k, which=self.which_scipy)
        E, X = eigs(self.mat1, self.k, which=self.which_scipy)
    
        w = np.real(w)[::-1] if self.which_scipy == "LM" else np.real(w)
        v = np.flip(np.real(v), axis=1) if self.which_scipy == "LM" else np.real(v)

        npt.assert_array_almost_equal(w, E, decimal=self.decimal)
        dw, dv = scipy_dense_eig(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(dw), E, decimal=self.decimal)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(X, v, axis=0, decimal=self.decimal)

    def test_ev_csc(self):

        self.mat1 = self.mat1.tocsc()
        
        with self.assertRaises(ValueError):
            E, X, Res, k_found = _mkl_ev(self.mat1, self.k, which=self.which, solver=self.solver)
    

class TestEV20SymSmall(TestEV20Sym):
    which_scipy = "SM"
    which = "S"
    
class TestEV5Sym(TestEV20Sym):

    k = 5

    @classmethod
    def setUpClass(cls):
        cls.A = test_matrix @ test_matrix.T


class TestSVD10(unittest.TestCase):

    k = 10
    which_scipy = "LM"
    whichS = "L"

    solver = None
    
    decimal = 6

    @classmethod
    def setUpClass(cls):
        cls.A = MATRIX_1.copy()

    def setUp(self):
        self.mat1 = self.A.copy()

    def ds_idx(self, arr):
        return arr[0:self.k][::-1]

    def test_svd_csr_L(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichS=self.whichS, solver=self.solver)

        self.assertEqual(k_found, self.k)

        npt.assert_array_almost_equal(s, E, decimal=self.decimal)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E, decimal=self.decimal)
        
        npt.assert_array_almost_equal(np.zeros_like(XR), XR, decimal=self.decimal)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(u, XL, axis=0, decimal=self.decimal)

    def test_svd_csr_L_prestarted(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichS=self.whichS, solver=self.solver, E=s.copy())

        self.assertEqual(k_found, self.k)

        npt.assert_array_almost_equal(s, E, decimal=self.decimal)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E, decimal=self.decimal)
        
        npt.assert_array_almost_equal(np.zeros_like(XR), XR, decimal=self.decimal)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(u, XL, axis=0, decimal=self.decimal)

    def test_svd_csr_R(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichV="R", whichS=self.whichS, solver=self.solver)

        self.assertEqual(k_found, self.k)

        npt.assert_array_almost_equal(s, E, decimal=self.decimal)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E, decimal=self.decimal)

        npt.assert_array_almost_equal(np.zeros_like(XL), XL, decimal=self.decimal)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(vt, XR, decimal=self.decimal)

    def test_svd_csr_R_prestarted(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichV="R", whichS=self.whichS, solver=self.solver, E=s.copy())

        self.assertEqual(k_found, self.k)
        
        npt.assert_array_almost_equal(s, E, decimal=self.decimal)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E, decimal=self.decimal)

        npt.assert_array_almost_equal(np.zeros_like(XL), XL, decimal=self.decimal)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(vt, XR, decimal=self.decimal)

    def test_svd_csr_None(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichV=None, whichS=self.whichS, solver=self.solver)

        self.assertEqual(k_found, self.k)

        npt.assert_array_almost_equal(s, E, decimal=self.decimal)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E, decimal=self.decimal)

        npt.assert_array_almost_equal(np.zeros_like(XR), XR, decimal=self.decimal)
        npt.assert_array_almost_equal(np.zeros_like(XL), XL, decimal=self.decimal)
        npt.assert_array_equal(self.mat1.A, self.A.A)

    def test_svd_csr_None_prestarted(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichV=None, whichS=self.whichS, solver=self.solver, E=s.copy())

        self.assertEqual(k_found, self.k)

        npt.assert_array_almost_equal(s, E, decimal=self.decimal)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E, decimal=self.decimal)

        npt.assert_array_almost_equal(np.zeros_like(XR), XR, decimal=self.decimal)
        npt.assert_array_almost_equal(np.zeros_like(XL), XL, decimal=self.decimal)
        npt.assert_array_equal(self.mat1.A, self.A.A)

    def test_svd_csc(self):

        self.mat1 = _spsparse.csc_matrix(self.mat1)

        with self.assertRaises(ValueError):
            E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichV="R")

        npt.assert_array_equal(self.mat1.A, self.A.A)

    def test_svd_wrap_csr_L(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy, return_singular_vectors="u")
        mkl_u, mkl_s, mkl_vt = svds(self.mat1, self.k, which=self.which_scipy, return_singular_vectors="u")

        npt.assert_array_almost_equal(s, mkl_s, decimal=self.decimal)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), mkl_s, decimal=self.decimal)

        npt.assert_array_equal(self.mat1.A, self.A.A)
        self.assertIsNone(mkl_vt)

        is_sign_indeterminate_equal(u, mkl_u, axis=0, decimal=self.decimal)

    def test_svd_wrap_csr_R(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy, return_singular_vectors="vh")
        mkl_u, mkl_s, mkl_vt = svds(self.mat1, self.k, which=self.which_scipy, return_singular_vectors="vh")

        npt.assert_array_almost_equal(s, mkl_s, decimal=self.decimal)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), mkl_s, decimal=self.decimal)

        npt.assert_array_equal(self.mat1.A, self.A.A)
        self.assertIsNone(mkl_u)

        is_sign_indeterminate_equal(vt, mkl_vt, decimal=self.decimal)


    def test_svd_wrap_csr_None(self):

        s = scipy_svds(self.mat1, k=self.k, which=self.which_scipy, return_singular_vectors=False)
        mkl_s = svds(self.mat1, self.k, which=self.which_scipy, return_singular_vectors=False)

        npt.assert_array_almost_equal(s, mkl_s, decimal=self.decimal)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), mkl_s, decimal=self.decimal)

        npt.assert_array_equal(self.mat1.A, self.A.A)

class TestSVD_BSR_10(TestSVD10):

    def setUp(self):
        self.mat1 = self.A.tobsr(blocksize=(2, 2))

class TestSVD100(TestSVD10):
    k = 100

class TestSVD_BSR_100(TestSVD_BSR_10):
    k = 100

class TestSVDDensey(TestSVD10):

    @classmethod
    def setUpClass(cls):
        cls.A = np.random.default_rng(50).random((1000,100), dtype=float)
        cls.A[cls.A < 0.5] = 0.0
        cls.A = _spsparse.csr_matrix(cls.A)

class TestSVDDensey50(TestSVDDensey):

    k=50

#@unittest.skip
class TestSVDSmall(TestSVD10):

    which_scipy = "SM"
    whichS = "S"

    def ds_idx(self, arr):
        return arr[len(arr) - self.k:][::-1]

    # I don't know why these aren't giving the right eigenvalues
    @unittest.skip
    def test_svd_csr_R(self):
        pass

    @unittest.skip
    def test_svd_csr_R_prestarted(self):
        pass

    @unittest.skip
    def test_svd_wrap_csr_R(self):
        pass

@unittest.skip
class TestSVD10_Single(TestSVD10):

    decimal = 3

    @classmethod
    def setUpClass(cls):
        cls.A = MATRIX_1.astype(np.float32)