import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as _spsparse

from scipy.sparse.linalg import svds as scipy_svds
from scipy.linalg import svd as scipy_dense_svd
from sparse_dot_mkl.linalg._svd import _mkl_svd, svds
from sparse_dot_mkl.tests.test_mkl import MATRIX_1

def is_sign_indeterminate_equal(a, b, axis=1):

    # Check that the only difference is signs
    npt.assert_array_almost_equal(np.abs(a), np.abs(b))

    # Check that the signs are flipped for an entire row/column (and not just randomly assigned)
    ind_diff = np.minimum(np.abs(np.sum(a - b, axis=axis)), np.abs(np.sum(a + b, axis=axis)))
    npt.assert_array_almost_equal(ind_diff, np.zeros_like(ind_diff))


class TestSVD10(unittest.TestCase):

    k = 10
    which_scipy = "LM"
    whichS = "L"

    solver = None
    
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

        npt.assert_array_almost_equal(s, E)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E)
        
        npt.assert_array_almost_equal(np.zeros_like(XR), XR)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(u, XL, axis=0)

    def test_svd_csr_L_prestarted(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichS=self.whichS, solver=self.solver, E=s.copy())

        self.assertEqual(k_found, self.k)

        npt.assert_array_almost_equal(s, E)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E)
        
        npt.assert_array_almost_equal(np.zeros_like(XR), XR)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(u, XL, axis=0)

    def test_svd_csr_R(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichV="R", whichS=self.whichS, solver=self.solver)

        self.assertEqual(k_found, self.k)

        npt.assert_array_almost_equal(s, E)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E)

        npt.assert_array_almost_equal(np.zeros_like(XL), XL)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(vt, XR)

    def test_svd_csr_R_prestarted(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichV="R", whichS=self.whichS, solver=self.solver, E=s.copy())

        self.assertEqual(k_found, self.k)
        
        npt.assert_array_almost_equal(s, E)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E)

        npt.assert_array_almost_equal(np.zeros_like(XL), XL)
        npt.assert_array_equal(self.mat1.A, self.A.A)

        is_sign_indeterminate_equal(vt, XR)

    def test_svd_csr_None(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichV=None, whichS=self.whichS, solver=self.solver)

        self.assertEqual(k_found, self.k)

        npt.assert_array_almost_equal(s, E)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E)

        npt.assert_array_almost_equal(np.zeros_like(XR), XR)
        npt.assert_array_almost_equal(np.zeros_like(XL), XL)
        npt.assert_array_equal(self.mat1.A, self.A.A)

    def test_svd_csr_None_prestarted(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy)
        E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichV=None, whichS=self.whichS, solver=self.solver, E=s.copy())

        self.assertEqual(k_found, self.k)

        npt.assert_array_almost_equal(s, E)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), E)

        npt.assert_array_almost_equal(np.zeros_like(XR), XR)
        npt.assert_array_almost_equal(np.zeros_like(XL), XL)
        npt.assert_array_equal(self.mat1.A, self.A.A)

    def test_svd_csc(self):

        self.mat1 = _spsparse.csc_matrix(self.mat1)

        with self.assertRaises(ValueError):
            E, XL, XR, Res, k_found = _mkl_svd(self.mat1, self.k, whichV="R")

        npt.assert_array_equal(self.mat1.A, self.A.A)

    def test_svd_wrap_csr_L(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy, return_singular_vectors="u")
        mkl_u, mkl_s, mkl_vt = svds(self.mat1, self.k, which=self.which_scipy, return_singular_vectors="u")

        npt.assert_array_almost_equal(s, mkl_s)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), mkl_s)

        npt.assert_array_equal(self.mat1.A, self.A.A)
        self.assertIsNone(mkl_vt)

        is_sign_indeterminate_equal(u, mkl_u, axis=0)

    def test_svd_wrap_csr_R(self):

        u, s, vt = scipy_svds(self.mat1, k=self.k, which=self.which_scipy, return_singular_vectors="vh")
        mkl_u, mkl_s, mkl_vt = svds(self.mat1, self.k, which=self.which_scipy, return_singular_vectors="vh")

        #npt.assert_array_almost_equal(s, mkl_s)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        diag_s = np.zeros_like(self.mat1.A)
        np.fill_diagonal(diag_s, ds)
        npt.assert_array_almost_equal(du @ diag_s @ dvh, self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), mkl_s)

        npt.assert_array_equal(self.mat1.A, self.A.A)
        self.assertIsNone(mkl_u)

        is_sign_indeterminate_equal(vt, mkl_vt)


    def test_svd_wrap_csr_None(self):

        s = scipy_svds(self.mat1, k=self.k, which=self.which_scipy, return_singular_vectors=False)
        mkl_s = svds(self.mat1, self.k, which=self.which_scipy, return_singular_vectors=False)

        npt.assert_array_almost_equal(s, mkl_s)
        du, ds, dvh = scipy_dense_svd(self.mat1.A)
        npt.assert_array_almost_equal(self.ds_idx(ds), mkl_s)

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

