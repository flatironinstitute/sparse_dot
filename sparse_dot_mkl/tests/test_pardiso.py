import unittest
import numpy as np
import numpy.testing as npt
from sparse_dot_mkl.solvers import pardiso, pardisoinit
from sparse_dot_mkl.tests.test_mkl import make_matrixes
from sparse_dot_mkl import sparse_qr_solve_mkl

A, B = make_matrixes(50, 10, 50, 0.2)
B = B.toarray()
A.sort_indices()


class TestPARDISO(unittest.TestCase):

    dtype = np.float32
    mtype = 11
    single_precision = True

    def setUp(self):

        self.pt, self.iparm = pardisoinit(
            self.mtype,
            single_precision=self.single_precision
        )

    def test_pardiso_init(self):

        npt.assert_equal(self.pt, np.zeros_like(self.pt))

        _iparm_init = np.zeros_like(self.iparm)
        _iparm_init[0] = 1
        _iparm_init[1] = 2
        _iparm_init[9] = 13
        _iparm_init[[10, 12, 34]] = 1
        _iparm_init[[17, 18]] = -1

        if self.single_precision:
            _iparm_init[27] = 1

        npt.assert_equal(self.iparm, _iparm_init)

    def test_pardiso_analysis(self):

        X, pt, perm, error = pardiso(
            A.astype(self.dtype),
            B[:, 0].astype(self.dtype),
            self.pt,
            self.mtype,
            self.iparm,
            11
        )

        self.assertEqual(error, 0)
        npt.assert_array_almost_equal(
            X,
            np.zeros_like(X)
        )
        npt.assert_array_almost_equal(
            perm,
            np.zeros_like(perm)
        )
        with self.assertRaises(AssertionError):
            npt.assert_equal(pt, np.zeros_like(pt))

    def test_pardiso_solve(self):

        X, pt, perm, error = pardiso(
            A.astype(self.dtype),
            B[:, 0].astype(self.dtype),
            self.pt,
            self.mtype,
            self.iparm,
            13
        )

        self.assertEqual(error, 0)
        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(
                X,
                np.zeros_like(X)
            )

        with self.assertRaises(AssertionError):
            npt.assert_equal(pt, np.zeros_like(pt))

        # Test the real solver
        if self.mtype == 11:
            QR_X = sparse_qr_solve_mkl(
                A.astype(self.dtype),
                B[:, 0].astype(self.dtype)
            )
        
        # Test the complex solver; because the img components are
        # all zero, it's the same result as the real solver
        else:
            _real_B = np.ascontiguousarray(
                B[:, 0].astype(self.dtype).real
            )
            QR_X = np.zeros_like(X)
            QR_X.real = sparse_qr_solve_mkl(
                A.astype(_real_B.dtype),
                _real_B
            )

        npt.assert_array_almost_equal(
            X,
            QR_X,
            decimal=3
        )

    def test_pardiso_solve_mrhs(self):

        X, pt, perm, error = pardiso(
            A.astype(self.dtype),
            B.astype(self.dtype),
            self.pt,
            self.mtype,
            self.iparm,
            13
        )

        self.assertEqual(error, 0)
        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(
                X,
                np.zeros_like(X)
            )

        with self.assertRaises(AssertionError):
            npt.assert_equal(pt, np.zeros_like(pt))


class TestPARDISODouble(TestPARDISO):

    dtype = np.float64
    mtype = 11
    single_precision = False


class TestPARDISOSingleComplex(TestPARDISO):

    dtype = np.complex64
    mtype = 13
    single_precision = True

class TestPARDISODoubleComplex(TestPARDISO):

    dtype = np.complex128
    mtype = 13
    single_precision = False
