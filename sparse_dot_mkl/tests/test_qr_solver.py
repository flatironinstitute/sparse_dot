import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as _spsparse
from sparse_dot_mkl import sparse_qr_solve_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1


class TestSparseSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.A = _spsparse.diags((MATRIX_1.data[0:100].copy()), format="csr")
        cls.B = MATRIX_1.data[0:100].copy().reshape(-1, 1)
        cls.X = np.linalg.lstsq(cls.A.A, cls.B, rcond=None)[0]

    def setUp(self):
        self.mat1 = self.A.copy()
        self.mat2 = self.B.copy()
        self.mat3 = self.X.copy()

    def test_sparse_solver(self):
        mat3 = sparse_qr_solve_mkl(self.mat1, self.mat2, debug=True)
        npt.assert_array_almost_equal(self.mat3, mat3)

    def test_sparse_solver_single(self):
        mat3 = sparse_qr_solve_mkl(self.mat1.astype(np.float32), self.mat2.astype(np.float32))
        npt.assert_array_almost_equal(self.mat3, mat3)

    def test_sparse_solver_cast_B(self):
        mat3 = sparse_qr_solve_mkl(self.mat1, self.mat2.astype(np.float32), cast=True)
        npt.assert_array_almost_equal(self.mat3, mat3)

    def test_sparse_solver_cast_A(self):
        mat3 = sparse_qr_solve_mkl(self.mat1.astype(np.float32), self.mat2, cast=True)
        npt.assert_array_almost_equal(self.mat3, mat3)

    def test_sparse_solver_cast_CSC(self):
        mat3 = sparse_qr_solve_mkl(self.mat1.tocsc(), self.mat2, cast=True)
        npt.assert_array_almost_equal(self.mat3, mat3)

    def test_sparse_solver_1d_d(self):
        mat3 = sparse_qr_solve_mkl(self.mat1, self.mat2.ravel())
        npt.assert_array_almost_equal(self.mat3.ravel(), mat3)

    def test_solver_guard_errors(self):
        with self.assertRaises(ValueError):
            mat3 = sparse_qr_solve_mkl(self.mat1, self.mat2.T)

        with self.assertRaises(ValueError):
            mat3 = sparse_qr_solve_mkl(self.mat1.tocsc(), self.mat2)

        with self.assertRaises(ValueError):
            mat3 = sparse_qr_solve_mkl(self.mat1.tocoo(), self.mat2, cast=True)
