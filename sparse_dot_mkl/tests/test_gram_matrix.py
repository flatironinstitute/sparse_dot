import unittest
import numpy as np
import numpy.testing as npt
from sparse_dot_mkl import gram_matrix_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1


class TestGramMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        gram_ut = np.dot(MATRIX_1.A.T, MATRIX_1.A)
        gram_ut[np.tril_indices(gram_ut.shape[0], k=-1)] = 0.
        cls.gram_ut = gram_ut

        gram_ut_t = np.dot(MATRIX_1.A, MATRIX_1.A.T)
        gram_ut_t[np.tril_indices(gram_ut_t.shape[0], k=-1)] = 0.
        cls.gram_ut_t = gram_ut_t

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat1_d = MATRIX_1.A

    def test_gram_matrix_sp(self):
        mat2 = gram_matrix_mkl(self.mat1)
        npt.assert_array_almost_equal(mat2.A, self.gram_ut)

        with self.assertRaises(ValueError):
            gram_matrix_mkl(self.mat1, out=np.zeros((self.mat1.shape[0], self.mat1.shape[0])))

    def test_gram_matrix_sp_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(np.float32))
        npt.assert_array_almost_equal(mat2.A, self.gram_ut)

    def test_gram_matrix_d_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(np.float32), dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(self.mat1.astype(np.float32), dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=np.float32),  out_scalar=1.)
        mat2[np.tril_indices(mat2.shape[0], k=-1)] = 0.
        npt.assert_array_almost_equal(mat2, self.gram_ut)

        with self.assertRaises(ValueError):
            mat2 = gram_matrix_mkl(self.mat1.astype(np.float32), dense=True,
                                   out=np.zeros((self.mat1.shape[1], self.mat1.shape[1])),
                                   out_scalar=1.)

    def test_gram_matrix_d(self):
        mat2 = gram_matrix_mkl(self.mat1, dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(self.mat1, dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=np.float64),  out_scalar=1.)
        mat2[np.tril_indices(mat2.shape[0], k=-1)] = 0.
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_sp_t(self):
        mat2 = gram_matrix_mkl(self.mat1, transpose=True)
        npt.assert_array_almost_equal(mat2.A, self.gram_ut_t)

    def test_gram_matrix_d_t(self):
        mat2 = gram_matrix_mkl(self.mat1, dense=True, transpose=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut_t)

    def test_gram_matrix_csc_sp(self):
        mat2 = gram_matrix_mkl(self.mat1.tocsc(), cast=True)
        npt.assert_array_almost_equal(mat2.A, self.gram_ut)

    def test_gram_matrix_csc_d(self):
        mat2 = gram_matrix_mkl(self.mat1.tocsc(), dense=True, cast=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_double(self):
        mat2 = gram_matrix_mkl(self.mat1.A, dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(self.mat1.A, dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=np.float64),  out_scalar=1.)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(np.float32).A, dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(self.mat1.astype(np.float32).A, dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=np.float32),  out_scalar=1.)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_double_F(self):
        mat2 = gram_matrix_mkl(np.asarray(self.mat1.A, order="F"), dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(np.asarray(self.mat1.A, order="F"), dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=np.float64, order="F"),
                               out_scalar=1.)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_single_F(self):
        mat2 = gram_matrix_mkl(np.asarray(self.mat1.astype(np.float32).A, order="F"), dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(np.asarray(self.mat1.astype(np.float32).A, order="F"), dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=np.float32, order="F"),
                               out_scalar=1.)
        npt.assert_array_almost_equal(mat2, self.gram_ut)
