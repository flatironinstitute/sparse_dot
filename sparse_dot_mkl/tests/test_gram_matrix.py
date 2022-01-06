import unittest
import numpy as np
import numpy.testing as npt
from sparse_dot_mkl import gram_matrix_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1, make_matrixes


class TestGramMatrix(unittest.TestCase):

    double_dtype = np.float64
    single_dtype = np.float32

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1 = MATRIX_1.copy()

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()
        self.mat1_d = self.MATRIX_1.A

        gram_ut = np.dot(self.MATRIX_1.A.T, self.MATRIX_1.A)
        gram_ut[np.tril_indices(gram_ut.shape[0], k=-1)] = 0.
        self.gram_ut = gram_ut

        gram_ut_t = np.dot(self.MATRIX_1.A, self.MATRIX_1.A.T)
        gram_ut_t[np.tril_indices(gram_ut_t.shape[0], k=-1)] = 0.
        self.gram_ut_t = gram_ut_t


class TestGramMatrixSparse(TestGramMatrix):

    def test_gram_matrix_sp(self):
        mat2 = gram_matrix_mkl(self.mat1)
        npt.assert_array_almost_equal(mat2.A, self.gram_ut)

        with self.assertRaises(ValueError):
            gram_matrix_mkl(self.mat1, out=np.zeros((self.mat1.shape[0], self.mat1.shape[0]), dtype=self.double_dtype))

    def test_gram_matrix_sp_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(self.single_dtype))
        npt.assert_array_almost_equal(mat2.A, self.gram_ut, decimal=5)

    def test_gram_matrix_d_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(self.single_dtype), dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut, decimal=5)

        mat2 = gram_matrix_mkl(self.mat1.astype(self.single_dtype), dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=self.single_dtype),  out_scalar=1.)
        mat2[np.tril_indices(mat2.shape[0], k=-1)] = 0.
        npt.assert_array_almost_equal(mat2, self.gram_ut, decimal=5)

        with self.assertRaises(ValueError):
            mat2 = gram_matrix_mkl(self.mat1.astype(self.single_dtype), dense=True,
                                   out=np.zeros((self.mat1.shape[1], self.mat1.shape[1])),
                                   out_scalar=1.)

    def test_gram_matrix_d(self):
        print(self.mat1)

        mat2 = gram_matrix_mkl(self.mat1, dense=True)
        print(mat2 - self.gram_ut)
        print(mat2[np.tril_indices(mat2.shape[0], k=1)])

        npt.assert_array_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(self.mat1, dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=self.double_dtype),  out_scalar=1.)
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
        mat = self.mat1.tocsc()
        mat2 = gram_matrix_mkl(mat, dense=True, cast=True)
        npt.assert_array_almost_equal(mat.A, self.mat1.A)
        npt.assert_array_almost_equal(mat2, self.gram_ut)


class TestGramMatrixDense(TestGramMatrix):

    def test_gram_matrix_dd_double(self):
        mat2 = gram_matrix_mkl(self.mat1.A, dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(self.mat1.A, dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=self.double_dtype),  out_scalar=1.)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(self.single_dtype).A, dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut, decimal=5)

        mat2 = gram_matrix_mkl(self.mat1.astype(self.single_dtype).A, dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=self.single_dtype),  out_scalar=1.)
        npt.assert_array_almost_equal(mat2, self.gram_ut, decimal=5)

    def test_gram_matrix_dd_double_F(self):
        mat2 = gram_matrix_mkl(np.asarray(self.mat1.A, order="F"), dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(np.asarray(self.mat1.A, order="F"), dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=self.double_dtype, order="F"),
                               out_scalar=1.)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_single_F(self):
        mat2 = gram_matrix_mkl(np.asarray(self.mat1.astype(self.single_dtype).A, order="F"), dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut, decimal=5)

        mat2 = gram_matrix_mkl(np.asarray(self.mat1.astype(self.single_dtype).A, order="F"), dense=True,
                               out=np.zeros((self.mat1.shape[1], self.mat1.shape[1]), dtype=self.single_dtype, order="F"),
                               out_scalar=1.)
        npt.assert_array_almost_equal(mat2, self.gram_ut, decimal=5)


class _TestGramMatrixComplex:

    double_dtype = np.cdouble
    single_dtype = np.csingle

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, _ = make_matrixes(200, 100, 300, 0.05, dtype=np.cdouble)

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()
        self.mat1_d = self.MATRIX_1.A

        gram_ut = np.dot(self.MATRIX_1.A.conj().T, self.MATRIX_1.A)
        gram_ut[np.tril_indices(gram_ut.shape[0], k=-1)] = 0.
        self.gram_ut = gram_ut

        gram_ut_t = np.dot(self.MATRIX_1.A, self.MATRIX_1.A.conj().T)
        gram_ut_t[np.tril_indices(gram_ut_t.shape[0], k=-1)] = 0.
        self.gram_ut_t = gram_ut_t

@unittest.skip
class TestGramMatrixSparseComplex(_TestGramMatrixComplex, TestGramMatrixSparse):
    pass

@unittest.skip
class TestGramMatrixDenseComplex(_TestGramMatrixComplex, TestGramMatrixDense):
    pass
