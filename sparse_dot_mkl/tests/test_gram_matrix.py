import unittest
import numpy as np
from scipy.sparse import csr_matrix
from sparse_dot_mkl import gram_matrix_mkl
from sparse_dot_mkl.tests.test_mkl import (
    MATRIX_1,
    make_matrixes,
    np_almost_equal
)


class TestGramMatrix(unittest.TestCase):
    double_dtype = np.float64
    single_dtype = np.float32

    sparse_func = csr_matrix

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1 = cls.sparse_func(MATRIX_1.copy())

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()
        self.mat1_d = self.MATRIX_1.toarray()

        gram_ut = np.dot(self.MATRIX_1.toarray().T, self.MATRIX_1.toarray())
        gram_ut[np.tril_indices(gram_ut.shape[0], k=-1)] = 0.0
        self.gram_ut = gram_ut

        gram_ut_t = np.dot(self.MATRIX_1.toarray(), self.MATRIX_1.toarray().T)
        gram_ut_t[np.tril_indices(gram_ut_t.shape[0], k=-1)] = 0.0
        self.gram_ut_t = gram_ut_t


class TestGramMatrixSparse(TestGramMatrix):
    def test_gram_matrix_sp(self):
        mat2 = gram_matrix_mkl(self.mat1)
        np_almost_equal(mat2.toarray(), self.gram_ut)

        with self.assertRaises(ValueError):
            gram_matrix_mkl(
                self.mat1,
                out=np.zeros(
                    (self.mat1.shape[0], self.mat1.shape[0]),
                    dtype=self.double_dtype
                ),
            )

    def test_gram_matrix_sp_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(self.single_dtype))
        np_almost_equal(mat2.toarray(), self.gram_ut, decimal=5)

    def test_gram_matrix_d_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(self.single_dtype), dense=True)
        np_almost_equal(mat2, self.gram_ut, decimal=5)

        mat2 = gram_matrix_mkl(
            self.mat1.astype(self.single_dtype),
            dense=True,
            out=np.zeros(
                (self.mat1.shape[1], self.mat1.shape[1]),
                dtype=self.single_dtype
            ),
            out_scalar=1.0,
        )
        mat2[np.tril_indices(mat2.shape[0], k=-1)] = 0.0
        np_almost_equal(mat2, self.gram_ut, decimal=5)

        with self.assertRaises(ValueError):
            mat2 = gram_matrix_mkl(
                self.mat1.astype(self.single_dtype),
                dense=True,
                out=np.zeros((self.mat1.shape[1], self.mat1.shape[1])),
                out_scalar=1.0,
            )

    def test_gram_matrix_d(self):
        print(self.mat1)

        mat2 = gram_matrix_mkl(self.mat1, dense=True)
        print(mat2 - self.gram_ut)
        print(mat2[np.tril_indices(mat2.shape[0], k=1)])

        np_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(
            self.mat1,
            dense=True,
            out=np.zeros(
                (self.mat1.shape[1], self.mat1.shape[1]),
                dtype=self.double_dtype
            ),
            out_scalar=1.0,
        )
        mat2[np.tril_indices(mat2.shape[0], k=-1)] = 0.0
        np_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_sp_t(self):
        mat2 = gram_matrix_mkl(self.mat1, transpose=True)
        np_almost_equal(mat2.toarray(), self.gram_ut_t)

    def test_gram_matrix_d_t(self):
        mat2 = gram_matrix_mkl(self.mat1, dense=True, transpose=True)
        np_almost_equal(mat2, self.gram_ut_t)

    def test_gram_matrix_csc_sp(self):
        mat2 = gram_matrix_mkl(self.mat1.tocsc(), cast=True)
        np_almost_equal(mat2.toarray(), self.gram_ut)

    def test_gram_matrix_csc_d(self):
        mat = self.mat1.tocsc()
        mat2 = gram_matrix_mkl(mat, dense=True, cast=True)
        np_almost_equal(mat.toarray(), self.mat1.toarray())
        np_almost_equal(mat2, self.gram_ut)


class TestGramMatrixDense(TestGramMatrix):
    def test_gram_matrix_dd_double(self):
        mat2 = gram_matrix_mkl(self.mat1.toarray(), dense=True)
        np_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(
            self.mat1.toarray(),
            dense=True,
            out=np.zeros(
                (self.mat1.shape[1], self.mat1.shape[1]),
                dtype=self.double_dtype
            ),
            out_scalar=1.0,
        )
        np_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_single(self):
        mat2 = gram_matrix_mkl(
            self.mat1.astype(self.single_dtype).toarray(), dense=True
        )
        np_almost_equal(mat2, self.gram_ut, decimal=5)

        mat2 = gram_matrix_mkl(
            self.mat1.astype(self.single_dtype).toarray(),
            dense=True,
            out=np.zeros(
                (self.mat1.shape[1], self.mat1.shape[1]),
                dtype=self.single_dtype
            ),
            out_scalar=1.0,
        )
        np_almost_equal(mat2, self.gram_ut, decimal=5)

    def test_gram_matrix_dd_double_F(self):
        mat2 = gram_matrix_mkl(
            np.asarray(self.mat1.toarray(), order="F"),
            dense=True
        )
        np_almost_equal(mat2, self.gram_ut)

        mat2 = gram_matrix_mkl(
            np.asarray(self.mat1.toarray(), order="F"),
            dense=True,
            out=np.zeros(
                (self.mat1.shape[1], self.mat1.shape[1]),
                dtype=self.double_dtype,
                order="F",
            ),
            out_scalar=1.0,
        )
        np_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_single_F(self):
        mat2 = gram_matrix_mkl(
            np.asarray(
                self.mat1.astype(self.single_dtype).toarray(),
                order="F"
            ),
            dense=True,
        )
        np_almost_equal(mat2, self.gram_ut, decimal=5)

        mat2 = gram_matrix_mkl(
            np.asarray(
                self.mat1.astype(self.single_dtype).toarray(),
                order="F"
            ),
            dense=True,
            out=np.zeros(
                (self.mat1.shape[1], self.mat1.shape[1]),
                dtype=self.single_dtype,
                order="F",
            ),
            out_scalar=1.0,
        )
        np_almost_equal(mat2, self.gram_ut, decimal=5)


class _TestGramMatrixComplex:
    double_dtype = np.cdouble
    single_dtype = np.csingle

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, _ = make_matrixes(200, 100, 300, 0.05, dtype=np.cdouble)

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()
        self.mat1_d = self.MATRIX_1.toarray()

        gram_ut = np.dot(self.MATRIX_1.toarray().conj().T, self.MATRIX_1.toarray())
        gram_ut[np.tril_indices(gram_ut.shape[0], k=-1)] = 0.0
        self.gram_ut = gram_ut

        gram_ut_t = np.dot(self.MATRIX_1.toarray(), self.MATRIX_1.toarray().conj().T)
        gram_ut_t[np.tril_indices(gram_ut_t.shape[0], k=-1)] = 0.0
        self.gram_ut_t = gram_ut_t


@unittest.skip
class TestGramMatrixSparseComplex(_TestGramMatrixComplex, TestGramMatrixSparse):
    pass


@unittest.skip
class TestGramMatrixDenseComplex(_TestGramMatrixComplex, TestGramMatrixDense):
    pass


try:
    from scipy.sparse import (
        csr_array
    )

    class TestGramMatrixSparseArray(TestGramMatrixSparse):
        sparse_func = csr_array

except ImportError:
    pass
