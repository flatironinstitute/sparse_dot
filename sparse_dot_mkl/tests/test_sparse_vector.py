import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as _spsparse
from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1, MATRIX_2, VECTOR


class TestSparseVectorMultiplication(unittest.TestCase):

    def make_2d(self, arr):
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = VECTOR.copy()

        self.mat1_d = np.asarray(MATRIX_1.A, order="C")
        self.mat2_d = VECTOR.copy()

    def test_mult_1d(self):
        mat3 = dot_product_mkl(self.mat1.astype(np.float64), self.mat2, cast=True)
        mat3_np = np.dot(self.mat1_d, self.mat2_d)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_1d_out(self):
        mat3_np = np.dot(self.mat1_d, self.mat2_d)
        mat3_np += 2

        out = np.ones(mat3_np.shape)
        mat3 = dot_product_mkl(self.mat1, self.mat2, cast=True, out=out, out_scalar=2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(mat3_np, out)
        self.assertEqual(id(mat3), id(out))

    def test_mult_1d_float32(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(self.mat1_d, self.mat2_d)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_1d_float32_out(self):
        mat3_np = np.dot(self.mat1_d, self.mat2_d)
        mat3_np += 2

        out = np.ones(mat3_np.shape, dtype=np.float32)
        mat3 = dot_product_mkl(self.mat1.astype(np.float32), self.mat2.astype(np.float32), cast=False,
                               out=out, out_scalar=2)

        npt.assert_array_almost_equal(mat3_np, out, decimal=5)
        self.assertEqual(id(out), id(mat3))

        with self.assertRaises(ValueError):
            mat3 = dot_product_mkl(self.mat1.astype(np.float32), self.mat2.astype(np.float32), cast=True,
                                   out=np.ones(mat3_np.shape).astype(np.float64), out_scalar=2)

    def test_mult_1d_both_float32(self):
        mat3 = dot_product_mkl(self.mat1.astype(np.float32), self.mat2.astype(np.float32), cast=True)
        mat3_np = np.dot(self.mat1_d, self.mat2_d)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d(self):
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(np.float64)), self.make_2d(self.mat2), cast=True)
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d_out(self):
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))
        mat3_np += 2

        out = np.ones(mat3_np.shape, dtype=np.float64)
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(np.float64)), self.make_2d(self.mat2), cast=True,
                               out=out, out_scalar=2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        self.assertEqual(id(out), id(mat3))

        with self.assertRaises(ValueError):
            mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(np.float64)), self.make_2d(self.mat2), cast=True,
                                   out=np.ones(mat3_np.shape).astype(np.float32), out_scalar=2)

    def test_mult_2d_float32(self):
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(np.float32)), self.make_2d(self.mat2), cast=True)
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d_other_float32(self):
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(np.float32)), self.make_2d(self.mat2), cast=True)
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d_both_float32(self):
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(np.float32)), self.make_2d(self.mat2.astype(np.float32)))
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))

        npt.assert_array_almost_equal(mat3_np, mat3, decimal=5)

    def test_mult_2d_both_float32_out(self):
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))
        mat3_np += 2

        out = np.ones(mat3_np.shape, dtype=np.float32)
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(np.float32)), self.make_2d(self.mat2.astype(np.float32)),
                               out=out, out_scalar=2)

        npt.assert_array_almost_equal(mat3_np, mat3, decimal=5)
        self.assertEqual(id(out), id(mat3))


class TestSparseVectorMultiplicationCSC(TestSparseVectorMultiplication):

    def setUp(self):
        self.mat1 = _spsparse.csc_matrix(MATRIX_1).copy()
        self.mat2 = VECTOR.copy()

        self.mat1_d = np.asarray(MATRIX_1.A, order="C")
        self.mat2_d = VECTOR.copy()


class TestSparseVectorMultiplicationBSR(TestSparseVectorMultiplication):

    def setUp(self):
        self.mat1 = _spsparse.bsr_matrix(MATRIX_1, blocksize=(10, 10)).copy()
        self.mat2 = VECTOR.copy()

        self.mat1_d = np.asarray(MATRIX_1.A, order="C")
        self.mat2_d = VECTOR.copy()


class TestSparseVectorMultiplicationCOO(unittest.TestCase):

    def setUp(self):
        self.mat1 = _spsparse.coo_matrix(MATRIX_1).copy()
        self.mat2 = VECTOR.copy()

        self.mat1_d = np.asarray(MATRIX_1.A, order="C")
        self.mat2_d = VECTOR.copy()

    def make_2d(self, arr):
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    def test_fails(self):
        with self.assertRaises(ValueError):
            dot_product_mkl(self.mat1, self.make_2d(self.mat2), cast=True)

        with self.assertRaises(ValueError):
            dot_product_mkl(self.make_2d(self.mat2), self.mat1.T, cast=True)


class TestVectorSparseMultiplication(TestSparseVectorMultiplication):

    sparse_func = _spsparse.csr_matrix
    sparse_args = {}

    def setUp(self):
        self.mat2 = MATRIX_2.copy()
        self.mat1 = VECTOR.copy()

        self.mat2_d = np.asarray(MATRIX_2.A, order="C")
        self.mat1_d = VECTOR.copy()

    def make_2d(self, arr):
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def test_mult_outer_product_ds(self):
        d1, d2 = self.mat1.reshape(-1, 1), self.sparse_func(self.mat2_d[:, 0].reshape(1, -1))

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3_np += 2.
        out = np.ones(mat3_np.shape)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=2.)

        npt.assert_array_almost_equal(mat3_np, mat3)
        self.assertEqual(id(out), id(mat3))

    def test_mult_outer_product_sd(self):
        d1, d2 = self.sparse_func(self.mat1.reshape(-1, 1)), self.mat2_d[:, 0].reshape(1, -1).copy()

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3_np += 2.
        out = np.ones(mat3_np.shape)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=2.)

        npt.assert_array_almost_equal(mat3_np, mat3)
        self.assertEqual(id(out), id(mat3))


class TestVectorSparseMultiplicationCSC(TestVectorSparseMultiplication):

    sparse_func = _spsparse.csc_matrix


class TestVectorSparseMultiplicationBSR(TestVectorSparseMultiplication):

    sparse_func = _spsparse.bsr_matrix


class TestVectorVectorMultplication(unittest.TestCase):

    def test_1d_1d(self):
        mat3 = dot_product_mkl(VECTOR, VECTOR)
        mat3_np = np.dot(VECTOR, VECTOR)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_1d_2d(self):
        mat3 = dot_product_mkl(VECTOR, VECTOR.reshape(-1, 1))
        mat3_np = np.dot(VECTOR, VECTOR.reshape(-1, 1))

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_2d_1d(self):
        mat3 = dot_product_mkl(VECTOR.reshape(1, -1), VECTOR)
        mat3_np = np.dot(VECTOR.reshape(1, -1), VECTOR)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_2d_2d(self):
        mat3 = dot_product_mkl(VECTOR.reshape(1, -1), VECTOR.reshape(-1, 1))
        mat3_np = np.dot(VECTOR.reshape(1, -1), VECTOR.reshape(-1, 1))

        npt.assert_array_almost_equal(mat3_np, mat3)
