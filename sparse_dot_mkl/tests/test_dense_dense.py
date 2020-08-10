import unittest
import numpy as np
import numpy.testing as npt
from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1, MATRIX_2


class TestDenseDenseMultiplication(unittest.TestCase):

    order = "C"

    def setUp(self):
        self.mat1 = MATRIX_1.copy().A
        self.mat2 = MATRIX_2.copy().A

    def test_float32(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2.astype(np.float32)
        mat3_np = np.dot(d1, d2)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3 = dot_product_mkl(d1, d2, out=np.ones(mat3_np.shape, dtype=np.float32, order=self.order), out_scalar=3)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)

    def test_float64(self):
        d1, d2 = self.mat1, self.mat2
        mat3_np = np.dot(d1, d2)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3_np += 3.
        out = np.ones(mat3_np.shape, dtype=np.float64, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3)
        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(mat3_np, out)
        self.assertEqual(id(mat3), id(out))

    def test_float64_cast(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2
        mat3_np = np.dot(d1, d2)

        mat3 = dot_product_mkl(d1, d2, cast=True)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3 = dot_product_mkl(d1, d2, out=np.ones(mat3_np.shape, dtype=np.float64, order=self.order), out_scalar=3,
                               cast=True)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)

    def test_outer_product(self):
        d1, d2 = self.mat1[:, 0].reshape(-1, 1).copy(), self.mat2[0, :].reshape(1, -1).copy()

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3 = dot_product_mkl(d1, d2, out=np.ones(mat3_np.shape, dtype=np.float64, order="C"), out_scalar=3)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)

    def test_fails(self):
        mat3_np = np.dot(self.mat1, self.mat2)
        n, m = mat3_np.shape

        with self.assertRaises(ValueError):
            dot_product_mkl(self.mat1[0:10, 0:20], self.mat2[0:20, 0:10])

        with self.assertRaises(ValueError):
            non_contig_out = np.ones((n + 10, m + 10), order=self.order)
            dot_product_mkl(self.mat1, self.mat2, out=non_contig_out[0:n, 0:m], out_scalar=3)

        with self.assertRaises(ValueError):
            non_float_out = np.ones((n, m), dtype=np.int32, order=self.order)
            dot_product_mkl(self.mat1, self.mat2, out=non_float_out, out_scalar=3)

        with self.assertRaises(ValueError):
            non_shape_out = np.ones((n, m), order=self.order).T
            dot_product_mkl(self.mat1, self.mat2, out=non_shape_out, out_scalar=3)


class TestDenseDenseFCMultiplication(TestDenseDenseMultiplication):

    order = "F"

    def setUp(self):
        self.mat1 = np.asarray(MATRIX_1.copy().A, order='F')
        self.mat2 = np.asarray(MATRIX_2.copy().A, order='C')


class TestDenseDenseFFMultiplication(TestDenseDenseMultiplication):

    order = "F"

    def setUp(self):
        self.mat1 = np.asarray(MATRIX_1.copy().A, order='F')
        self.mat2 = np.asarray(MATRIX_2.copy().A, order='F')


class TestDenseDenseCFMultiplication(TestDenseDenseMultiplication):

    order = "C"

    def setUp(self):
        self.mat1 = np.asarray(MATRIX_1.copy().A, order='C')
        self.mat2 = np.asarray(MATRIX_2.copy().A, order='F')
