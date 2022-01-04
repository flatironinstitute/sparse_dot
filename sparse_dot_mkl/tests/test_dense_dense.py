import unittest
import numpy as np
import numpy.testing as npt
from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1, MATRIX_2, make_matrixes


class TestDenseDenseMultiplication(unittest.TestCase):

    order = "C"

    double_dtype = np.float64
    single_dtype = np.float32

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, cls.MATRIX_2 = MATRIX_1.copy().A, MATRIX_2.copy().A

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()
        self.mat2 = self.MATRIX_2.copy()

    def test_float32(self):
        d1, d2 = self.mat1.astype(self.single_dtype), self.mat2.astype(self.single_dtype)
        mat3_np = np.dot(d1, d2)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3 = dot_product_mkl(d1, d2, out=np.ones(mat3_np.shape, dtype=self.single_dtype, order=self.order), out_scalar=3)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)

    def test_float64(self):
        d1, d2 = self.mat1, self.mat2
        mat3_np = np.dot(d1, d2)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3_np += 3.
        out = np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3)
        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(mat3_np, out)
        self.assertEqual(id(mat3), id(out))

    def test_float64_cast(self):
        d1, d2 = self.mat1.astype(self.single_dtype), self.mat2
        mat3_np = np.dot(d1, d2)

        mat3 = dot_product_mkl(d1, d2, cast=True)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3 = dot_product_mkl(d1, d2, out=np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order), out_scalar=3,
                               cast=True)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)

    def test_outer_product(self):
        d1, d2 = self.mat1[:, 0].reshape(-1, 1).copy(), self.mat2[0, :].reshape(1, -1).copy()

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3 = dot_product_mkl(d1, d2, out=np.ones(mat3_np.shape, dtype=self.double_dtype, order="C"), out_scalar=3)
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
        self.mat1 = np.asarray(self.MATRIX_1.copy(), order='F')
        self.mat2 = np.asarray(self.MATRIX_2.copy(), order='C')


class TestDenseDenseFFMultiplication(TestDenseDenseMultiplication):

    order = "F"

    def setUp(self):
        self.mat1 = np.asarray(self.MATRIX_1.copy(), order='F')
        self.mat2 = np.asarray(self.MATRIX_2.copy(), order='F')


class TestDenseDenseCFMultiplication(TestDenseDenseMultiplication):

    order = "C"

    def setUp(self):
        self.mat1 = np.asarray(self.MATRIX_1.copy(), order='C')
        self.mat2 = np.asarray(self.MATRIX_2.copy(), order='F')

class _ComplexMixin:

    double_dtype = np.cdouble
    single_dtype = np.csingle

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, cls.MATRIX_2 = make_matrixes(200, 100, 300, 0.05, dtype=np.cdouble)
        cls.MATRIX_1 = cls.MATRIX_1.A
        cls.MATRIX_2 = cls.MATRIX_2.A

    def test_inherits(self):

        self.assertIs(self.double_dtype, np.cdouble)
        self.assertIs(self.single_dtype, np.csingle)
        self.assertTrue(self.MATRIX_1.dtype == np.cdouble)

class TestDDComplexMultiplication(_ComplexMixin, TestDenseDenseMultiplication):

    pass

class TestDDFCComplexMultiplication(_ComplexMixin, TestDenseDenseFCMultiplication):

    pass

class TestDDCFComplexMultiplication(_ComplexMixin, TestDenseDenseCFMultiplication):

    pass

class TestDDFFComplexMultiplication(_ComplexMixin, TestDenseDenseFFMultiplication):

    pass