import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as _spsparse
from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1, MATRIX_2, VECTOR, make_matrixes, make_vector


class TestSparseVectorMultiplication(unittest.TestCase):

    double_dtype = np.float64
    single_dtype = np.float32
    export_complex = True

    sparse_func = _spsparse.csr_matrix
    sparse_args = {}

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, cls.MATRIX_2, cls.VECTOR = MATRIX_1.copy(), MATRIX_2.copy(), VECTOR.copy() 

    def make_2d(self, arr):
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    def setUp(self):
        self.mat1 = self.sparse_func(self.MATRIX_1.copy(), **self.sparse_args)
        self.mat2 = self.VECTOR.copy()

        self.mat1_d = np.asarray(self.MATRIX_1.A, order="C")
        self.mat2_d = self.VECTOR.copy()

    def test_mult_1d(self):
        mat3 = dot_product_mkl(self.mat1.astype(self.double_dtype), self.mat2, cast=True)
        mat3_np = np.dot(self.mat1_d, self.mat2_d)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_1d_out(self):
        mat3_np = np.dot(self.mat1_d, self.mat2_d)
        mat3_np += 2

        out = np.ones(mat3_np.shape, dtype=self.double_dtype)
        mat3 = dot_product_mkl(self.mat1, self.mat2, cast=True, out=out, out_scalar=2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(mat3_np, out)
        self.assertEqual(id(mat3), id(out))

    def test_mult_1d_float32(self):
        d1, d2 = self.mat1.astype(self.single_dtype), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(self.mat1_d, self.mat2_d)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_1d_float32_out(self):
        mat3_np = np.dot(self.mat1_d, self.mat2_d)
        mat3_np += 2

        out = np.ones(mat3_np.shape, dtype=self.single_dtype)
        mat3 = dot_product_mkl(self.mat1.astype(self.single_dtype), self.mat2.astype(self.single_dtype), cast=False,
                               out=out, out_scalar=2)

        npt.assert_array_almost_equal(mat3_np, out, decimal=5)
        self.assertEqual(id(out), id(mat3))

        with self.assertRaises(ValueError):
            mat3 = dot_product_mkl(self.mat1.astype(self.single_dtype), self.mat2.astype(self.single_dtype), cast=True,
                                   out=np.ones(mat3_np.shape).astype(self.double_dtype), out_scalar=2)

    def test_mult_1d_both_float32(self):
        mat3 = dot_product_mkl(self.mat1.astype(self.single_dtype), self.mat2.astype(self.single_dtype), cast=True)
        mat3_np = np.dot(self.mat1_d, self.mat2_d)

        npt.assert_array_almost_equal(mat3_np, mat3, decimal=5)

    def test_mult_2d(self):
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(self.double_dtype)), self.make_2d(self.mat2), cast=True)
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d_out(self):
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))
        mat3_np += 2

        out = np.ones(mat3_np.shape, dtype=self.double_dtype)
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(self.double_dtype)), self.make_2d(self.mat2), cast=True,
                               out=out, out_scalar=2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        self.assertEqual(id(out), id(mat3))

        with self.assertRaises(ValueError):
            mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(self.double_dtype)), self.make_2d(self.mat2), cast=True,
                                   out=np.ones(mat3_np.shape).astype(self.single_dtype), out_scalar=2)

    def test_mult_2d_float32(self):
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(self.single_dtype)), self.make_2d(self.mat2), cast=True)
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d_other_float32(self):
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(self.single_dtype)), self.make_2d(self.mat2), cast=True)
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d_both_float32(self):
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(self.single_dtype)), self.make_2d(self.mat2.astype(self.single_dtype)))
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))

        npt.assert_array_almost_equal(mat3_np, mat3, decimal=5)

    def test_mult_2d_both_float32_out(self):
        mat3_np = np.dot(self.make_2d(self.mat1_d), self.make_2d(self.mat2_d))
        mat3_np += 2

        out = np.ones(mat3_np.shape, dtype=self.single_dtype)
        mat3 = dot_product_mkl(self.make_2d(self.mat1.astype(self.single_dtype)), self.make_2d(self.mat2.astype(self.single_dtype)),
                               out=out, out_scalar=2)

        npt.assert_array_almost_equal(mat3_np, mat3, decimal=5)
        self.assertEqual(id(out), id(mat3))


class TestSparseVectorMultiplicationCSC(TestSparseVectorMultiplication):

    sparse_func = _spsparse.csc_matrix
    sparse_args = {}


class TestSparseVectorMultiplicationBSR(TestSparseVectorMultiplication):

    sparse_func = _spsparse.bsr_matrix
    sparse_args = {"blocksize": (10, 10)}

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
        self.mat2 = self.MATRIX_2.copy()
        self.mat1 = self.VECTOR.copy()

        self.mat2_d = np.asarray(self.MATRIX_2.A, order="C")
        self.mat1_d = self.VECTOR.copy()

    def make_2d(self, arr):
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def test_mult_outer_product_ds(self):
        d1, d2 = self.mat1.reshape(-1, 1), self.sparse_func(self.mat2_d[:, 0].reshape(1, -1))

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3_np += 2.
        out = np.ones(mat3_np.shape, dtype=self.double_dtype)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=2.)

        npt.assert_array_almost_equal(mat3_np, mat3)
        self.assertEqual(id(out), id(mat3))

    def test_mult_outer_product_sd(self):
        d1, d2 = self.sparse_func(self.mat1.reshape(-1, 1)), self.mat2_d[:, 0].reshape(1, -1).copy()

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3_np += 2.
        out = np.ones(mat3_np.shape, dtype=self.double_dtype)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=2.)

        npt.assert_array_almost_equal(mat3_np, mat3)
        self.assertEqual(id(out), id(mat3))


class TestVectorSparseMultiplicationCSC(TestVectorSparseMultiplication):

    sparse_func = _spsparse.csc_matrix


class TestVectorSparseMultiplicationBSR(TestVectorSparseMultiplication):

    sparse_func = _spsparse.bsr_matrix


class TestVectorVectorMultplication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, cls.VECTOR = MATRIX_1.copy(), VECTOR.copy() 

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


class _ComplexMixin:

    double_dtype = np.cdouble
    single_dtype = np.csingle
    export_complex = True

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, cls.MATRIX_2 = make_matrixes(200, 100, 300, 0.05, dtype=np.cdouble)
        cls.VECTOR = make_vector(300, complex=True)


class TestSparseVectorMultiplicationComplex(_ComplexMixin, TestSparseVectorMultiplication):
    pass


class TestSparseVectorMultiplicationCSCComplex(_ComplexMixin, TestSparseVectorMultiplicationCSC):
    pass


class TestSparseVectorMultiplicationBSRComplex(_ComplexMixin, TestSparseVectorMultiplicationBSR):
    pass


class TestVectorSparseMultiplicationComplex(_ComplexMixin, TestVectorSparseMultiplication):
    pass


class TestVectorSparseMultiplicationCSCComplex(_ComplexMixin, TestVectorSparseMultiplicationCSC):
    pass


class TestVectorSparseMultiplicationBSRComplex(_ComplexMixin, TestVectorSparseMultiplicationBSR):
    pass


class TestVectorVectorMultplicationComplex(_ComplexMixin, TestVectorVectorMultplication):
    pass