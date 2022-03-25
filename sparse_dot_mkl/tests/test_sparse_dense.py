import unittest
import numpy as np
import numpy.testing as npt
from scipy.sparse import csr_matrix
from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1, MATRIX_2, make_matrixes


class TestSparseDenseMultiplication(unittest.TestCase):

    order = "C"

    double_dtype = np.float64
    single_dtype = np.float32

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, cls.MATRIX_2 = MATRIX_1.copy(), MATRIX_2.copy()

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()
        self.mat2 = self.MATRIX_2.copy()
        self.mat1_d = self.mat1.A
        self.mat2_d = self.mat2.A

    def test_float32_b_sparse(self):
        d1, d2 = self.mat1_d.astype(self.single_dtype), self.mat2.astype(self.single_dtype)

        with self.assertWarns(DeprecationWarning):
            mat3 = dot_product_mkl(d1, d2, debug=True)
            mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3_np += 3.
        out = np.ones(mat3_np.shape, dtype=self.single_dtype, order=self.order)

        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(mat3_np, out)

        mat3 += 1.

        npt.assert_array_almost_equal(mat3, out)
        self.assertEqual(id(mat3), id(out))

    def test_float64_b_sparse(self):
        d1, d2 = self.mat1_d.astype(self.double_dtype), self.mat2.astype(self.double_dtype)

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3 = dot_product_mkl(d1, d2, out=np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order), out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)

    def test_float64_cast_b_sparse(self):
        d1, d2 = self.mat1_d.astype(self.single_dtype), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

        out = np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3., cast=True)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float32_csc_sparse(self):
        d1, d2 = self.mat1_d.astype(self.single_dtype), self.mat2.astype(self.single_dtype).tocsc()
        mat3_np = np.dot(d1, d2.A)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=self.single_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float32_bsr_sparse(self):

        bs = min(*self.mat2.shape, 10)
        d1, d2 = self.mat1_d.astype(self.single_dtype), self.mat2.astype(self.single_dtype).tobsr(blocksize=(bs, bs))
        mat3_np = np.dot(d1, d2.A)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=self.single_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_csc_sparse(self):
        d1, d2 = self.mat1_d, self.mat2.tocsc()

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2.A)
        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_bsr_sparse(self):
        bs = min(*self.mat2.shape, 10)
        d1, d2 = self.mat1_d, self.mat2.tobsr(blocksize=(bs, bs))

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_cast_csc_sparse(self):
        d1, d2 = self.mat1_d.astype(self.single_dtype), self.mat2.tocsc()

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3., cast=True)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_cast_bsr_sparse(self):

        bs = min(*self.mat2.shape, 10)
        d1, d2 = self.mat1_d.astype(self.single_dtype), self.mat2.tobsr(blocksize=(bs, bs))

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3., cast=True)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float32_a_sparse(self):
        d1, d2 = self.mat1.astype(self.single_dtype), self.mat2_d.astype(self.single_dtype)

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        out = np.ones(mat3_np.shape, dtype=self.single_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_a_sparse(self):
        d1, d2 = self.mat1, self.mat2_d

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        out = np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_a_csc_sparse(self):
        d1, d2 = self.mat1.tocsc(), self.mat2_d

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d1.A, self.mat1_d)

        out = np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_a_bsr_sparse(self):

        bs = min(*self.mat1.shape, 10)
        d1, d2 = self.mat1.tobsr(blocksize=(bs, bs)), self.mat2_d

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d1.A, self.mat1_d)

        out = np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float32_a_csc_sparse(self):
        d1, d2 = self.mat1.astype(self.single_dtype).tocsc(), self.mat2_d.astype(self.single_dtype)

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d1.A, self.mat1_d)

        out = np.ones(mat3_np.shape, dtype=self.single_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float32_a_bsr_sparse(self):

        bs = min(*self.mat1.shape, 10)
        d1, d2 = self.mat1.astype(self.single_dtype).tobsr(blocksize=(bs, bs)), self.mat2_d.astype(self.single_dtype)

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d1.A, self.mat1_d)

        out = np.ones(mat3_np.shape, dtype=self.single_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_cast_a_sparse(self):
        d1, d2 = self.mat1.astype(self.single_dtype), self.mat2_d

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        out = np.ones(mat3_np.shape, dtype=self.double_dtype, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3., cast=True)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))


class TestSparseDenseFMultiplication(TestSparseDenseMultiplication):

    order = "F"

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()
        self.mat2 = self.MATRIX_2.copy()

        self.mat1_d = np.asarray(self.MATRIX_1.A, order=self.order)
        self.mat2_d = np.asarray(self.MATRIX_2.A, order=self.order)


class TestSparseVectorDenseCMultiplication(TestSparseDenseMultiplication):

    order = "C"

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()[0, :]
        self.mat2 = self.MATRIX_2.copy()

        self.mat1_d = np.asarray(self.MATRIX_1.A, order=self.order)[0, :].reshape(1, -1)
        self.mat2_d = np.asarray(self.MATRIX_2.A, order=self.order)


class TestSparseVector2DenseCMultiplication(TestSparseDenseMultiplication):

    order = "C"

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()
        self.mat2 = self.MATRIX_2.copy()[:, 0]

        self.mat1_d = np.asarray(self.MATRIX_1.A, order=self.order)
        self.mat2_d = np.asarray(self.MATRIX_2.A, order=self.order)[:, 0].reshape(-1, 1)


class TestSparseVectorDenseFMultiplication(TestSparseDenseMultiplication):

    order = "F"

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()[0, :]
        self.mat2 = self.MATRIX_2.copy()

        self.mat1_d = np.asarray(self.MATRIX_1.A, order=self.order)[0, :].reshape(1, -1)
        self.mat2_d = np.asarray(self.MATRIX_2.A, order=self.order)


class TestSparseVector2DenseFMultiplication(TestSparseDenseMultiplication):

    order = "F"

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()
        self.mat2 = self.MATRIX_2.copy()[:, 0]

        self.mat1_d = np.asarray(self.MATRIX_1.A, order=self.order)
        self.mat2_d = np.asarray(self.MATRIX_2.A, order=self.order)[:, 0].reshape(-1, 1)

class _ComplexMixin:

    double_dtype = np.cdouble
    single_dtype = np.csingle

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, cls.MATRIX_2 = make_matrixes(200, 100, 300, 0.05, dtype=np.cdouble)


class TestSparseDenseMultiplicationComplex(_ComplexMixin, TestSparseDenseMultiplication):

    pass

class TestSparseDenseFMultiplicationComplex(_ComplexMixin, TestSparseDenseFMultiplication):

    pass

class TestSparseVectorDenseCMultiplicationComplex(_ComplexMixin, TestSparseVectorDenseCMultiplication):

    pass