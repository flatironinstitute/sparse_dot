import unittest
import numpy as np
import numpy.testing as npt
from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1, MATRIX_2


class TestSparseDenseMultiplication(unittest.TestCase):

    order = "C"

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()

        self.mat1_d = MATRIX_1.A
        self.mat2_d = MATRIX_2.A

    def test_float32_b_sparse(self):
        d1, d2 = self.mat1_d.astype(np.float32), self.mat2.astype(np.float32)

        mat3 = dot_product_mkl(d1, d2, debug=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3_np += 3.
        out = np.ones(mat3_np.shape, dtype=np.float32, order=self.order)

        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3., debug=True)
        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(mat3_np, out)

        mat3 += 1.

        npt.assert_array_almost_equal(mat3, out)
        self.assertEqual(id(mat3), id(out))

    def test_float64_b_sparse(self):
        d1, d2 = self.mat1_d, self.mat2

        mat3 = dot_product_mkl(d1, d2, debug=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

        mat3 = dot_product_mkl(d1, d2, out=np.ones(mat3_np.shape, dtype=np.float64, order=self.order), out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)

    def test_float64_cast_b_sparse(self):
        d1, d2 = self.mat1_d.astype(np.float32), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

        out = np.ones(mat3_np.shape, dtype=np.float64, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3., cast=True)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float32_csc_sparse(self):
        d1, d2 = self.mat1_d.astype(np.float32), self.mat2.astype(np.float32).tocsc()
        mat3_np = np.dot(d1, d2.A)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=np.float32, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float32_bsr_sparse(self):

        bs = min(*self.mat2.shape, 10)
        d1, d2 = self.mat1_d.astype(np.float32), self.mat2.astype(np.float32).tobsr(blocksize=(bs, bs))
        mat3_np = np.dot(d1, d2.A)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=np.float32, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_csc_sparse(self):
        d1, d2 = self.mat1_d, self.mat2.tocsc()

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=np.float64, order=self.order)
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

        out = np.ones(mat3_np.shape, dtype=np.float64, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_cast_csc_sparse(self):
        d1, d2 = self.mat1_d.astype(np.float32), self.mat2.tocsc()

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=np.float64, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3., cast=True)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_cast_bsr_sparse(self):

        bs = min(*self.mat2.shape, 10)
        d1, d2 = self.mat1_d.astype(np.float32), self.mat2.tobsr(blocksize=(bs, bs))

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

        out = np.ones(mat3_np.shape, dtype=np.float64, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3., cast=True)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float32_a_sparse(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2_d.astype(np.float32)

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        out = np.ones(mat3_np.shape, dtype=np.float32, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_a_sparse(self):
        d1, d2 = self.mat1, self.mat2_d

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        out = np.ones(mat3_np.shape, dtype=np.float64, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_a_csc_sparse(self):
        d1, d2 = self.mat1.tocsc(), self.mat2_d

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d1.A, self.mat1_d)

        out = np.ones(mat3_np.shape, dtype=np.float64, order=self.order)
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

        out = np.ones(mat3_np.shape, dtype=np.float64, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float32_a_csc_sparse(self):
        d1, d2 = self.mat1.astype(np.float32).tocsc(), self.mat2_d.astype(np.float32)

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d1.A, self.mat1_d)

        out = np.ones(mat3_np.shape, dtype=np.float32, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float32_a_bsr_sparse(self):

        bs = min(*self.mat1.shape, 10)
        d1, d2 = self.mat1.astype(np.float32).tobsr(blocksize=(bs, bs)), self.mat2_d.astype(np.float32)

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d1.A, self.mat1_d)

        out = np.ones(mat3_np.shape, dtype=np.float32, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3.)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))

    def test_float64_cast_a_sparse(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2_d

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

        out = np.ones(mat3_np.shape, dtype=np.float64, order=self.order)
        mat3 = dot_product_mkl(d1, d2, out=out, out_scalar=3., cast=True)
        npt.assert_array_almost_equal(mat3_np + 3., mat3)
        self.assertEqual(id(mat3), id(out))


class TestSparseDenseFMultiplication(TestSparseDenseMultiplication):

    order = "F"

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()

        self.mat1_d = np.asarray(MATRIX_1.A, order="F")
        self.mat2_d = np.asarray(MATRIX_2.A, order="F")


class TestSparseVectorDenseCMultiplication(TestSparseDenseMultiplication):

    order = "C"

    def setUp(self):
        self.mat1 = MATRIX_1.copy()[0, :]
        self.mat2 = MATRIX_2.copy()

        self.mat1_d = np.asarray(MATRIX_1.A, order="C")[0, :].reshape(1, -1)
        self.mat2_d = np.asarray(MATRIX_2.A, order="C")


class TestSparseVector2DenseCMultiplication(TestSparseDenseMultiplication):

    order = "C"

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()[:, 0]

        self.mat1_d = np.asarray(MATRIX_1.A, order="C")
        self.mat2_d = np.asarray(MATRIX_2.A, order="C")[:, 0].reshape(-1, 1)


class TestSparseVectorDenseFMultiplication(TestSparseDenseMultiplication):

    order = "F"

    def setUp(self):
        self.mat1 = MATRIX_1.copy()[0, :]
        self.mat2 = MATRIX_2.copy()

        self.mat1_d = np.asarray(MATRIX_1.A, order="F")[0, :].reshape(1, -1)
        self.mat2_d = np.asarray(MATRIX_2.A, order="F")


class TestSparseVector2DenseFMultiplication(TestSparseDenseMultiplication):

    order = "F"

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()[:, 0]

        self.mat1_d = np.asarray(MATRIX_1.A, order="F")
        self.mat2_d = np.asarray(MATRIX_2.A, order="F")[:, 0].reshape(-1, 1)
