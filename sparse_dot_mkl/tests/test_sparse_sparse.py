import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as _spsparse
from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl._mkl_interface import _create_mkl_sparse, _export_mkl, sparse_matrix_t, set_debug_mode
from sparse_dot_mkl._sparse_sparse import _matmul_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1, MATRIX_2, make_matrixes


class TestMultiplicationCSR(unittest.TestCase):

    sparse_func = _spsparse.csr_matrix
    sparse_args = {}
    sparse_output = "csr"

    def setUp(self):
        self.mat1 = self.sparse_func(MATRIX_1, **self.sparse_args).copy()
        self.mat2 = self.sparse_func(MATRIX_2, **self.sparse_args).copy()

    def test_spmm_success(self):
        set_debug_mode(True)

        ref_1, precision_1 = _create_mkl_sparse(self.mat1)
        ref_2, precision_2 = _create_mkl_sparse(self.mat2)

        self.assertTrue(precision_1)
        self.assertTrue(precision_2)

        ref_3 = _matmul_mkl(ref_1, ref_2)
        mat3 = _export_mkl(ref_3, precision_1 or precision_2, output_type=self.sparse_output)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)

        set_debug_mode(False)

    def test_spmm_success_float32(self):
        self.mat1.data = self.mat1.data.astype(np.float32)
        self.mat2.data = self.mat2.data.astype(np.float32)

        ref_1, precision_1 = _create_mkl_sparse(self.mat1)
        ref_2, precision_2 = _create_mkl_sparse(self.mat2)

        self.assertFalse(precision_1)
        self.assertFalse(precision_2)

        ref_3 = _matmul_mkl(ref_1, ref_2)
        mat3 = _export_mkl(ref_3, precision_1 or precision_2, output_type=self.sparse_output)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)

    def test_spmm_error_bad_dims(self):
        ref_1, prec_1 = _create_mkl_sparse(self.mat1.transpose())
        ref_2, prec_2 = _create_mkl_sparse(self.mat2)

        with self.assertRaises(ValueError):
            _matmul_mkl(ref_1, ref_2)

    def test_spmm_error_bad_handle(self):
        with self.assertRaises(ValueError):
            _matmul_mkl(sparse_matrix_t(), sparse_matrix_t())

    def test_dot_product_mkl(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)

    def test_error_bad_dims(self):
        with self.assertRaises(ValueError):
            mat3 = dot_product_mkl(self.mat1.transpose(), self.mat2)

    def test_all_zeros(self):
        zero_mat_1 = self.sparse_func((50, 100))
        zero_mat_2 = self.sparse_func((100, 20))

        zm_sp = zero_mat_1.dot(zero_mat_2)
        zm_mkl = dot_product_mkl(zero_mat_1, zero_mat_2)

        self.assertTupleEqual(zm_sp.shape, zm_mkl.shape)
        self.assertEqual(len(zm_mkl.data), 0)

    def test_highly_sparse(self):
        hsp1, hsp2 = make_matrixes(2000, 1000, 3000, 0.0005)
        hsp1 = self.sparse_func(hsp1, **self.sparse_args)
        hsp2 = self.sparse_func(hsp2, **self.sparse_args)

        hsp3_sp = hsp1.dot(hsp2)
        hsp3 = dot_product_mkl(hsp1, hsp2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)

    def test_highly_highly_sparse(self):
        hsp1, hsp2 = make_matrixes(2000, 1000, 3000, 0.000005)
        hsp1 = self.sparse_func(hsp1, **self.sparse_args)
        hsp2 = self.sparse_func(hsp2, **self.sparse_args)

        hsp3_sp = hsp1.dot(hsp2)
        hsp3 = dot_product_mkl(hsp1, hsp2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)

    def test_dense(self):
        d1, d2 = make_matrixes(10, 20, 50, 1)
        d1 = self.sparse_func(d1, **self.sparse_args)
        d2 = self.sparse_func(d2, **self.sparse_args)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float64)

    def test_CSC(self):
        d1, d2 = self.mat1, _spsparse.csc_matrix(self.mat2)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float64)

    def test_CSR(self):
        d1, d2 = self.mat1, _spsparse.csc_matrix(self.mat2)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float64)

    @unittest.skip
    def test_BSR(self):
        d1, d2 = self.mat1, _spsparse.bsr_matrix(self.mat2)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2, debug=True)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float64)

    def test_COO(self):
        d1, d2 = self.mat1, _spsparse.coo_matrix(self.mat2)

        with self.assertRaises(ValueError):
            hsp3 = dot_product_mkl(d1, d2)

    def test_mixed(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2, cast=True)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float64)

    def test_mixed_2(self):
        d1, d2 = self.mat1, self.mat2.astype(np.float32)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2, cast=True)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float64)

    def test_mixed_nocast(self):
        d1, d2 = self.mat1, self.mat2.astype(np.float32)

        with self.assertRaises(ValueError):
            hsp3 = dot_product_mkl(d1, d2, cast=False)

    def test_float32(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2.astype(np.float32)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float32)

    def test_dot_product_mkl_copy(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2, copy=True)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)

    def test_dot_product_mkl_order(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2, reorder_output=True)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)


class TestMultiplicationCSC(TestMultiplicationCSR):
    sparse_func = _spsparse.csc_matrix
    sparse_args = {}
    sparse_output = "csc"


class TestMultiplicationBSR(TestMultiplicationCSR):

    sparse_func = _spsparse.bsr_matrix
    sparse_args = {"blocksize": (10, 10)}
    sparse_output = "bsr"

    @unittest.skip
    def test_CSC(self):
        pass

    @unittest.skip
    def test_CSR(self):
        pass


class TestSparseToDenseMultiplication(unittest.TestCase):

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()

    def test_float32_CSR(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2.astype(np.float32)
        mat3_np = np.dot(d1.A, d2.A)

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float32_CSC(self):
        d1, d2 = self.mat1.astype(np.float32).tocsc(), self.mat2.astype(np.float32).tocsc()
        mat3_np = np.dot(d1.A, d2.A)

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float64_CSR(self):
        d1, d2 = self.mat1, self.mat2
        mat3_np = np.dot(d1.A, d2.A)

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float64_CSC(self):
        d1, d2 = self.mat1.tocsc(), self.mat2.tocsc()
        mat3_np = np.dot(d1.A, d2.A)

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float64_BSR(self):
        d1, d2 = self.mat1.tobsr(blocksize=(10, 10)), self.mat2.tobsr(blocksize=(10, 10))
        mat3_np = np.dot(d1.A, d2.A)

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float32_BSR(self):
        d1 = self.mat1.astype(np.float32).tobsr(blocksize=(10, 10))
        d2 = self.mat2.astype(np.float32).tobsr(blocksize=(10, 10))
        mat3_np = np.dot(d1.A, d2.A)

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        npt.assert_array_almost_equal(mat3_np, mat3)
