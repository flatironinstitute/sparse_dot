import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as _spsparse
from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl._mkl_interface import _create_mkl_sparse, _export_mkl, sparse_matrix_t
from sparse_dot_mkl._sparse_sparse import _matmul_mkl
from sparse_dot_mkl.tests.test_mkl import MATRIX_1, MATRIX_2, make_matrixes


class TestHandles(unittest.TestCase):

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()

    @staticmethod
    def is_sparse_identical_internal(sparse_1, sparse_2):
        npt.assert_array_almost_equal(sparse_1.data, sparse_2.data)
        npt.assert_array_equal(sparse_1.indptr, sparse_2.indptr)
        npt.assert_array_equal(sparse_1.indices, sparse_2.indices)

    def is_sparse_identical_A(self, sparse_1, sparse_2):
        self.assertEqual(np.sum((sparse_1 != sparse_2).data), 0)

    def test_create_export(self):
        mat1 = _spsparse.csc_matrix(self.mat1).copy()
        mat2 = self.mat2.copy()
        mat3 = mat1.astype(np.float32).copy()
        mat4 = self.mat2.astype(np.float32).copy()

        ref_1, precision_1 = _create_mkl_sparse(mat1)
        ref_2, precision_2 = _create_mkl_sparse(mat2)
        ref_3, precision_3 = _create_mkl_sparse(mat3)
        ref_4, precision_4 = _create_mkl_sparse(mat4)

        self.assertTrue(precision_1)
        self.assertTrue(precision_2)
        self.assertFalse(precision_3)
        self.assertFalse(precision_4)

        cycle_1 = _export_mkl(ref_1, precision_1, output_type="csc")
        cycle_2 = _export_mkl(ref_2, precision_2)
        cycle_3 = _export_mkl(ref_3, precision_3, output_type="csc")
        cycle_4 = _export_mkl(ref_4, precision_4)

        self.is_sparse_identical_A(self.mat1, cycle_1)
        self.is_sparse_identical_internal(self.mat2, cycle_2)
        self.is_sparse_identical_A(self.mat1.astype(np.float32), cycle_3)
        self.is_sparse_identical_internal(self.mat2.astype(np.float32), cycle_4)


class TestMultiplication(unittest.TestCase):

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()

    def test_spmm_success(self):
        ref_1, precision_1 = _create_mkl_sparse(self.mat1)
        ref_2, precision_2 = _create_mkl_sparse(self.mat2)

        self.assertTrue(precision_1)
        self.assertTrue(precision_2)

        ref_3 = _matmul_mkl(ref_1, ref_2)
        mat3 = _export_mkl(ref_3, precision_1 or precision_2)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)

    def test_spmm_success_float32(self):
        self.mat1.data = self.mat1.data.astype(np.float32)
        self.mat2.data = self.mat2.data.astype(np.float32)

        ref_1, precision_1 = _create_mkl_sparse(self.mat1)
        ref_2, precision_2 = _create_mkl_sparse(self.mat2)

        self.assertFalse(precision_1)
        self.assertFalse(precision_2)

        ref_3 = _matmul_mkl(ref_1, ref_2)
        mat3 = _export_mkl(ref_3, precision_1 or precision_2, output_type="csr")

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

    def test_csr_dot_product_mkl(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)

    def test_csr_error_bad_dims(self):
        with self.assertRaises(ValueError):
            mat3 = dot_product_mkl(self.mat1.transpose(), self.mat2)

    def test_csr_all_zeros(self):
        zero_mat_1 = _spsparse.csr_matrix((50, 100))
        zero_mat_2 = _spsparse.csr_matrix((100, 20))

        zm_sp = zero_mat_1.dot(zero_mat_2)
        zm_mkl = dot_product_mkl(zero_mat_1, zero_mat_2)

        self.assertTupleEqual(zm_sp.shape, zm_mkl.shape)
        self.assertEqual(len(zm_mkl.data), 0)

    def test_highly_sparse_CSR(self):
        hsp1, hsp2 = make_matrixes(2000, 1000, 3000, 0.0005)
        hsp3_sp = hsp1.dot(hsp2)
        hsp3 = dot_product_mkl(hsp1, hsp2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)

    def test_highly_highly_sparse_CSR(self):
        hsp1, hsp2 = make_matrixes(2000, 1000, 3000, 0.000005)
        hsp3_sp = hsp1.dot(hsp2)
        hsp3 = dot_product_mkl(hsp1, hsp2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)

    def test_dense_CSR(self):
        d1, d2 = make_matrixes(10, 20, 50, 1)
        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float64)

    def test_CSC(self):
        d1, d2 = _spsparse.csc_matrix(self.mat1), _spsparse.csc_matrix(self.mat2)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float64)

    def test_CSR_CSC(self):
        d1, d2 = self.mat1, _spsparse.csc_matrix(self.mat2)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float64)

    def test_CSC_CSR(self):
        d1, d2 = _spsparse.csc_matrix(self.mat1), self.mat2

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)
        self.assertTrue(hsp3.dtype == np.float64)

    def test_COO(self):
        d1, d2 = _spsparse.coo_matrix(self.mat1), self.mat2

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

    def test_csr_dot_product_mkl_copy(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2, copy=True)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)

    def test_csr_dot_product_mkl_order(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2, reorder_output=True)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)


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
