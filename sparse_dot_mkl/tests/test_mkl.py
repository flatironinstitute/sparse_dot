import os

os.environ["MKL_NUM_THREADS"] = "1"

import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as _spsparse
from sparse_dot_mkl import dot_product_mkl, gram_matrix_mkl, sparse_qr_solve_mkl
from sparse_dot_mkl._mkl_interface import (_create_mkl_sparse, _export_mkl, sparse_matrix_t, _destroy_mkl_handle,
                                           _convert_to_csr, _order_mkl_handle, MKL)
from sparse_dot_mkl._sparse_sparse import _matmul_mkl

SEED = 86


def make_matrixes(a, b, n, density):
    m1 = _spsparse.random(a, n, density=density, format="csr", dtype=np.float64, random_state=SEED)
    m2 = _spsparse.random(n, b, density=density, format="csr", dtype=np.float64, random_state=SEED + 1)
    return m1, m2


MATRIX_1, MATRIX_2 = make_matrixes(200, 100, 300, 0.05)
VECTOR = np.random.rand(300).astype(np.float64)
MATRIX_1_EMPTY = _spsparse.csr_matrix((200, 300), dtype=np.float64)


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


class TestDenseDenseMultiplication(unittest.TestCase):

    def setUp(self):
        self.mat1 = MATRIX_1.copy().A
        self.mat2 = MATRIX_2.copy().A

    def test_float32(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2.astype(np.float32)
        mat3_np = np.dot(d1, d2)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float64(self):
        d1, d2 = self.mat1, self.mat2
        mat3_np = np.dot(d1, d2)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float64_cast(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2
        mat3_np = np.dot(d1, d2)

        mat3 = dot_product_mkl(d1, d2, cast=True)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_outer_product(self):
        d1, d2 = self.mat1[:, 0].reshape(-1, 1).copy(), self.mat2[0, :].reshape(1, -1).copy()

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)


class TestDenseDenseFCMultiplication(TestDenseDenseMultiplication):

    def setUp(self):
        self.mat1 = np.asarray(MATRIX_1.copy().A, order='F')
        self.mat2 = np.asarray(MATRIX_2.copy().A, order='C')


class TestDenseDenseFFMultiplication(TestDenseDenseMultiplication):

    def setUp(self):
        self.mat1 = np.asarray(MATRIX_1.copy().A, order='F')
        self.mat2 = np.asarray(MATRIX_2.copy().A, order='F')


class TestDenseDenseCFMultiplication(TestDenseDenseMultiplication):

    def setUp(self):
        self.mat1 = np.asarray(MATRIX_1.copy().A, order='C')
        self.mat2 = np.asarray(MATRIX_2.copy().A, order='F')


class TestSparseDenseMultiplication(unittest.TestCase):

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

    def test_float64_b_sparse(self):
        d1, d2 = self.mat1_d, self.mat2

        mat3 = dot_product_mkl(d1, d2, debug=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float64_cast_b_sparse(self):
        d1, d2 = self.mat1_d.astype(np.float32), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float32_csc_sparse(self):
        d1, d2 = self.mat1_d.astype(np.float32), self.mat2.astype(np.float32).tocsc()
        mat3_np = np.dot(d1, d2.A)

        mat3 = dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

    def test_float64_csc_sparse(self):
        d1, d2 = self.mat1_d, self.mat2.tocsc()

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

    def test_float64_cast_csc_sparse(self):
        d1, d2 = self.mat1_d.astype(np.float32), self.mat2.tocsc()

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d2.A, self.mat2_d)

    def test_float32_a_sparse(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2_d.astype(np.float32)

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float64_a_sparse(self):
        d1, d2 = self.mat1, self.mat2_d

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_float64_a_csc_sparse(self):
        d1, d2 = self.mat1.tocsc(), self.mat2_d

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d1.A, self.mat1_d)

    def test_float32_a_csc_sparse(self):
        d1, d2 = self.mat1.astype(np.float32).tocsc(), self.mat2_d.astype(np.float32)

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)
        npt.assert_array_almost_equal(d1.A, self.mat1_d)

    def test_float64_cast_a_sparse(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2_d

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)


class TestSparseDenseFMultiplication(TestSparseDenseMultiplication):

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()

        self.mat1_d = np.asarray(MATRIX_1.A, order="F")
        self.mat2_d = np.asarray(MATRIX_2.A, order="F")


class TestSparseVectorMultiplication(unittest.TestCase):

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = VECTOR.copy()

    def test_mult_1d(self):
        d1, d2 = self.mat1.astype(np.float64), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_1d_float32(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_1d_both_float32(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2.astype(np.float32)

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d(self):
        d1, d2 = self.mat1.astype(np.float64), self.mat2.reshape(-1, 1)

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d_float32(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2.reshape(-1, 1)

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d_both_float32(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2.reshape(-1, 1).astype(np.float32)

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)


class TestVectorSparseMultiplication(unittest.TestCase):

    def setUp(self):
        self.mat1 = VECTOR.copy()
        self.mat2 = MATRIX_2.copy()

        self.mat2_d = np.asarray(MATRIX_2.A, order="C")

    def test_mult_1d(self):
        d1, d2 = self.mat1.astype(np.float64), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_1d_float32(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_1d_both_float32(self):
        d1, d2 = self.mat1.astype(np.float32), self.mat2.astype(np.float32)

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d(self):
        d1, d2 = self.mat1.astype(np.float64).reshape(1, -1), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d_float32(self):
        d1, d2 = self.mat1.astype(np.float32).reshape(1, -1), self.mat2

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_2d_both_float32(self):
        d1, d2 = self.mat1.astype(np.float32).reshape(1, -1), self.mat2.astype(np.float32)

        mat3 = dot_product_mkl(d1, d2, cast=True)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_outer_product_ds(self):
        d1, d2 = self.mat1.reshape(-1, 1), _spsparse.csr_matrix(self.mat2_d[:, 0].reshape(1, -1))

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1, d2.A)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_mult_outer_product_sd(self):
        d1, d2 = _spsparse.csr_matrix(self.mat1.reshape(-1, 1)), self.mat2_d[:, 0].reshape(1, -1).copy()

        mat3 = dot_product_mkl(d1, d2)
        mat3_np = np.dot(d1.A, d2)

        npt.assert_array_almost_equal(mat3_np, mat3)


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


class TestEmptyConditions(unittest.TestCase):

    def setUp(self):
        self.mat1 = MATRIX_1_EMPTY.copy()
        self.mat2 = MATRIX_2.copy()

        self.mat1_d = np.asarray(MATRIX_1_EMPTY.A, order="C")
        self.mat2_d = np.asarray(MATRIX_2.A, order="C")

        self.mat1_zero = np.zeros((0, 300))

    def test_sparse_sparse(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2)
        mat3_np = np.dot(self.mat1_d, self.mat2_d)

        npt.assert_array_almost_equal(mat3_np, mat3.A)

    def test_sparse_dense(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2_d)
        mat3_np = np.dot(self.mat1_d, self.mat2_d)

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_sparse_vector(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2_d[:, 0])
        mat3_np = np.dot(self.mat1_d, self.mat2_d[:, 0])

        npt.assert_array_almost_equal(mat3_np, mat3)

    def test_dense_dense(self):
        mat3 = dot_product_mkl(self.mat1_zero, self.mat2_d)
        mat3_np = np.dot(self.mat1_zero, self.mat2_d)

        npt.assert_array_almost_equal(mat3_np, mat3)


class TestGramMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        gram_ut = np.dot(MATRIX_1.A.T, MATRIX_1.A)
        gram_ut[np.tril_indices(gram_ut.shape[0], k=-1)] = 0.
        cls.gram_ut = gram_ut

        gram_ut_t = np.dot(MATRIX_1.A, MATRIX_1.A.T)
        gram_ut_t[np.tril_indices(gram_ut_t.shape[0], k=-1)] = 0.
        cls.gram_ut_t = gram_ut_t

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat1_d = MATRIX_1.A

    def test_gram_matrix_sp(self):
        mat2 = gram_matrix_mkl(self.mat1)
        npt.assert_array_almost_equal(mat2.A, self.gram_ut)

    def test_gram_matrix_sp_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(np.float32))
        npt.assert_array_almost_equal(mat2.A, self.gram_ut)

    def test_gram_matrix_d_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(np.float32), dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_d(self):
        mat2 = gram_matrix_mkl(self.mat1, dense=True)
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
        mat2 = gram_matrix_mkl(self.mat1.tocsc(), dense=True, cast=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_double(self):
        mat2 = gram_matrix_mkl(self.mat1.A, dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_single(self):
        mat2 = gram_matrix_mkl(self.mat1.astype(np.float32).A, dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_double_F(self):
        mat2 = gram_matrix_mkl(np.asarray(self.mat1.A, order="F"), dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)

    def test_gram_matrix_dd_single_F(self):
        mat2 = gram_matrix_mkl(np.asarray(self.mat1.astype(np.float32).A, order="F"), dense=True)
        npt.assert_array_almost_equal(mat2, self.gram_ut)


class TestSparseSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.A = _spsparse.diags((MATRIX_1.data[0:100].copy()), format="csr")
        cls.B = MATRIX_1.data[0:100].copy().reshape(-1, 1)
        cls.X = np.linalg.lstsq(cls.A.A, cls.B, rcond=None)[0]

    def setUp(self):
        self.mat1 = self.A.copy()
        self.mat2 = self.B.copy()
        self.mat3 = self.X.copy()

    def test_sparse_solver(self):
        mat3 = sparse_qr_solve_mkl(self.mat1, self.mat2, debug=True)
        npt.assert_array_almost_equal(self.mat3, mat3)

    def test_sparse_solver_single(self):
        mat3 = sparse_qr_solve_mkl(self.mat1.astype(np.float32), self.mat2.astype(np.float32))
        npt.assert_array_almost_equal(self.mat3, mat3)

    def test_sparse_solver_cast_B(self):
        mat3 = sparse_qr_solve_mkl(self.mat1, self.mat2.astype(np.float32), cast=True)
        npt.assert_array_almost_equal(self.mat3, mat3)

    def test_sparse_solver_cast_A(self):
        mat3 = sparse_qr_solve_mkl(self.mat1.astype(np.float32), self.mat2, cast=True)
        npt.assert_array_almost_equal(self.mat3, mat3)

    def test_sparse_solver_cast_CSC(self):
        mat3 = sparse_qr_solve_mkl(self.mat1.tocsc(), self.mat2, cast=True)
        npt.assert_array_almost_equal(self.mat3, mat3)

    def test_sparse_solver_1d_d(self):
        mat3 = sparse_qr_solve_mkl(self.mat1, self.mat2.ravel())
        npt.assert_array_almost_equal(self.mat3.ravel(), mat3)

    def test_solver_guard_errors(self):
        with self.assertRaises(ValueError):
            mat3 = sparse_qr_solve_mkl(self.mat1, self.mat2.T)

        with self.assertRaises(ValueError):
            mat3 = sparse_qr_solve_mkl(self.mat1.tocsc(), self.mat2)

        with self.assertRaises(ValueError):
            mat3 = sparse_qr_solve_mkl(self.mat1.tocoo(), self.mat2, cast=True)


class TestFailureConditions(unittest.TestCase):

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()
        self.vec = VECTOR.copy()

    def test_make_mkl_bad_type(self):
        with self.assertRaises(ValueError):
            _create_mkl_sparse(self.mat1.tocoo())

        with self.assertRaises(ValueError):
            _create_mkl_sparse(self.mat1.astype(np.int64))

    def test_export_mkl_bad_type(self):
        mkl_handle, dbl = _create_mkl_sparse(self.mat1)

        with self.assertRaises(ValueError):
            _export_mkl(mkl_handle, dbl, output_type="coo")

        _destroy_mkl_handle(mkl_handle)

    def test_empty_handle(self):
        mkl_handle_empty = sparse_matrix_t()

        with self.assertRaises(ValueError):
            _export_mkl(mkl_handle_empty, True, output_type="csr")

        with self.assertRaises(ValueError):
            _convert_to_csr(mkl_handle_empty)

        with self.assertRaises(ValueError):
            _order_mkl_handle(mkl_handle_empty)

        with self.assertRaises(ValueError):
            _destroy_mkl_handle(mkl_handle_empty)

    def test_3d_matrixes(self):
        d1, d2 = self.mat1.A.reshape(200, 300, 1), self.mat2.A.reshape(300, 100, 1)

        with self.assertRaises(ValueError):
            dot_product_mkl(d1, d2)

        with self.assertRaises(ValueError):
            dot_product_mkl(d1, self.mat2)

        with self.assertRaises(ValueError):
            dot_product_mkl(self.mat1, d2)

    def test_bad_shapes(self):
        with self.assertRaises(ValueError):
            dot_product_mkl(self.vec.reshape(-1, 1), self.mat2)

        with self.assertRaises(ValueError):
            dot_product_mkl(self.mat1, self.vec.reshape(1, -1))

        with self.assertRaises(ValueError):
            dot_product_mkl(self.vec.reshape(-1, 1), self.vec.reshape(-1, 1))

        with self.assertRaises(ValueError):
            dot_product_mkl(self.mat1.transpose(), self.mat2)

        with self.assertRaises(ValueError):
            dot_product_mkl(self.vec[100:], self.vec)

    def test_lets_be_honest_this_is_just_to_make_codecov_bigger(self):
        with self.assertRaises(NotImplementedError):
            MKL()


def run():
    unittest.main(module='sparse_dot_mkl.tests.test_mkl')


if __name__ == '__main__':
    unittest.main()
