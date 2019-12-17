import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as _spsparse
from sparse_dot import sparse_matrix_t
from sparse_dot._common import _check_mkl_typing, _matmul_mkl
from sparse_dot.csr import _create_mkl_csr, _export_csr_mkl, csr_dot_product_mkl

SEED = 86


def make_matrixes(a, b, n, density):
    m1 = _spsparse.random(a, n, density=density, format="csr", dtype=np.float64, random_state=SEED)
    m2 = _spsparse.random(n, b, density=density, format="csr", dtype=np.float64, random_state=SEED + 1)
    return m1, m2


MATRIX_1, MATRIX_2 = make_matrixes(2000, 1000, 3000, 0.05)


class TestCSR(unittest.TestCase):

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()

    def test_typing(self):
        self.assertTrue(_check_mkl_typing(self.mat1, self.mat2))
        self.assertFalse(_check_mkl_typing(self.mat1.astype(np.float32),
                                           self.mat2.astype(np.float32)))

        with self.assertWarns(Warning, msg="Matrix dtypes are not identical. All data will be coerced to float64."):
            _check_mkl_typing(self.mat1, self.mat2.astype(np.float32))

        with self.assertWarns(Warning, msg="Matrix dtypes are not float32 or float64. All data will be coerced to float64."):
            _check_mkl_typing(self.mat1, self.mat2.astype(np.int32))

    def test_create_export(self):

        ref_1 = _create_mkl_csr(self.mat1)
        ref_2 = _create_mkl_csr(self.mat2)

        cycle_1 = _export_csr_mkl(ref_1)
        cycle_2 = _export_csr_mkl(ref_2)

        npt.assert_array_almost_equal(cycle_1.data, self.mat1.data)
        npt.assert_array_equal(cycle_1.indptr, self.mat1.indptr)
        npt.assert_array_equal(cycle_1.indices, self.mat1.indices)

        npt.assert_array_almost_equal(cycle_2.data, self.mat2.data)
        npt.assert_array_equal(cycle_2.indptr, self.mat2.indptr)
        npt.assert_array_equal(cycle_2.indices, self.mat2.indices)

    def test_spmm_success(self):

        ref_1 = _create_mkl_csr(self.mat1)
        ref_2 = _create_mkl_csr(self.mat2)

        ref_3 = _matmul_mkl(ref_1, ref_2)
        mat3 = _export_csr_mkl(ref_3)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)

    def test_spmm_error_bad_dims(self):

        ref_1 = _create_mkl_csr(self.mat1.transpose())
        ref_2 = _create_mkl_csr(self.mat2)

        with self.assertRaises(ValueError):
            _matmul_mkl(ref_1, ref_2)

    def test_spmm_error_bad_handle(self):

        with self.assertRaises(ValueError):
            _matmul_mkl(sparse_matrix_t(), sparse_matrix_t())

    def test_csr_dot_product_mkl(self):

        mat3 = csr_dot_product_mkl(self.mat1, self.mat2)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.A, self.mat2.A)

        npt.assert_array_almost_equal(mat3.A, mat3_sp.A)
        npt.assert_array_almost_equal(mat3_np, mat3.A)

    def test_csr_error_bad_dims(self):

        with self.assertRaises(ValueError):
            mat3 = csr_dot_product_mkl(self.mat1.transpose(), self.mat2)

    def test_csr_all_zeros(self):

        zero_mat_1 = _spsparse.csr_matrix((50,100))
        zero_mat_2 = _spsparse.csr_matrix((100,20))

        zm_sp = zero_mat_1.dot(zero_mat_2)
        zm_mkl = csr_dot_product_mkl(zero_mat_1, zero_mat_2)

        self.assertTupleEqual(zm_sp.shape, zm_mkl.shape)
        self.assertEqual(len(zm_mkl.data), 0)

    def test_not_csr(self):

        with self.assertRaises(ValueError):
            csr_dot_product_mkl(self.mat1.tocsc(), self.mat2)

    def test_highly_sparse(self):
        hsp1, hsp2 = make_matrixes(2000, 1000, 3000, 0.0005)
        hsp3_sp = hsp1.dot(hsp2)
        hsp3 = csr_dot_product_mkl(hsp1, hsp2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)

    def test_dense_CSR(self):
        d1, d2 = make_matrixes(10, 20, 50, 1)
        hsp3_sp = d1.dot(d2)
        hsp3 = csr_dot_product_mkl(d1, d2)

        npt.assert_array_almost_equal(hsp3.A, hsp3_sp.A)

