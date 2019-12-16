import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as _spsparse
from sparse_dot import sparse_matrix_t
from sparse_dot._common import _check_mkl_typing, _matmul_mkl
from sparse_dot.csr import _create_mkl_csr, _export_csr_mkl, csr_dot_product_mkl

SEED = 86

MATRIX_1 = _spsparse.random(2000, 3000, density=0.05, format="csr", dtype=np.float64, random_state=SEED)
MATRIX_2 = _spsparse.random(3000, 1000, density=0.05, format="csr", dtype=np.float64, random_state=SEED + 1)


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

