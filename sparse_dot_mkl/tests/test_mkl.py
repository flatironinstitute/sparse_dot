import os

os.environ["MKL_NUM_THREADS"] = "1"

# Make sure sklearn isn't breaking anything
try:
    import sklearn
except ImportError:
    pass

import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as _spsparse
from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl._mkl_interface import (_create_mkl_sparse, _export_mkl, sparse_matrix_t, _destroy_mkl_handle,
                                           _convert_to_csr, _order_mkl_handle, MKL)

SEED = 86


def make_matrixes(a, b, n, density):
    m1 = _spsparse.random(a, n, density=density, format="csr", dtype=np.float64, random_state=SEED)
    m2 = _spsparse.random(n, b, density=density, format="csr", dtype=np.float64, random_state=SEED + 1)
    return m1, m2


MATRIX_1, MATRIX_2 = make_matrixes(200, 100, 300, 0.05)
VECTOR = np.random.rand(300).astype(np.float64)
MATRIX_1_EMPTY = _spsparse.csr_matrix((200, 300), dtype=np.float64)


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
    unittest.main(module='sparse_dot_mkl.tests.test_gram_matrix')
    unittest.main(module='sparse_dot_mkl.tests.test_sparse_sparse')
    unittest.main(module='sparse_dot_mkl.tests.test_sparse_dense')
    unittest.main(module='sparse_dot_mkl.tests.test_dense_dense')
    unittest.main(module='sparse_dot_mkl.tests.test_qr_solver')
    unittest.main(module='sparse_dot_mkl.tests.test_sparse_vector')


if __name__ == '__main__':
    unittest.main()
