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
                                           _convert_to_csr, _order_mkl_handle, MKL, _type_check)

SEED = 86

def make_matrixes(a, b, n, density, dtype=np.float64):
    m1 = _spsparse.random(a, n, density=density, format="csr", dtype=dtype, random_state=SEED)
    m2 = _spsparse.random(n, b, density=density, format="csr", dtype=dtype, random_state=SEED + 1)
    return m1, m2

def make_vector(n, complex=False):
    rng = np.random.default_rng(SEED + 2)
    if not complex:
        return rng.random(n).astype(np.float64)
    else:
        return rng.random(n) + rng.random(n) * 1j

MATRIX_1, MATRIX_2 = make_matrixes(200, 100, 300, 0.05)
MATRIX_1_EMPTY = _spsparse.csr_matrix((200, 300), dtype=np.float64)
VECTOR = make_vector(300)


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
        mkl_handle, dbl, cplx = _create_mkl_sparse(self.mat1)

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

    def test_bsr_not_square_blocks(self):
        with self.assertRaises(ValueError):
            _create_mkl_sparse(self.mat1.tobsr(blocksize=(10, 5)))


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
        npt.assert_array_almost_equal(sparse_1.A, sparse_2.A)

    def test_create_export(self):
        mat1 = _spsparse.csc_matrix(self.mat1).copy()
        mat2 = self.mat2.copy()
        mat3 = mat1.astype(np.float32).copy()
        mat4 = self.mat2.astype(np.float32).copy()

        ref_1, precision_1, cplx_1 = _create_mkl_sparse(mat1)
        ref_2, precision_2, cplx_2 = _create_mkl_sparse(mat2)
        ref_3, precision_3, cplx_3 = _create_mkl_sparse(mat3)
        ref_4, precision_4, cplx_4 = _create_mkl_sparse(mat4)

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

    def test_create_bsr(self):
        mat1 = _spsparse.bsr_matrix(self.mat1, blocksize=(2, 2))
        mat3 = mat1.astype(np.float32).copy()

        ref_1, precision_1, cplx_1 = _create_mkl_sparse(mat1)
        ref_3, precision_3, cplx_3 = _create_mkl_sparse(mat3)

        self.assertTrue(precision_1)
        self.assertFalse(precision_3)

        cycle_1 = _export_mkl(ref_1, precision_1, complex_type=cplx_1, output_type="bsr")
        cycle_3 = _export_mkl(ref_3, precision_3, complex_type=cplx_3, output_type="bsr")

        self.is_sparse_identical_A(self.mat1, cycle_1)
        self.is_sparse_identical_internal(mat1, cycle_1)
        self.is_sparse_identical_A(self.mat1.astype(np.float32), cycle_3)
        self.is_sparse_identical_internal(mat3, cycle_3)

        npt.assert_array_equal(mat1.data, cycle_1.data)
        npt.assert_array_equal(mat3.data, cycle_3.data)

    def test_create_convert_bsr(self):
        mat1 = _spsparse.bsr_matrix(self.mat1, blocksize=(2, 2))
        mat3 = mat1.astype(np.float32).copy()

        ref_1, precision_1, cplx_1 = _create_mkl_sparse(mat1)
        ref_3, precision_3, cplx_3 = _create_mkl_sparse(mat3)

        cref_1 = _convert_to_csr(ref_1)
        cref_3 = _convert_to_csr(ref_3)

        self.assertTrue(precision_1)
        self.assertFalse(precision_3)

        cycle_1 = _export_mkl(cref_1, precision_1, complex_type=cplx_1, output_type="csr")
        cycle_3 = _export_mkl(cref_3, precision_3, complex_type=cplx_3, output_type="csr")

        self.is_sparse_identical_A(self.mat1, cycle_1)
        self.is_sparse_identical_A(self.mat1.astype(np.float32), cycle_3)


class TestTypeConversions(unittest.TestCase):

    dtype = np.float32
    cast_dtype = np.float64

    final_dtype=None
    always_cast=False

    def setUp(self):
        self.mat1 = MATRIX_1.copy()
        self.mat2 = MATRIX_2.copy()

    def test_valid_pairs(self):

        a, b = self.mat1.astype(self.dtype), self.mat2.astype(self.dtype)
        c, d = _type_check(a, b, cast=self.always_cast)

        if self.always_cast:
            self.assertNotEqual(id(a), id(c))
            self.assertNotEqual(id(b), id(d))
        else:
            self.assertEqual(id(a), id(c))
            self.assertEqual(id(b), id(d))

        fd = self.final_dtype if self.final_dtype is not None else self.dtype

        self.assertEqual(c.dtype, fd)
        self.assertEqual(d.dtype, fd)

    def test_cast_pairs_right(self):

        a, b = self.mat1.astype(self.dtype), self.mat2.astype(self.cast_dtype)

        with self.assertRaises(ValueError):
            c, d = _type_check(a, b)

        c, d = _type_check(a, b, cast=True)

        self.assertNotEqual(id(a), id(c))

        if self.always_cast:
            self.assertNotEqual(id(b), id(d))
        else:
            self.assertEqual(id(b), id(d))

        fd = self.final_dtype if self.final_dtype is not None else self.cast_dtype

        self.assertEqual(c.dtype, fd)
        self.assertEqual(d.dtype, fd)

    def test_cast_pairs_left(self):

        a, b = self.mat1.astype(self.cast_dtype), self.mat2.astype(self.dtype)

        with self.assertRaises(ValueError):
            c, d = _type_check(a, b)

        c, d = _type_check(a, b, cast=True)

        if self.always_cast:
            self.assertNotEqual(id(a), id(c))
        else:
            self.assertEqual(id(a), id(c))
    
        self.assertNotEqual(id(b), id(d))

        fd = self.final_dtype if self.final_dtype is not None else self.cast_dtype

        self.assertEqual(c.dtype, fd)
        self.assertEqual(d.dtype, fd)


class TestTypeConversions2(TestTypeConversions):

    dtype = np.csingle
    cast_dtype = np.cdouble


class TestTypeConversions3(TestTypeConversions):

    dtype = np.float32
    cast_dtype = np.cdouble


class TestTypeConversions4(TestTypeConversions):

    dtype = np.float64
    cast_dtype = np.cdouble


class TestTypeConversions5(TestTypeConversions):

    dtype = np.int32
    cast_dtype = np.float32

    final_dtype = np.float64
    always_cast = True


class TestTypeConversions6(TestTypeConversions):

    dtype = np.int32
    cast_dtype = np.int64

    final_dtype = np.float64
    always_cast = True


class TestTypeConversions7(TestTypeConversions):

    dtype = np.clongdouble
    cast_dtype = np.clongdouble

    final_dtype = np.cdouble
    always_cast = True


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
