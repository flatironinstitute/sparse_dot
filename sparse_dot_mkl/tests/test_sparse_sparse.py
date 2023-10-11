import unittest
import numpy as np
import scipy.sparse as _spsparse
from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl._mkl_interface import (
    _create_mkl_sparse,
    _export_mkl,
    sparse_matrix_t,
    set_debug_mode,
)
from sparse_dot_mkl._sparse_sparse import _matmul_mkl
from sparse_dot_mkl.tests.test_mkl import (
    MATRIX_1,
    MATRIX_2,
    make_matrixes,
    np_almost_equal
)


class TestMultiplicationCSR(unittest.TestCase):
    sparse_func = _spsparse.csr_matrix
    sparse_args = {}
    sparse_output = "csr_matrix"

    double_dtype = np.float64
    single_dtype = np.float32

    export_complex = False

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, cls.MATRIX_2 = MATRIX_1.copy(), MATRIX_2.copy()

    def setUp(self):
        self.mat1 = self.sparse_func(self.MATRIX_1, **self.sparse_args).copy()
        self.mat2 = self.sparse_func(self.MATRIX_2, **self.sparse_args).copy()

    def test_spmm_success(self):
        set_debug_mode(True)

        ref_1, precision_1, cplx_1 = _create_mkl_sparse(self.mat1)
        ref_2, precision_2, cplx_2 = _create_mkl_sparse(self.mat2)

        self.assertTrue(precision_1)
        self.assertTrue(precision_2)

        ref_3 = _matmul_mkl(ref_1, ref_2)
        mat3 = _export_mkl(
            ref_3,
            precision_1 or precision_2,
            complex_type=self.export_complex,
            output_type=self.sparse_output,
        )

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.toarray(), self.mat2.toarray())

        np_almost_equal(mat3, mat3_sp)
        np_almost_equal(mat3_np, mat3)

        set_debug_mode(False)

    def test_spmm_success_float32(self):
        self.mat1.data = self.mat1.data.astype(self.single_dtype)
        self.mat2.data = self.mat2.data.astype(self.single_dtype)

        ref_1, precision_1, cplx_1 = _create_mkl_sparse(self.mat1)
        ref_2, precision_2, cplx_2 = _create_mkl_sparse(self.mat2)

        self.assertFalse(precision_1)
        self.assertFalse(precision_2)

        ref_3 = _matmul_mkl(ref_1, ref_2)
        mat3 = _export_mkl(
            ref_3,
            precision_1 or precision_2,
            complex_type=self.export_complex,
            output_type=self.sparse_output,
        )

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.toarray(), self.mat2.toarray())

        np_almost_equal(mat3, mat3_sp)
        np_almost_equal(mat3_np, mat3)

    def test_spmm_error_bad_dims(self):
        ref_1, prec_1, cplx_1 = _create_mkl_sparse(self.mat1.transpose())
        ref_2, prec_2, cplx_2 = _create_mkl_sparse(self.mat2)

        with self.assertRaises(ValueError):
            _matmul_mkl(ref_1, ref_2)

    def test_spmm_error_bad_handle(self):
        with self.assertRaises(ValueError):
            _matmul_mkl(sparse_matrix_t(), sparse_matrix_t())

    def test_dot_product_mkl(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.toarray(), self.mat2.toarray())

        np_almost_equal(mat3, mat3_sp)
        np_almost_equal(mat3_np, mat3)

    def test_error_bad_dims(self):
        with self.assertRaises(ValueError):
            _ = dot_product_mkl(self.mat1.transpose(), self.mat2)

    def test_all_zeros(self):
        zero_mat_1 = self.sparse_func((50, 100))
        zero_mat_2 = self.sparse_func((100, 20))

        zm_sp = zero_mat_1.dot(zero_mat_2)
        zm_mkl = dot_product_mkl(zero_mat_1, zero_mat_2)

        self.assertTupleEqual(zm_sp.shape, zm_mkl.shape)
        self.assertEqual(len(zm_mkl.data), 0)

    def test_highly_sparse(self):
        hsp1, hsp2 = make_matrixes(
            2000, 1000, 3000, 0.0005, dtype=self.double_dtype
        )
        hsp1 = self.sparse_func(hsp1, **self.sparse_args)
        hsp2 = self.sparse_func(hsp2, **self.sparse_args)

        hsp3_sp = hsp1.dot(hsp2)
        hsp3 = dot_product_mkl(hsp1, hsp2)

        np_almost_equal(hsp3, hsp3_sp)

    def test_highly_highly_sparse(self):
        hsp1, hsp2 = make_matrixes(
            2000, 1000, 3000, 0.000005, dtype=self.double_dtype
        )
        hsp1 = self.sparse_func(hsp1, **self.sparse_args)
        hsp2 = self.sparse_func(hsp2, **self.sparse_args)

        hsp3_sp = hsp1.dot(hsp2)
        hsp3 = dot_product_mkl(hsp1, hsp2)

        np_almost_equal(hsp3, hsp3_sp)

    def test_dense(self):
        d1, d2 = make_matrixes(10, 20, 50, 1, dtype=self.double_dtype)
        d1 = self.sparse_func(d1, **self.sparse_args)
        d2 = self.sparse_func(d2, **self.sparse_args)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        np_almost_equal(hsp3, hsp3_sp)
        self.assertTrue(hsp3.dtype == self.double_dtype)

    def test_CSC(self):
        d1, d2 = self.mat1, _spsparse.csc_matrix(self.mat2)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        np_almost_equal(hsp3, hsp3_sp)
        self.assertTrue(hsp3.dtype == self.double_dtype)

    def test_CSR(self):
        d1, d2 = self.mat1, _spsparse.csc_matrix(self.mat2)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        np_almost_equal(hsp3, hsp3_sp)
        self.assertTrue(hsp3.dtype == self.double_dtype)

    @unittest.skip
    def test_BSR(self):
        d1, d2 = self.mat1, _spsparse.bsr_matrix(self.mat2)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2, debug=True)

        np_almost_equal(hsp3, hsp3_sp)
        self.assertTrue(hsp3.dtype == self.double_dtype)

    def test_COO(self):
        d1, d2 = self.mat1, _spsparse.coo_matrix(self.mat2)

        with self.assertRaises(ValueError):
            _ = dot_product_mkl(d1, d2)

    def test_mixed(self):
        d1, d2 = self.mat1.astype(self.single_dtype), self.mat2

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2, cast=True)

        np_almost_equal(hsp3, hsp3_sp)
        self.assertTrue(hsp3.dtype == self.double_dtype)

    def test_mixed_2(self):
        d1, d2 = self.mat1, self.mat2.astype(self.single_dtype)

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2, cast=True)

        np_almost_equal(hsp3, hsp3_sp)
        self.assertTrue(hsp3.dtype == self.double_dtype)

    def test_mixed_nocast(self):
        d1, d2 = self.mat1, self.mat2.astype(self.single_dtype)

        with self.assertRaises(ValueError):
            _ = dot_product_mkl(d1, d2, cast=False)

    def test_float32(self):
        d1, d2 = self.mat1.astype(self.single_dtype), self.mat2.astype(
            self.single_dtype
        )

        hsp3_sp = d1.dot(d2)
        hsp3 = dot_product_mkl(d1, d2)

        np_almost_equal(hsp3, hsp3_sp)
        self.assertTrue(hsp3.dtype == self.single_dtype)

    def test_dot_product_mkl_copy(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2, copy=True)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.toarray(), self.mat2.toarray())

        np_almost_equal(mat3, mat3_sp)
        np_almost_equal(mat3_np, mat3)

    def test_dot_product_mkl_order(self):
        mat3 = dot_product_mkl(self.mat1, self.mat2, reorder_output=True)

        mat3_sp = self.mat1.dot(self.mat2)
        mat3_np = np.dot(self.mat1.toarray(), self.mat2.toarray())

        np_almost_equal(mat3, mat3_sp)
        np_almost_equal(mat3_np, mat3)


class TestMultiplicationCSC(TestMultiplicationCSR):
    sparse_func = _spsparse.csc_matrix
    sparse_args = {}
    sparse_output = "csc_matrix"


class TestMultiplicationBSR(TestMultiplicationCSR):
    sparse_func = _spsparse.bsr_matrix
    sparse_args = {"blocksize": (10, 10)}
    sparse_output = "bsr_matrix"

    @unittest.skip
    def test_CSC(self):
        pass

    @unittest.skip
    def test_CSR(self):
        pass


class TestSparseToDenseMultiplication(unittest.TestCase):
    double_dtype = np.float64
    single_dtype = np.float32

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, cls.MATRIX_2 = MATRIX_1.copy(), MATRIX_2.copy()

    def setUp(self):
        self.mat1 = self.MATRIX_1.copy()
        self.mat2 = self.MATRIX_2.copy()

    def test_float32_CSR(self):
        d1, d2 = self.mat1.astype(self.single_dtype), self.mat2.astype(
            self.single_dtype
        )
        mat3_np = np.dot(d1.toarray(), d2.toarray())

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        np_almost_equal(mat3_np, mat3)

    def test_float32_CSC(self):
        d1, d2 = (
            self.mat1.astype(self.single_dtype).tocsc(),
            self.mat2.astype(self.single_dtype).tocsc(),
        )
        mat3_np = np.dot(d1.toarray(), d2.toarray())

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        np_almost_equal(mat3_np, mat3)

    def test_float64_CSR(self):
        d1, d2 = self.mat1, self.mat2
        mat3_np = np.dot(d1.toarray(), d2.toarray())

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        np_almost_equal(mat3_np, mat3)

    def test_float64_CSC(self):
        d1, d2 = self.mat1.tocsc(), self.mat2.tocsc()
        mat3_np = np.dot(d1.toarray(), d2.toarray())

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        np_almost_equal(mat3_np, mat3)

    def test_float64_BSR(self):
        d1, d2 = self.mat1.tobsr(blocksize=(10, 10)), self.mat2.tobsr(
            blocksize=(10, 10)
        )
        mat3_np = np.dot(d1.toarray(), d2.toarray())

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        np_almost_equal(mat3_np, mat3)

    def test_float32_BSR(self):
        d1 = self.mat1.astype(self.single_dtype).tobsr(blocksize=(10, 10))
        d2 = self.mat2.astype(self.single_dtype).tobsr(blocksize=(10, 10))
        mat3_np = np.dot(d1.toarray(), d2.toarray())

        mat3 = dot_product_mkl(d1, d2, copy=True, dense=True)

        np_almost_equal(mat3_np, mat3)


class _ComplexMixin:
    double_dtype = np.cdouble
    single_dtype = np.csingle
    export_complex = True

    @classmethod
    def setUpClass(cls):
        cls.MATRIX_1, cls.MATRIX_2 = make_matrixes(
            200, 100, 300, 0.05, dtype=np.cdouble
        )


class TestMultiplicationCSRComplex(_ComplexMixin, TestMultiplicationCSR):
    pass


class TestMultiplicationCSCComplex(_ComplexMixin, TestMultiplicationCSC):
    pass


class TestMultiplicationBSRComplex(_ComplexMixin, TestMultiplicationBSR):
    pass


class TestSparseToDenseMultiplicationComplex(
    _ComplexMixin, TestSparseToDenseMultiplication
):
    pass


try:
    from scipy.sparse import (
        csr_array,
        csc_array,
        bsr_array
    )

    class TestMultiplicationCSRArray(TestMultiplicationCSR):

        sparse_func = csr_array
        sparse_args = {}
        sparse_output = "csr_array"

    class TestMultiplicationCSCArray(TestMultiplicationCSC):

        sparse_func = csc_array
        sparse_args = {}
        sparse_output = "csc_array"

    class TestMultiplicationBSRArray(TestMultiplicationBSR):

        sparse_func = bsr_array
        sparse_args = {}
        sparse_output = "bsr_array"

    class TestSparseToDenseMultiplicationArray(
        TestSparseToDenseMultiplication
    ):

        def setUp(self):
            self.mat1 = csr_array(self.MATRIX_1.copy())
            self.mat2 = csr_array(self.MATRIX_2.copy())

    class TestMultiplicationCSRArrayComplex(
        _ComplexMixin,
        TestMultiplicationCSRArray
    ):
        pass

    class TestMultiplicationCSCArrayComplex(
        _ComplexMixin,
        TestMultiplicationCSCArray
    ):
        pass

    class TestMultiplicationBSRArrayComplex(
        _ComplexMixin,
        TestMultiplicationBSRArray
    ):
        pass

    class TestSparseToDenseMultiplicationArrayComplex(
        _ComplexMixin,
        TestSparseToDenseMultiplicationArray
    ):
        pass


except ImportError:
    pass
