import unittest
import numpy.testing as npt
import scipy.sparse as sps
from types import MethodType

from sparse_dot_mkl import dot_product_mkl
from sparse_dot_mkl.sparse_array import (
    csr_array,
    csc_array,
    bsr_array,
    csc_matrix,
    csr_matrix,
    bsr_matrix
)
from sparse_dot_mkl.tests.test_mkl import MATRIX_1, MATRIX_2, make_matrixes

MATMUL = MATRIX_1 @ MATRIX_2
MATMUL = MATMUL.toarray()

def _tripwire(self, other):
    raise RuntimeError("Shouldn't be here")

def install_wire(x):
    x._matmul_dispatch = MethodType(_tripwire, x)
    x._rmatmul_dispatch = MethodType(_tripwire, x)


class TestCSR(unittest.TestCase):

    arr = csr_array

    def test_matmul(self):

        a = self.arr(MATRIX_1)
        b = self.arr(MATRIX_2)

        install_wire(a)
        install_wire(b)

        c = a @ b

        npt.assert_almost_equal(
            c.toarray(),
            MATMUL
        )

    def test_matmul_dense(self):

        a = self.arr(MATRIX_1)
        b = self.arr(MATRIX_2)

        install_wire(a)
        install_wire(b)

        a.dense_matmul = True
        a.dense_matmul = True

        c = a @ b

        self.assertFalse(
            sps.issparse(c)
        )

        npt.assert_almost_equal(
            c,
            MATMUL
        )

    def test_matmul_fail(self):

        a = self.arr(MATRIX_1)
        b = self.arr(MATRIX_2)

        with self.assertRaises(ValueError):
            b @ a
        
        m1 = MATRIX_1.copy()
        install_wire(m1)

        with self.assertRaises(RuntimeError):
            m1 @ MATRIX_2


class TestCSRMat(TestCSR):
    arr = csr_matrix


class TestCSC(TestCSR):
    arr = csc_array


class TestCSCMat(TestCSR):
    arr = csc_matrix


class TestBSRMat(TestCSR):
    arr = bsr_matrix


class TestBSC(TestCSR):
    arr = bsr_array

