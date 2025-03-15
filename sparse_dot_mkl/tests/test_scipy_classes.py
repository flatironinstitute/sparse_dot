import unittest
import numpy.testing as npt
import scipy as sp
import numpy as np
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

class TripError(RuntimeError):
    pass

def _tripwire(self, other):
    raise TripError("Shouldn't be here")

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

    def test_sum_0(self):
        
        a = self.arr(MATRIX_1)
        
        npt.assert_almost_equal(
            a.sum(axis=0),
            np.sum(a, axis=0)
        )

    def test_sum_1(self):
        
        a = self.arr(MATRIX_1)

        npt.assert_almost_equal(
            a.sum(axis=1),
            np.sum(a, axis=1)
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

        a = self.arr(MATRIX_1.copy())
        b = self.arr(MATRIX_2.copy())

        with self.assertRaises(ValueError):
            b @ a
        
        # Following tests dont work with old scipy
        if (
            (int(sp.__version__.split('.')[0]) > 1) or
            (int(sp.__version__.split('.')[1]) > 13)
        ):
            
            m1 = MATRIX_1.copy()
            m2 = MATRIX_2.copy()

            install_wire(m1)
            install_wire(m2)
            install_wire(a)
            install_wire(b)
            # SCIPY
            with self.assertRaises(TripError):
                m1 @ m2

            # SCIPY CSR_MATRIX USES RMATMUL DUNNO WHY
            if self.arr != csr_matrix:
                with self.assertRaises(TripError):
                    m1 @ b

            # MKL
            a @ m2 

            # MKL
            a @ b 


class TestCSRMat(TestCSR):
    arr = csr_matrix


class TestCSC(TestCSR):
    arr = csc_array


class TestCSCMat(TestCSR):
    arr = csc_matrix


class TestBSRMat(TestCSR):
    arr = bsr_matrix


class TestBSR(TestCSR):
    arr = bsr_array

