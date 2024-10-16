import unittest
import numpy as np
import numpy.testing as npt
from scipy.sparse import csr_matrix
from sparse_dot_mkl._mkl_interface import (
    SPARSE_FILL_MODE_UPPER,
    SPARSE_DIAG_NON_UNIT,
    SPARSE_MATRIX_TYPE_SYMMETRIC
)
from sparse_dot_mkl.solvers import (
    CGIterativeSparseSolver,
    FGMRESIterativeSparseSolver,
    fgmres,
    cg
)
from sparse_dot_mkl.tests.test_mkl import MATRIX_1

test_rhs = np.zeros(8, dtype=float)

test_matrix_indptr = np.array([0, 1, 5, 8, 10, 12, 15, 17, 18], dtype=int)
test_matrix_index = np.array(
  [ 1,    3,       6, 7,
       2, 3,    5,
          3,             8,
             4,       7,
                5, 6, 7,
                   6,    8,
                      7,
                         8], dtype=int)
test_matrix_data = np.array( 
  [ 7E0,       1E0,             2.E0, 7.E0,
         -4E0, 8E0,       2E0,
                1E0,                         5E0,
                      7E0,             9E0,
                            5E0, 1E0, 5E0,
                                 -1E0,       5E0,
                                       11E0,
                                              5E0], dtype=float)

                                              
test_matrix = csr_matrix((test_matrix_data, test_matrix_index, test_matrix_indptr))
test_expected_solution = np.array([1e0, 0e0, 1e0, 0e0, 1e0, 0e0, 1e0, 0e0, 0e0], dtype=float)

class TestSparseSolverCG(unittest.TestCase):

    def setUp(self):
        self.mat1 = test_matrix.copy()
        self.mat2 = test_rhs.copy()
        self.mat3 = test_expected_solution.copy()

    def test_cg_solver_square_perfect(self):
        mat3 = np.linalg.lstsq(self.mat1.toarray(), test_rhs, rcond=None)[0]

        with CGIterativeSparseSolver(self.mat1, self.mat2, x=self.mat3, verbose=False) as cg_solver:
            cg_solver.set_sparse_matrix_descr(SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT)
            x = cg_solver.solve()

        npt.assert_array_equal(test_matrix.toarray(), self.mat1.toarray())
        npt.assert_array_equal(test_rhs, self.mat2)
        npt.assert_array_almost_equal(x, mat3)

    def test_cg_wrapper_square_perfect(self):

        mat3 = np.linalg.lstsq(self.mat1.toarray(), test_rhs, rcond=None)[0]

        x, _code = cg(self.mat1, self.mat2)

        self.assertEqual(_code, 0)

        npt.assert_array_equal(test_matrix.toarray(), self.mat1.toarray())
        npt.assert_array_equal(test_rhs, self.mat2)
        npt.assert_array_almost_equal(x, mat3)

class TestSparseSolverFGMRES(unittest.TestCase):

    def setUp(self):
        self.mat1 = test_matrix.copy()
        self.mat2 = test_rhs.copy()
        self.mat3 = test_expected_solution.copy()

    def test_fgmres_solver_square_perfect(self):
        mat3 = np.linalg.lstsq(self.mat1.toarray(), test_rhs, rcond=None)[0]

        with FGMRESIterativeSparseSolver(self.mat1, self.mat2, x=self.mat3, verbose=False) as fgmres_solver:
            fgmres_solver.set_sparse_matrix_descr(SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT)
            x = fgmres_solver.solve()

        npt.assert_array_equal(test_matrix.toarray(), self.mat1.toarray())
        npt.assert_array_equal(test_rhs, self.mat2)
        npt.assert_array_almost_equal(x, mat3)

    def test_fgmres_wrapper_square_perfect(self):

        mat3 = np.linalg.lstsq(self.mat1.toarray(), test_rhs, rcond=None)[0]

        x, _code = fgmres(self.mat1, self.mat2)

        self.assertEqual(_code, 0)

        npt.assert_array_equal(test_matrix.toarray(), self.mat1.toarray())
        npt.assert_array_equal(test_rhs, self.mat2)
        npt.assert_array_almost_equal(x, mat3)