from scipy.sparse import (
    csr_array as _sps_csr_array,
    csr_matrix as _sps_csr_matrix,
    csc_array as _sps_csc_array,
    csc_matrix as _sps_csc_matrix,
    bsr_array as _sps_bsr_array,
    bsr_matrix as _sps_bsr_matrix
)

from sparse_dot_mkl import dot_product_mkl

class _mkl_matmul_mixin:

    dense_matmul = False
    cast_matmul = True

    def __matmul__(self, other):
        return dot_product_mkl(
            self,
            other,
            dense=self.dense_matmul,
            cast=self.cast_matmul
        )
    
    def __rmatmul__(self, other):
        return dot_product_mkl(
            other,
            self,
            dense=self.dense_matmul,
            cast=self.cast_matmul
        )

class csr_array(_mkl_matmul_mixin, _sps_csr_array):
    pass

class csr_matrix(_mkl_matmul_mixin, _sps_csr_matrix):
    pass

class csc_array(_mkl_matmul_mixin, _sps_csc_array):
    pass

class csc_matrix(_mkl_matmul_mixin, _sps_csc_matrix):
    pass

class bsr_array(_mkl_matmul_mixin, _sps_bsr_array):
    pass

class bsr_matrix(_mkl_matmul_mixin, _sps_bsr_matrix):
    pass
