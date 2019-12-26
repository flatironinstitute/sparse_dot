from sparse_dot import _create_mkl_csr, _export_csr_mkl, _matmul_mkl, _destroy_mkl_handle, _order_mkl_handle
from sparse_dot._common import _check_mkl_typing, _check_alignment
import scipy.sparse as _spsparse


def csr_dot_product_mkl(csr_matrix_a, csr_matrix_b, copy=False, reorder_output=False):
    """
    Multiply together two scipy CSR matrixes using the intel Math Kernel Library.
    This is efficient if both matrices have float32 data or both matrices have float64 data.
    It will cast data to float64 if necessary but this functionality may cause a memory leak and should be avoided

    :param csr_matrix_a: Sparse matrix A in CSR format
    :type csr_matrix_a: scipy.sparse.csr_matrix
    :param csr_matrix_b: Sparse matrix B in CSR format
    :type csr_matrix_b: scipy.sparse.csr_matrix
    :param copy: Should the MKL arrays get copied and then explicitly deallocated.
    If set to True, there is a copy, but there is less risk of memory leaking.
    If set to False, numpy arrays will be created from C pointers without a copy.
    I don't know if these arrays will be garbage collected correctly by python.
    :type copy: bool
    :param reorder_output: Should the array indices be reordered using MKL
    If set to True, the object in C will be ordered and then exported into python
    If set to False, the array column indices will not be ordered.
    The scipy sparse dot product does not yield ordered column indices so this defaults to False
    :type reorder_output: bool
    :return: Sparse matrix that is the result of A * B in CSR format
    :rtype: scipy.sparse.csr_matrix
    """

    if not _spsparse.isspmatrix_csr(csr_matrix_a) or not _spsparse.isspmatrix_csr(csr_matrix_b):
        raise ValueError("Both input matrices to csr_dot_product_mkl must be CSR format")

    _check_alignment(csr_matrix_a, csr_matrix_b)
    mkl_double_precision = _check_mkl_typing(csr_matrix_a, csr_matrix_b)

    # Create intel MKL objects
    csr_mkl_a = _create_mkl_csr(csr_matrix_a, double_precision=mkl_double_precision)
    csr_mkl_b = _create_mkl_csr(csr_matrix_b, double_precision=mkl_double_precision)

    # Dot product
    csr_mkl_c = _matmul_mkl(csr_mkl_a, csr_mkl_b)

    # Reorder
    if reorder_output:
        _order_mkl_handle(csr_mkl_c)

    # Extract
    csr_python_c = _export_csr_mkl(csr_mkl_c, copy=copy)

    # Destroy
    if copy:
        _destroy_mkl_handle(csr_mkl_c)

    return csr_python_c



