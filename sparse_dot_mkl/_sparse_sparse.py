from sparse_dot_mkl._mkl_interface import (
    MKL,
    sparse_matrix_t,
    _create_mkl_sparse,
    debug_timer,
    _export_mkl,
    _order_mkl_handle,
    _destroy_mkl_handle,
    _type_check,
    _empty_output_check,
    _sanity_check,
    _is_allowed_sparse_format,
    _check_return_value,
    _output_dtypes,
    sparse_output_type,
    _out_matrix
)
import ctypes as _ctypes


def _matmul_mkl(sp_ref_a, sp_ref_b):
    """
    Dot product two MKL objects together and return a handle to the result

    :param sp_ref_a: Sparse matrix A handle
    :type sp_ref_a: sparse_matrix_t
    :param sp_ref_b: Sparse matrix B handle
    :param sp_ref_b: sparse_matrix_t
    :return: Sparse matrix handle that is the dot product A * B
    :rtype: sparse_matrix_t
    """

    ref_handle = sparse_matrix_t()

    ret_val = MKL._mkl_sparse_spmm(
        _ctypes.c_int(10),
        sp_ref_a,
        sp_ref_b,
        _ctypes.byref(ref_handle)
    )

    # Check return
    _check_return_value(ret_val, "mkl_sparse_spmm")
    return ref_handle


# Dict keyed by ('double_precision_bool', 'complex_bool')
_mkl_spmmd_funcs = {
    (False, False): MKL._mkl_sparse_s_spmmd,
    (True, False): MKL._mkl_sparse_d_spmmd,
    (False, True): MKL._mkl_sparse_c_spmmd,
    (True, True): MKL._mkl_sparse_z_spmmd,
}


def _matmul_mkl_dense(
    sp_ref_a,
    sp_ref_b,
    output_shape,
    double_precision,
    out=None,
    complex_type=False
):
    """
    Dot product two MKL objects together into a dense numpy array and
    return the result

    :param sp_ref_a: Sparse matrix A handle
    :type sp_ref_a: sparse_matrix_t
    :param sp_ref_b: Sparse matrix B handle
    :type sp_ref_b: sparse_matrix_t
    :param output_shape: The shape of the output array
        This must be correct or the preallocated numpy array won't be correct
        and this will probably segfault
    :type output_shape: tuple(int, int)
    :param double_precision: The resulting array will be float64
    :type double_precision: bool

    :return: Dense numpy array that's the output of A dot B
    :rtype: np.array
    """

    # Allocate an array for outputs
    output_arr = _out_matrix(
        output_shape,
        _output_dtypes[(double_precision, complex_type)],
        out_arr=out
    )

    # Set types
    output_ctype = _ctypes.c_double if double_precision else _ctypes.c_float
    func = _mkl_spmmd_funcs[(double_precision, complex_type)]

    ret_val = func(
        10,
        sp_ref_a,
        sp_ref_b,
        101,
        output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
        output_shape[1],
    )

    # Check return
    _check_return_value(ret_val, func.__name__)

    return output_arr


def _sparse_dot_sparse(
    matrix_a,
    matrix_b,
    cast=False,
    reorder_output=False,
    dense=False,
    out=None
):
    """
    Multiply together two scipy sparse matrixes using the intel
    Math Kernel Library. This currently only supports float32 and float64 data

    :param matrix_a: Sparse matrix A in CSC/CSR format
    :type matrix_a: scipy.sparse.spmatrix
    :param matrix_b: Sparse matrix B in CSC/CSR format
    :type matrix_b: scipy.sparse.spmatrix
    :param cast: Should the data be coerced into float64 if it isn't
        float32 or float64
        If set to True and any other dtype is passed, the matrix data will
        be modified in-place
        If set to False and any dtype that isn't float32 or float64 is passed,
        a ValueError will be raised
        Defaults to False
    :param reorder_output: Should the array indices be reordered using MKL
        If set to True, the object in C will be ordered and then exported
        into python
        If set to False, the array column indices will not be ordered.
        The scipy sparse dot product does not yield ordered column indices so
        this defaults to False
    :type reorder_output: bool
    :param dense: Should the matrix multiplication yield a dense numpy array
        This does not require any copy and is more memory efficient if the
        output array density is > 50%
    :type dense: bool
    :return: Sparse matrix that is the result of A @ B in appropriate
        sparse format
    :rtype: scipy.sparse.csr_matrix
    """

    # Check for allowed sparse matrix types
    if (
        not _is_allowed_sparse_format(matrix_a) or
        not _is_allowed_sparse_format(matrix_b)
    ):
        raise ValueError(
            "Input matrices to dot_product_mkl must be CSR, CSC, or BSR; "
            "COO is not supported"
        )

    if out is not None and not dense:
        raise ValueError(
            "out argument cannot be used with sparse (dot) sparse "
            "matrix multiplication unless dense=True"
        )

    default_output, output_type = sparse_output_type(matrix_a)

    # Check to make sure that this multiplication can work and check dtypes
    _sanity_check(matrix_a, matrix_b)

    # Check for edge condition inputs which result in empty outputs
    if _empty_output_check(matrix_a, matrix_b):

        if dense:
            return _out_matrix(
                (matrix_a.shape[0], matrix_b.shape[1]),
                matrix_a.dtype,
                out_arr=out
            )
        else:
            return default_output(
                (matrix_a.shape[0], matrix_b.shape[1]),
                dtype=matrix_a.dtype
            )

    # Check dtypes
    matrix_a, matrix_b = _type_check(matrix_a, matrix_b, cast=cast)

    t = debug_timer()

    # Create intel MKL objects
    mkl_a, a_dbl, a_cplx = _create_mkl_sparse(matrix_a)
    mkl_b, b_dbl, b_cplx = _create_mkl_sparse(matrix_b)

    t = debug_timer("Created MKL sparse handles", t)

    # Call spmmd for dense output directly if the dense flag is set
    if dense:

        try:
            dense_arr = _matmul_mkl_dense(
                mkl_a,
                mkl_b,
                (matrix_a.shape[0], matrix_b.shape[1]),
                a_dbl or b_dbl,
                complex_type=a_cplx,
                out=out
            )
        finally:
            _destroy_mkl_handle(mkl_a)
            _destroy_mkl_handle(mkl_b)

        debug_timer("Multiplied matrices", t)

        return dense_arr

    # Call spmm for sparse output if the dense flag is not set and then export
    # the sparse matrix to python
    else:
        # Dot product
        mkl_c = _matmul_mkl(mkl_a, mkl_b)

        _destroy_mkl_handle(mkl_a)
        _destroy_mkl_handle(mkl_b)

        t = debug_timer("Multiplied matrices", t)

        # Reorder
        if reorder_output:
            _order_mkl_handle(mkl_c)

            t = debug_timer("Reordered output indices", t)

        # Extract
        python_c = _export_mkl(
            mkl_c,
            a_dbl or b_dbl,
            complex_type=a_cplx,
            output_type=output_type
        )

        _destroy_mkl_handle(mkl_c)

        debug_timer("Created python handle", t)

        return python_c
