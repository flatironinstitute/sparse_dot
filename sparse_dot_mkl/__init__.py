__version__ = '0.9.6'


from sparse_dot_mkl.sparse_dot import (
    dot_product_mkl,
    dot_product_transpose_mkl,
    gram_matrix_mkl,
    sparse_qr_solve_mkl,
    set_debug_mode
)

from ._mkl_interface import (
    mkl_get_max_threads,
    mkl_set_interface_layer,
    mkl_set_num_threads,
    mkl_get_version,
    mkl_get_version_string,
    mkl_set_num_threads_local,
    mkl_interface_integer_dtype
)

from .solvers import (
    pardiso,
    pardisoinit,
    fgmres,
    cg
)

get_version_string = mkl_get_version_string
