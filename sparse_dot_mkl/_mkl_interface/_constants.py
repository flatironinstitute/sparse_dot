# Define standard return codes
RETURN_CODES = {0: "SPARSE_STATUS_SUCCESS",
                1: "SPARSE_STATUS_NOT_INITIALIZED",
                2: "SPARSE_STATUS_ALLOC_FAILED",
                3: "SPARSE_STATUS_INVALID_VALUE",
                4: "SPARSE_STATUS_EXECUTION_FAILED",
                5: "SPARSE_STATUS_INTERNAL_ERROR",
                6: "SPARSE_STATUS_NOT_SUPPORTED"}

# Define order codes
LAYOUT_CODE_C = 101
LAYOUT_CODE_F = 102

# Define transpose codes
SPARSE_OPERATION_NON_TRANSPOSE = 10
SPARSE_OPERATION_TRANSPOSE = 11
SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12

# Define cblas transpose codes
CBLAS_NO_TRANS = 111
CBLAS_TRANS = 112
CBLAS_CONJ_TRANS = 113

# Define index codes
SPARSE_INDEX_BASE_ZERO = 0
SPARSE_INDEX_BASE_ONE = 1

# Define sparse & cblas upper or lower triangle code
MKL_UPPER = 121,
MKL_LOWER = 122

# ILP64 message
ILP64_MSG = " Try changing MKL to int64 with the environment variable MKL_INTERFACE_LAYER=ILP64"

