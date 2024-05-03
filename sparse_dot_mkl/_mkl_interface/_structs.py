import ctypes as _ctypes


# Construct opaque struct & type
class _sparse_matrix(_ctypes.Structure):
    pass


sparse_matrix_t = _ctypes.POINTER(_sparse_matrix)


# Matrix description struct
class matrix_descr(_ctypes.Structure):
    _fields_ = [
        ("sparse_matrix_type_t", _ctypes.c_int),
        ("sparse_fill_mode_t", _ctypes.c_int),
        ("sparse_diag_type_t", _ctypes.c_int)
    ]

    def __init__(
        self,
        sparse_matrix_type_t=20,
        sparse_fill_mode_t=0,
        sparse_diag_type_t=0
    ):
        super(matrix_descr, self).__init__(
            sparse_matrix_type_t,
            sparse_fill_mode_t,
            sparse_diag_type_t
        )


# Complex type structs
# These are the same as the np.csingle and np.cdouble structs
# They're defined to allow passing complex scalars of specific
# precisions directly
class MKL_Complex8(_ctypes.Structure):
    _fields_ = [
        ("real", _ctypes.c_float),
        ("imag", _ctypes.c_float)
    ]

    def __init__(self, cplx):
        super(MKL_Complex8, self).__init__(
            cplx.real,
            cplx.imag
        )


class MKL_Complex16(_ctypes.Structure):
    _fields_ = [
        ("real", _ctypes.c_double),
        ("imag", _ctypes.c_double)
    ]

    def __init__(self, cplx):
        super(MKL_Complex16, self).__init__(
            cplx.real,
            cplx.imag
        )


# MKLVersion Struct
# For mkl_get_version

class MKLVersion(_ctypes.Structure):

    _fields_ = [
        ('MajorVersion', _ctypes.c_int),
        ('MinorVersion', _ctypes.c_int),
        ('UpdateVersion', _ctypes.c_int),
        ('ProductStatus', _ctypes.POINTER(_ctypes.c_char*128)),
        ('Build', _ctypes.POINTER(_ctypes.c_char*128)),
        ('Processor', _ctypes.POINTER(_ctypes.c_char*128)),
        ('Platform', _ctypes.POINTER(_ctypes.c_char*128))
    ]
