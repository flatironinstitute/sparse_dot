import unittest
import numpy as np

from sparse_dot_mkl import (
    mkl_get_version,
    mkl_get_version_string,
    mkl_get_max_threads,
    mkl_set_interface_layer,
    mkl_set_num_threads,
    mkl_set_num_threads_local,
    mkl_interface_integer_dtype
)


class TestServiceFunctions(unittest.TestCase):

    def test_version(self):
        version_info = mkl_get_version()
        self.assertTrue(isinstance(version_info[0], int))
        self.assertTrue(isinstance(version_info[1], int))
        self.assertTrue(isinstance(version_info[2], int))
        self.assertTrue(isinstance(version_info[3], str))
        self.assertTrue(isinstance(version_info[4], str))
        self.assertTrue(isinstance(version_info[5], str))
        self.assertTrue(isinstance(version_info[6], str))

    def test_version_str(self):
        version_info = mkl_get_version_string()
        self.assertTrue(isinstance(version_info, str))

    def test_get_threads(self):
        n_threads = mkl_get_max_threads()
        self.assertTrue(isinstance(n_threads, int))

    def test_set_threads(self):
        n_threads_before = mkl_set_num_threads_local(1)
        mkl_set_num_threads(1)
        self.assertEqual(mkl_get_max_threads(), 1)
        mkl_set_num_threads_local(n_threads_before)

    def test_set_interface_layer(self):
        mkl_set_interface_layer(0)

        with self.assertRaises(ValueError):
            mkl_set_interface_layer("MKL")

    def test_get_integer_interface(self):
        self.assertTrue(
            mkl_interface_integer_dtype() in [
                np.int32,
                np.int64
            ]
        )
