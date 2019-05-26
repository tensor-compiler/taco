import unittest
import pytaco as pt
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

types = [pt.bool, pt.float32, pt.float64, pt.int8, pt.int16, pt.int32, pt.int64,
         pt.uint8, pt.uint16, pt.uint32, pt.uint64]


class TestTensorCreation(unittest.TestCase):

    def setUp(self):
        self.shape2 = [10, 10]
        self.shape5 = [4, 4, 4, 4, 4]

        self.format2 = pt.csr
        self.format5 = pt.format([pt.compressed]*5)

        self.implicit_scalar = pt.tensor([])
        self.false_bool_tensor = pt.tensor(False)

        self.scalars = [pt.tensor(dtype=dt) for dt in types]
        self.order2 = [pt.tensor(self.shape2, self.format2, dtype=dt) for dt in types]
        self.order5 = [pt.tensor(self.shape5, pt.compressed, dt) for dt in types]

        self.setScalar = [pt.tensor(i, dtype=dt) for i, dt in enumerate(types)]

        self.c_array = np.array(np.arange(100).reshape([10, 10]), order='C')
        self.f_array = np.array(np.arange(100).reshape([10, 10]), order='F')

    def check_vals(self, tensor_list, expected_shape, expected_order, expected_format):
        for i, tensor in enumerate(tensor_list):
            self.assertEqual(tensor.shape, expected_shape)
            self.assertEqual(tensor.dtype, types[i])
            self.assertEqual(tensor.order, expected_order)
            self.assertEqual(tensor.format, expected_format)

    def test_values_correctly_set(self):
        self.check_vals(self.scalars, [], 0, pt.format())
        self.check_vals(self.order2, self.shape2, len(self.format2), self.format2)
        self.check_vals(self.order5, self.shape5, len(self.format5), self.format5)
        self.assertEqual(self.implicit_scalar[0], 0)
        self.assertFalse(self.false_bool_tensor[0])

        for i, tensor in enumerate(self.setScalar):
            self.assertEqual(tensor[0], i)

    def test_tensor_from_numpy(self):
        tensor = pt.from_array(self.c_array, copy=True)
        tensor_arr = np.array(tensor)
        tensor_copy = np.array(tensor, copy=False)
        self.assertFalse(tensor_copy.flags["WRITEABLE"])
        self.assertTrue(np.array_equal(tensor_arr, self.c_array))
        self.assertTrue(np.array_equal(tensor_copy, self.c_array))

    def test_array_copy_C_and_F_style(self):
        # Getting a reference to taco then back to numpy should return the same data with the read only flag set to true
        # only for C and F style arrays. Arrays of different forms will always be copied
        c_copy = pt.from_array(self.c_array)
        f_copy = pt.from_array(self.f_array)
        same_c_array = np.array(c_copy, copy=False)
        same_f_array = np.array(f_copy, copy=False)
        pointer_self_f, read_only_flag_self_f = self.f_array.__array_interface__['data']
        pointer_self_c, read_only_flag_self_c = self.c_array.__array_interface__['data']
        pointer_c, read_only_flag_c = same_c_array.__array_interface__['data']
        pointer_f, read_only_flag_f = same_f_array.__array_interface__['data']

        self.assertFalse(read_only_flag_self_c)
        self.assertFalse(read_only_flag_self_f)
        self.assertTrue(read_only_flag_c)
        self.assertTrue(read_only_flag_f)
        self.assertEqual(pointer_c, pointer_self_c)
        self.assertEqual(pointer_f, pointer_self_f)

    def test_reshaped_array(self):
        i, j, k = 2, 4, 3
        a = np.arange(i*j*k).reshape([i, j, k])
        a = a.transpose([1, 0, 2])

        # It is possible to handle transposes in general without copying but this isn't supported yet
        taco_should_copy = pt.from_array(a, copy=False)

        # taco should have ignored the copy flag
        new_arr = np.array(taco_should_copy, copy=False)
        pointer_a, read_only_flag_a = a.__array_interface__['data']
        pointer_new, read_only_flag_new = new_arr.__array_interface__['data']

        self.assertFalse(read_only_flag_a)
        self.assertTrue(read_only_flag_new)
        self.assertNotEqual(pointer_a, pointer_new) # taco should force copy
        self.assertEqual(taco_should_copy.shape, [j, i, k])

        # Check that data is the same
        for ii in range(i):
            for jj in range(j):
                for kk in range(k):
                    self.assertEqual(a[jj, ii, kk], taco_should_copy[jj, ii, kk])


suite = unittest.TestLoader().loadTestsFromTestCase(TestTensorCreation)
unittest.TextTestRunner(verbosity=2).run(suite)



