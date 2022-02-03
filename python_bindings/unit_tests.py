import sys
sys.path.insert(1, "@CMAKE_LIBRARY_OUTPUT_DIRECTORY@")
import unittest, os, shutil, tempfile
import pytaco as pt
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

types = [pt.bool, pt.float32, pt.float64, pt.int8, pt.int16, pt.int32, pt.int64,
         pt.uint8, pt.uint16, pt.uint32, pt.uint64]


class TestDataTypeMethods(unittest.TestCase):
    # This is just to ensure methods are callable from python. Correctness ensured by C++ suite

    # Covers == and != and ensures all dtypes in the list above are present and callable in python
    def test_dtype_equality(self):
        for i in range(len(types)):
            for j in range(len(types)):
                if i == j:
                    self.assertTrue(types[i] == types[j])
                    self.assertFalse(types[i] != types[j])
                else:
                    self.assertTrue(types[i] != types[j])
                    self.assertFalse(types[i] == types[j])

    # Tests methods of the datatype class are callable
    def test_dtype_inspectors(self):
        self.assertTrue(pt.float64.is_float())
        self.assertFalse(pt.float32.is_uint())
        self.assertTrue(pt.int16.is_int())
        self.assertFalse(pt.bool.is_complex())
        self.assertTrue(pt.bool.is_bool())
        self.assertEqual(pt.uint64.__repr__(), "pytaco.uint64_t")

    def test_dtype_conversion(self):
        expected_types = [np.bool_, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64]

        for i, dt in enumerate(types):
            self.assertEqual(pt.as_np_dtype(dt), expected_types[i])


class TestFormatMethods(unittest.TestCase):

    def test_mode_format_methods(self):
        self.assertEqual(pt.dense.name, "dense")
        self.assertEqual(pt.compressed.name, "compressed")

    def test_format_methods(self):
        mfs = [pt.dense, pt.compressed, pt.compressed]
        mf_ordering = [2, 0, 1]
        fmt = pt.format(mfs, mf_ordering)
        self.assertEqual(fmt.order, len(mf_ordering))
        self.assertEqual(fmt.mode_formats, mfs)
        self.assertEqual(fmt.mode_ordering, mf_ordering)
        self.assertTrue(pt.csr != fmt)
        self.assertTrue(fmt == fmt)
        self.assertTrue(pt.csc == pt.csc)
        self.assertFalse(pt.is_dense(pt.csc))


class TestIO(unittest.TestCase):

    def setUp(self):
        self.dir_name = tempfile.mkdtemp()
        self.names = [os.path.join(self.dir_name, dtype.__repr__() + "{}".format(i)) for i, dtype in enumerate(types)]
        tensors = [np.ones([3, 3]).astype(pt.as_np_dtype(dt)) for dt in types]
        self.tensors = [pt.from_array(t, copy=True) for t in tensors]
        self.comp_tensors = [pt.tensor([3, 3], pt.csc, dt) for dt in types]

        self.coord = [2, 2]
        self.val = 10
        for t2 in self.comp_tensors:
            t2.insert(self.coord, self.val)  # force .tns to infer 3x3 shape

    def test_write_then_read_dense(self):
        file_outs = [".tns", ".mtx", ".ttx"]

        for out_type in file_outs:
            idx = 0
            for t, fp in zip(self.tensors, self.names):
                pt.write(fp + out_type, t)
                self.assertEqual(self.tensors[idx], pt.read(fp + out_type, pt.dense))
                idx += 1

    def test_write_then_read_compressed(self):
        file_outs_2 = [".tns", ".mtx", ".ttx", ".rb"]

        for out_type in file_outs_2:
            idx = 0
            for t, fp in zip(self.comp_tensors, self.names):
                pt.write(fp + out_type, t)
                self.assertEqual(self.comp_tensors[idx], pt.read(fp + out_type, pt.csc))
                idx += 1

    def test_to_and_from_np_and_sp(self):
        arrs = [np.array([[1, 0], [0, 4]]).astype(pt.as_np_dtype(dt)) for dt in types]
        csrs = [csr_matrix(arr) for arr in arrs]
        cscs = [csc_matrix(arr) for arr in arrs]
        tens = [pt.from_array(arr) for arr in arrs]

        for ten, csr in zip(tens, csrs):
            self.assertEqual((ten.to_sp_csr() != csr).nnz, 0)

        for ten, csc in zip(tens, cscs):
            self.assertEqual((ten.to_sp_csc() != csc).nnz, 0)

        for ten, arr in zip(tens, arrs):
            self.assertTrue(np.array_equal(ten.to_array(), arr))

    def test_iterator(self):
        in_components = [([0, 1], 1.0), ([2, 2], 2.0), ([2, 3], 3.0), ([4, 0], 4.0)]
        A = pt.tensor([5, 5], pt.csr)
        for coords, val in in_components:
          A.insert(coords, val)
        A.insert([3, 3], 0.0)
        out_components = [components for components in A]
        self.assertTrue(in_components == out_components)

    def tearDown(self):
        shutil.rmtree(self.dir_name)


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
        if pt.should_use_cuda_codegen():
          # `from_array` always performs deep copy when GPU backend is enabled, 
          # so don't run this test
          return

        # Getting a reference to taco then back to numpy should return the same data with the read only flag set to true
        # only for C and F style arrays. Arrays of different forms will always be copied
        c_copy = pt.from_array(self.c_array, copy=False)
        f_copy = pt.from_array(self.f_array, copy=False)
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


class TestSchedulingCommands(unittest.TestCase):

    def setUp(self):
        self.original_schedule = pt.get_parallel_schedule()
        self.original_threads = pt.get_num_threads()

    def test_get_and_set_threads(self):
        self.assertEqual(pt.get_num_threads(), 1)
        pt.set_num_threads(4)
        self.assertEqual(pt.get_num_threads(), 4)

    def test_parallel_sched(self):
        pt.set_parallel_schedule("dynamic", 4)
        self.assertSequenceEqual(pt.get_parallel_schedule(), ("dynamic", 4))
        pt.set_parallel_schedule("static", 1)
        self.assertSequenceEqual(pt.get_parallel_schedule(), ("static", 1))

    def tearDown(self):
        pt.set_parallel_schedule(self.original_schedule[0], self.original_schedule[1])
        pt.set_num_threads(self.original_threads)


class TestIndexFuncs(unittest.TestCase):

    def test_reduce(self):
        arr = np.arange(1, 5).reshape([2, 2])
        t = pt.from_array(arr)
        res = pt.tensor()
        i, j = pt.get_index_vars(2)
        res[None] = pt.sum(j, pt.sum(i, t[i, j]))
        self.assertEqual(res[0], np.sum(arr))

    def test_mod(self):
        arr = np.arange(1, 5).reshape([2, 2])
        t = pt.from_array(arr)
        t1 = pt.tensor([2, 2], pt.dense)
        i, j = pt.get_index_vars(2)
        t1[i, j] = pt.remainder(t[i, j], 2)
        self.assertEqual(t1, arr % 2)

    def test_neg(self):
        arr = np.arange(1, 5).reshape([2, 2])
        t = pt.from_array(arr)
        self.assertEqual(-t, -arr)

class testParsers(unittest.TestCase):

    def test_evaluate(self):
        a = np.arange(25).reshape(5, 5)
        t = pt.tensor([5, 5], pt.csr)
        for i in range(5):
            t.insert([i, i], a[i, i])

        c = pt.evaluate("T(j) = A(i, j)", a)
        self.assertTrue(np.array_equal(c.to_array(), a.sum(axis=0)))

        result = pt.tensor([5], pt.dense)
        v = pt.evaluate("T(j) = A(i, j)", result, t)
        self.assertEqual(v, result)

unittest.main(verbosity=2)
