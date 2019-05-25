from ..core import core_modules as _cm
import numpy as np

dtype_to_tensor = {_cm.bool: _cm.TensorBool,
                   _cm.float64: _cm.TensorFloat,
                   _cm.float32: _cm.TensorDouble,
                   _cm.int8: _cm.TensorInt8,
                   _cm.int16: _cm.TensorInt16,
                   _cm.int32: _cm.TensorInt32,
                   _cm.int64: _cm.TensorInt64,
                   _cm.uint8: _cm.TensorUInt8,
                   _cm.uint16: _cm.TensorUInt16,
                   _cm.uint32: _cm.TensorUInt32,
                   _cm.uint64: _cm.TensorUInt64}

dtype_error = "Invalid datatype. Must be bool, float32/64, (u)int8, (u)int16, (u)int32 or (u)int64"


"""
    This is a class used to hide the different tensor types and export a numpy-style interface.

    This is a light wrapper that creates the correct C++ tensor given the dtype. This wrapper class
    needs to stay in sync with the underlying bindings. 
"""


class tensor:

    def __init__(self, arg1=None, format_type=_cm.compressed, dtype=_cm.float32, name=None):

        if name is None:
            name = _cm.unique_name('A')

        if isinstance(arg1, int) or isinstance(arg1, float) or not arg1:
            init_func = dtype_to_tensor.get(dtype)
            if init_func is None:
                raise ValueError(dtype_error)
            self._tensor = init_func(name)

            if arg1 is not None:
                self._tensor[None] = arg1
                self._tensor.pack()

        elif isinstance(arg1, tuple) or isinstance(arg1, list):
            shape = arg1
            init_func = dtype_to_tensor.get(dtype)
            if init_func is None:
                raise ValueError(dtype_error)
            self._tensor = init_func(name, shape, format_type)
        else:
            raise ValueError("Invalid argument for first argument. Must be a tuple or list if a shape or a single value"
                             "if initializing a scalar.")

    @classmethod
    def _fromCppTensor(cls, cppTensor):
        pytensor = cls()
        pytensor._tensor = cppTensor
        return pytensor

    @classmethod
    def _from_x(cls, x, dtype):
        init_func = dtype_to_tensor.get(dtype)
        if init_func is None:
            raise ValueError(dtype_error)
        return cls._fromCppTensor(init_func(x))

    @classmethod
    def from_tensor_base(cls, tensor_base):
        return cls._from_x(tensor_base, tensor_base.dtype())

    @property
    def order(self):
        return self._tensor.order()

    @property
    def name(self):
        return self._tensor.get_name()

    @name.setter
    def name(self, name):
        self._tensor.set_name(name)

    @property
    def shape(self):
        return self._tensor.get_dimensions()

    @property
    def dtype(self):
        return self._tensor.dtype()

    @property
    def format(self):
        return self._tensor.format()

    @property
    def T(self):
        new_ordering = list(range(self.order))[::-1]
        return self.transpose(new_ordering)

    def transpose(self, new_ordering, new_format=None, name=None):
        if name is None:
            name = _cm.unique_name('A')

        if new_format is None:
            new_format = self.format

        new_t = self._tensor.transpose(new_ordering, new_format, name)
        return tensor._fromCppTensor(new_t)

    def pack(self):
        self._tensor.pack()

    def compile(self):
        self._tensor.compile()

    def assemble(self):
        self._tensor.assemble()

    def evaluate(self):
        self._tensor.evaluate()

    def compute(self):
        self._tensor.compile()

    def __getitem__(self, index):
        return self._tensor[index]

    def __setitem__(self, key, value):
        self._tensor[key] = value

    def __repr__(self):
        return self._tensor.__repr__()

    def __array__(self):
        if not _cm.is_dense(self.format):
            raise ValueError("Cannot export a compressed tensor. Make sure all dimensions are dense "
                             "using to_dense() before attempting this conversion.")
        a = np.array(self._tensor, copy=False)
        a.setflags(write=False)  # forbid user from changing array via numpy if they request a copy.
        return a

    def to_dense(self):
        new_t = tensor(self.shape, _cm.dense, dtype=self.dtype)
        vars = _cm.get_index_vars(self.order)
        new_t[vars] = self[vars]
        return new_t

    def insert(self, coords, vals):
        self._tensor.insert(coords, vals)


def from_numpy_array(array, copy=False):
    # For some reason disabling conversion in pybind11 still copies C and F style arrays unnecessarily.
    # Disabling the force convert parameter also seems to not work. This explicity calls the different functions
    # to get this working for now
    col_major = array.flags["F_CONTIGUOUS"]
    t = _cm.fromNpF(array, copy) if col_major else _cm.fromNpC(array, copy)
    return tensor._fromCppTensor(t)


def _from_matrix(inp_mat, copy, csr):
    matrix = inp_mat
    if not inp_mat.has_sorted_indices:
        matrix = inp_mat.sorted_indices()

    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    shape = matrix.shape
    return tensor._fromCppTensor(_cm.fromSpMatrix(indptr, indices, data, shape, copy, csr))


def from_sp_csr(matrix, copy=True):
    return _from_matrix(matrix, copy, True)


def from_sp_csc(matrix, copy=True):
    return _from_matrix(matrix, copy, False)


