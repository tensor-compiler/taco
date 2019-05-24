from ..core import core_modules as _cm
import numpy as np

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
            if dtype == _cm.float32:
                self._tensor = _cm.TensorFloat(name)
            elif dtype == _cm.float64:
                self._tensor = _cm.TensorDouble(name)
            elif dtype == _cm.int8:
                self._tensor = _cm.TensorInt8(name)
            elif dtype == _cm.int16:
                self._tensor = _cm.TensorInt16(name)
            elif dtype == _cm.int32:
                self._tensor = _cm.TensorInt32(name)
            elif dtype == _cm.int64:
                self._tensor = _cm.TensorInt64(name)
            elif dtype == _cm.uint8:
                self._tensor = _cm.TensorUInt8(name)
            elif dtype == _cm.uint16:
                self._tensor = _cm.TensorUInt16(name)
            elif dtype == _cm.uint32:
                self._tensor = _cm.TensorUInt32(name)
            elif dtype == _cm.uint64:
                self._tensor = _cm.TensorUInt64(name)
            else:
                raise ValueError("Invalid datatype. Must be float32/64, (u)int8, (u)int16, (u)int32 or (u)int64")

            if arg1 is not None:
                self._tensor[None] = arg1
                self._tensor.pack()

        elif isinstance(arg1, tuple) or isinstance(arg1, list):
            shape = arg1
            if dtype == _cm.float32:
                self._tensor = _cm.TensorFloat(name, shape, format_type)
            elif dtype == _cm.float64:
                self._tensor = _cm.TensorDouble(name, shape, format_type)
            elif dtype == _cm.int8:
                self._tensor = _cm.TensorInt8(name, shape, format_type)
            elif dtype == _cm.int16:
                self._tensor = _cm.TensorInt16(name, shape, format_type)
            elif dtype == _cm.int32:
                self._tensor = _cm.TensorInt32(name, shape, format_type)
            elif dtype == _cm.int64:
                self._tensor = _cm.TensorInt64(name, shape, format_type)
            elif dtype == _cm.uint8:
                self._tensor = _cm.TensorUInt8(name, shape, format_type)
            elif dtype == _cm.uint16:
                self._tensor = _cm.TensorUInt16(name, shape, format_type)
            elif dtype == _cm.uint32:
                self._tensor = _cm.TensorUInt32(name, shape, format_type)
            elif dtype == _cm.uint64:
                self._tensor = _cm.TensorUInt64(name, shape, format_type)
            else:
                raise ValueError("Invalid datatype. Must be float32/64, (u)int8, (u)int16, (u)int32 or (u)int64")

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
        if dtype == _cm.float32:
            return cls._fromCppTensor(_cm.TensorFloat(x))
        elif dtype == _cm.float64:
            return cls._fromCppTensor(_cm.TensorDouble(x))
        elif dtype == _cm.int8:
            return cls._fromCppTensor(_cm.TensorInt8(x))
        elif dtype == _cm.int16:
            return cls._fromCppTensor(_cm.TensorInt16(x))
        elif dtype == _cm.int32:
            return cls._fromCppTensor(_cm.TensorInt32(x))
        elif dtype == _cm.int64:
            return cls._fromCppTensor(_cm.TensorInt64(x))
        elif dtype == _cm.uint8:
            return cls._fromCppTensor(_cm.TensorUInt8(x))
        elif dtype == _cm.uint16:
            return cls._fromCppTensor(_cm.TensorUInt16(x))
        elif dtype == _cm.uint32:
            return cls._fromCppTensor(_cm.TensorUInt32(x))
        elif dtype == _cm.uint64:
            return cls._fromCppTensor(_cm.TensorUInt64(x))
        else:
            raise ValueError("Invalid datatype ({}). Must be float32/64, (u)int8, (u)int16, "
                             "(u)int32 or (u)int64".format(dtype))

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

    def insert(self, coords, vals):
        self._tensor.insert(coords, vals)


def from_numpy_array(array, copy=False):
    # For some reason disabling conversion in pybind11 still copies C and F style arrays unnecessarily.
    # Disabling the force convert parameter also seems to not work. This explicity calls the different functions
    # to get this working for now
    col_major = array.flags["F_CONTIGUOUS"]
    t = _cm.fromNpF(array, copy) if col_major else _cm.fromNpC(array, copy)
    return tensor._fromCppTensor(t)


def _from_matrix(matrix, copy, csr):
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    shape = matrix.shape
    return tensor._fromCppTensor(_cm.fromSpMatrix(indptr, indices, data, shape, copy, csr))


def from_sp_csr(matrix, copy=True):
    return _from_matrix(matrix, copy, True)


def from_sp_csc(matrix, copy=True):
    return _from_matrix(matrix, copy, False)


