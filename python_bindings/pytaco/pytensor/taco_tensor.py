import operator
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ..core import core_modules as _cm

dtype_to_tensor = {_cm.bool:    _cm.TensorBool,
                   _cm.float64: _cm.TensorFloat,
                   _cm.float32: _cm.TensorDouble,
                   _cm.int8:    _cm.TensorInt8,
                   _cm.int16:   _cm.TensorInt16,
                   _cm.int32:   _cm.TensorInt32,
                   _cm.int64:   _cm.TensorInt64,
                   _cm.uint8:   _cm.TensorUInt8,
                   _cm.uint16:  _cm.TensorUInt16,
                   _cm.uint32:  _cm.TensorUInt32,
                   _cm.uint64:  _cm.TensorUInt64}

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

    def __add__(self, other):
        return add(self, other, _cm.dense)

    def __radd__(self, other):
        return add(other, self, _cm.dense)

    def __sub__(self, other):
        return subtract(self, other, _cm.dense)

    def __rsub__(self, other):
        return subtract(other, self, _cm.dense)

    def __mul__(self, other):
        return multiply(self, other, _cm.dense)

    def __rmul__(self, other):
        return multiply(other, self, _cm.dense)

    def __truediv__(self, other):
        return divide(self, other, _cm.dense)

    def __rtruediv__(self, other):
        return divide(other, self, _cm.dense)

    def __floordiv__(self, other):
        return floordiv(self, other, _cm.dense)

    def __rfloordiv__(self, other):
        return floordiv(other, self, _cm.dense)

    def __array__(self):
        if not _cm.is_dense(self.format):
            raise ValueError("Cannot export a compressed tensor. Make sure all dimensions are dense "
                             "using to_dense() before attempting this conversion.")
        a = np.array(self._tensor, copy=False)
        a.setflags(write=False)  # forbid user from changing array via numpy if they request a copy.
        return a

    def to_dense(self):
        new_t = tensor(self.shape, _cm.dense, dtype=self.dtype)
        idx_vars = _cm.get_index_vars(self.order)
        new_t[idx_vars] = self[idx_vars]
        return new_t

    def copy(self):
        new_t = tensor(self.shape, self.format, dtype=self.dtype)
        idx_vars = _cm.get_index_vars(self.order)
        new_t[idx_vars] = self[idx_vars]
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


def astensor(obj, copy=True):

    if isinstance(obj, tensor):
        return obj.copy() if copy else obj

    if isinstance(obj, int) or isinstance(obj, float):
        return tensor(obj)

    if isinstance(obj, np.ndarray):
        return from_numpy_array(obj, copy)

    if isinstance(obj, csc_matrix):
        return from_sp_csc(obj, copy)

    if isinstance(obj, csr_matrix):
        return from_sp_csr(obj, copy)

    # Try converting object to numpy array. This will ignore the copy flag
    arr = np.array(obj)
    return from_numpy_array(arr, True)


def _is_broadcastable(shape1, shape2):
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a != b:  # for singleton dimension we would need && a != 1 and b != 1 but this isn't current supported
            return False
    return True


def _compute_elt_wise_out_shape(shape1, shape2):
    if not _is_broadcastable(shape1, shape2):
        raise ValueError("Shapes {} and {} cannot be added together".format(shape1, shape2))

    return shape1 if len(shape1) >= len(shape2) else shape2


def _get_indices_for_operands(result_indices, order1, order2):
    # This returns a tuple of the index variables that should be used from
    # result_indices to access shapeA and shapeB
    start_a = len(result_indices) - order1
    start_b = len(result_indices) - order2
    return result_indices[start_a:], result_indices[start_b:]


def _compute_elt_wise_op(op, t1, t2, out_format, dtype=None):

    t1, t2 = astensor(t1, False), astensor(t2, False)
    out_dtype = _cm.max_type(t1.dtype, t2.dtype) if dtype is None else dtype
    out_shape = _compute_elt_wise_out_shape(t1.shape, t2.shape)

    if out_shape:
        result = tensor(out_shape, out_format, dtype=out_dtype)
        index_var_list = _cm.get_index_vars(len(out_shape))
        index_var1, index_var2 = _get_indices_for_operands(index_var_list, t1.order, t2.order)
        result[index_var_list] = op(t1[index_var1], t2[index_var2])
        return result
    else:
        result = tensor(dtype=out_dtype)
        result[None] = op(t1[None], t2[None])
        return result


def add(t1, t2, out_format, dtype=None):
    return _compute_elt_wise_op(operator.add, t1, t2, out_format, dtype)


def multiply(t1, t2, out_format, dtype=None):
    return _compute_elt_wise_op(operator.mul, t1, t2, out_format, dtype)


def subtract(t1, t2, out_format, dtype=None):
    return _compute_elt_wise_op(operator.sub, t1, t2, out_format, dtype)


def divide(t1, t2, out_format, dtype=None):
    return _compute_elt_wise_op(operator.truediv, t1, t2, out_format, dtype)


def floordiv(t1, t2, out_format, dtype=_cm.int64):
    if not dtype.is_int() or not dtype.is_uint():
        raise ValueError("Floor divide must have int data type as output")
    return _compute_elt_wise_op(operator.floordiv, t1, t2, out_format, dtype)


def _remove_elts_at_index(inp, elts_to_remove):
    result = inp[:]
    for elt in elts_to_remove:
        if elt >= len(inp) or elt < 0:
            raise ValueError("Axis {} too large for tensor of order {}".format(elt, len(inp)))
        result[elt] = None

    return list(filter(lambda x: x is not None, result))


def _as_list(x):
    if type(x) is list:
        return x
    else:
        return [x]


def reduce_sum(t1, axis=None, out_format=_cm.dense, dtype=None):
    t1 = astensor(t1, False)

    out_dtype = t1.dtype if dtype is None else dtype
    res_shape = [] if axis is None else _remove_elts_at_index(t1.shape, _as_list(axis))

    inp_index_vars = _cm.get_index_vars(t1.order)
    out_index_vars = [] if axis is None else _remove_elts_at_index(inp_index_vars, _as_list(axis))

    result_tensor = tensor(res_shape, format_type=out_format, dtype=out_dtype)
    result_tensor[out_index_vars] = t1[inp_index_vars]
    return result_tensor


def _matrix_out_shape(shape1, shape2):
    if len(shape1) < 2 or len(shape2) < 2:
        raise ValueError("Invalid tensor order for matrix multiply. Must be at least order 2 but operand1 has "
                         "order {} while operand 2 has order {}.".format(len(shape1), len(shape2)))

    if shape1[-1] != shape2[-2]:
        raise ValueError("Input operand1 has value {} in dimension 0 while operand2 has value {} in dimension 1. "
                         "These two dimensions must be equal".format(shape1[-1], shape2[-2]))

    if not _is_broadcastable(shape1[:-2], shape2[:-2]):
        raise ValueError("Cannot broadcast outer tensor elements. All the leading dimensions must be equal.")

    result_shape = shape1[:-2] if len(shape1) >= len(shape2) else shape2[:-2]
    result_shape.extend([shape1[-2], shape2[-1]])
    return result_shape


def matmul(t1, t2, out_format=_cm.dense, dtype=None):
    t1, t2 = astensor(t1, False), astensor(t2, False)

    out_dtype = _cm.max_type(t1.dtype, t2.dtype) if dtype is None else dtype
    out_shape = _matrix_out_shape(t1.shape, t2.shape)
    result_tensor = tensor(out_shape, format_type=out_format, dtype=out_dtype)

    reduction_var = _cm.index_var()
    leading_vars = _cm.get_index_vars(max(t1.order, t2.order))
    t1_vars, t2_vars = leading_vars[-t1.order:], leading_vars[-t2.order:]
    t1_vars[-1] = reduction_var
    t2_vars[-2] = reduction_var
    result_tensor[leading_vars] = t1[t1_vars] * t2[t2_vars]
    return result_tensor


def inner(t1, t2, out_format=_cm.dense, dtype=None):
    t1, t2 = astensor(t1, False), astensor(t2, False)

    if t1.order == 0 or t2.order == 0:
        return multiply(t1, t2, out_format, dtype)

    if t1.order != 0 and t2.order != 0 and t1.shape[-1] != t2.shape[-1]:
        raise ValueError("Last dimensions of t1 and t2 must be equal but t1 has dimension {} while "
                         "t2 has dimension %".format(t1.shape[-1], t2.shape[-1]))

    out_dtype = _cm.max_type(t1.dtype, t2.dtype) if dtype is None else dtype
    out_shape = t1.shape[:-1] + t2.shape[:-1]

    reduction_var = _cm.index_var()
    t1_vars, t2_vars = _cm.get_index_vars(len(t1.shape[:-1])), _cm.get_index_vars(len(t2.shape[:-1]))
    result_vars = t1_vars + t2_vars

    t1_vars.append(reduction_var)
    t2_vars.append(reduction_var)

    result_tensor = tensor(out_shape, out_format, dtype=out_dtype)
    result_tensor[result_vars] = t1[t1_vars] * t2[t2_vars]
    return result_tensor


def _dot_output_shape(shape1, shape2):

    if shape1[-1] != shape2[-2]:
        raise ValueError("Input operand1 has value {} in dimension 0 while operand2 has value {} in dimension 1."
                         " These two dimensions must be equal".format(shape1[-1], shape2[-2]))

    return shape1[:-1] + shape2[:-2] + shape2[-1:]


def dot(t1, t2, out_format=_cm.dense, dtype=None):
    t1, t2 = astensor(t1, False), astensor(t2, False)
    if t1.order == 0 or t2.order <= 1:
        return inner(t1, t2, out_format, dtype)

    # Here we know that a is a non-scalar and b has order at least 2
    out_shape = _dot_output_shape(t1.shape, t2.shape)
    out_dtype = _cm.max_type(t1.dtype, t2.dtype) if dtype is None else dtype

    t1_vars = _cm.get_index_vars(t1.order - 1)
    t2_vars = _cm.get_index_vars(t2.order - 1)
    res_vars = t1_vars + t2_vars

    reduction_var = _cm.index_var()
    t1_vars.append(reduction_var)
    t2_vars = t2_vars[:-1] + [reduction_var] + t2_vars[-1:]
    result_tensor = tensor(out_shape, format_type=out_format, dtype=out_dtype)
    result_tensor[res_vars] = t1[t1_vars] * t2[t2_vars]
    return result_tensor


def outer(t1, t2, out_format=_cm.dense, dtype=None):
    t1, t2 = astensor(t1, False), astensor(t2, False)
    t1_order = t1.order
    t2_order = t2.order
    if t1_order == 0 or t2_order == 0:
        return multiply(t1, t2, out_format, dtype)

    if t1_order != 1:
        raise ValueError("Can only perform outer product with vectors and scalars but first "
                         "operand has {} dimensions.".format(t1.order))

    if t2_order != 1:
        raise ValueError("Can only perform outer product with vectors and scalars but second "
                         "operand has {} dimensions.".format(t2.order))

    out_shape = t1.shape + t2.shape
    out_dtype = _cm.max_type(t1.dtype, t2.dtype) if dtype is None else dtype
    out_vars = _cm.get_index_vars(2)

    result_tensor = tensor(out_shape, format_type=out_format, dtype=out_dtype)
    result_tensor[out_vars] = t1[out_vars[0]] * t2[out_vars[1]]
    return result_tensor


def tensordot(t1, t2, axes=2, out_format=_cm.dense, dtype = None):

    # This is largely adapted from numpy's tensordot source code
    t1, t2 = astensor(t1, False), astensor(t2, False)
    try:
        iter(axes)
    except Exception:
        axes_t1 = list(range(-axes, 0))
        axes_t2 = list(range(0, axes))
    else:
        axes_t1, axes_t2 = axes

    try:
        nt1 = len(axes_t1)
        axes_t1 = list(axes_t1)
    except TypeError:
        axes_t1 = [axes_t1]
        nt1 = 1

    try:
        nt2 = len(axes_t2)
        axes_t2 = list(axes_t2)
    except TypeError:
        axes_t2 = [axes_t2]
        nt2 = 1

    t1_shape = t1.shape
    t1_order = t1.order
    t2_shape = t2.shape
    t2_order = t2.order

    equal = True

    if nt1 != nt2:
        equal = False
    else:
        for k in range(nt1):
            if t1_shape[axes_t1[k]] != t2_shape[axes_t2[k]]:
                equal = False
                break
            if axes_t1[k] < 0:
                axes_t1[k] += t1_order
            if axes_t2[k] < 0:
                axes_t2[k] += t2_order
    if not equal:
        raise ValueError("shape-mismatch for sum")

    notin_t1 = [k for k in range(t1_order) if k not in axes_t1]
    out_shape_t1 = [t1_shape[axis] for axis in notin_t1]

    notin_t2 = [k for k in range(t2_order) if k not in axes_t2]
    out_shape_t2 = [t2_shape[axis] for axis in notin_t2]

    static_vars_t1 = _cm.get_index_vars(len(notin_t1))
    static_vars_t2 = _cm.get_index_vars(len(notin_t2))
    reduction_vars = _cm.get_index_vars(nt1)

    t1_vars = [None] * t1_order
    t2_vars = [None] * t2_order
    for k in range(len(reduction_vars)):
        t1_vars[axes_t1[k]] = reduction_vars[k]
        t2_vars[axes_t2[k]] = reduction_vars[k]

    for k in range(len(static_vars_t1)):
        t1_vars[notin_t1[k]] = static_vars_t1[k]

    for k in range(len(static_vars_t2)):
        t2_vars[notin_t2[k]] = static_vars_t2[k]

    out_vars = static_vars_t1 + static_vars_t2

    out_shape = out_shape_t1 + out_shape_t2
    out_dtype = _cm.max_type(t1.dtype, t2.dtype) if dtype is None else dtype
    result_tensor = tensor(out_shape, format_type=out_format, dtype=out_dtype)
    result_tensor[out_vars] = t1[t1_vars] * t2[t2_vars]
    return result_tensor


def parse(expr, *args, out_format=None, dtype=None):
    args = [astensor(t) for t in args]
    if len(args) < 2:
        raise ValueError("Expression must have at least one operand on the LHS and one on the RHS.")

    out_dtype = args[0].dtype if dtype is None else dtype
    if dtype is None:
        for i in range(1, len(args)):
            out_dtype = _cm.max_type(out_dtype, args[i].dtype)

    tensor_base = _cm._parse(expr, [t._tensor for t in args], out_format, out_dtype)
    return tensor.from_tensor_base(tensor_base)


def einsum(expr, *args, out_format=None, dtype=None):
    args = [astensor(t) for t in args]
    out_dtype = args[0].dtype if dtype is None else dtype
    if dtype is None:
        for i in range(1, len(args)):
            out_dtype = _cm.max_type(out_dtype, args[i].dtype)

    ein = _cm._einsum(expr, [t._tensor for t in args], out_format, out_dtype)
    return tensor.from_tensor_base(ein)