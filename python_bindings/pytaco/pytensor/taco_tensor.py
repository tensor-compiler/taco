import operator
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ..core import core_modules as _cm

default_mode = _cm.compressed

_dtype_to_tensor = {_cm.bool:    _cm.TensorBool,
                    _cm.float32: _cm.TensorFloat,
                    _cm.float64: _cm.TensorDouble,
                    _cm.int8:    _cm.TensorInt8,
                    _cm.int16:   _cm.TensorInt16,
                    _cm.int32:   _cm.TensorInt32,
                    _cm.int64:   _cm.TensorInt64,
                    _cm.uint8:   _cm.TensorUInt8,
                    _cm.uint16:  _cm.TensorUInt16,
                    _cm.uint32:  _cm.TensorUInt32,
                    _cm.uint64:  _cm.TensorUInt64}

_dtype_error = "Invalid datatype. Must be bool, float32/64, (u)int8, (u)int16, (u)int32 or (u)int64"


class tensor:
    """ A mathematical tensor.

        A tensor object represents a mathematical tensor of arbitrary dimensions and is at the heart of this
        library . A tensor must consist of a homogeneous :class:`~pytaco.dtype` and be stored in a given
        :class:`~pytaco.format`. They can optionally be given a name.

        Taco allows users to compressed certain dimensions of tensors which means only the non-zeros and their index
        information is stored.

        Parameters
        -------------
            arg1:  int, float, iterable, optional

                If this argument is a python int or float, PyTaco will create a scalar tensor and initialize it to the
                with the value passed in. If arg1 is an iterable, PyTaco will interpret this as the shape and initialize
                a tensor with the given shape. The default value is none meaning that PyTaco will simply create an empty
                scalar tensor and ignore the fmt argument.

            fmt: :class:`~pytaco.format`, optional
                Format

            dtype: :class:`~pytaco.dtype`, optional
                Dtype

            name: string, optional
                Tensor name

        Examples
        ------------
        Create a scalar tensor with the value 42.

        >>> import pytaco as pt
        >>> t = pt.tensor(42)


    """

    def __init__(self, arg1=None, fmt=_cm.compressed, dtype=_cm.float32, name=None):

        if name is None:
            name = _cm.unique_name('A')

        if isinstance(arg1, int) or isinstance(arg1, float) or isinstance(arg1, bool) or not arg1:
            init_func = _dtype_to_tensor.get(dtype)
            if init_func is None:
                raise ValueError(_dtype_error)
            self._tensor = init_func(name)

            if arg1 is not None:
                self._tensor.insert([], arg1 if arg1 else 0)
                self._tensor.pack()

        elif isinstance(arg1, tuple) or isinstance(arg1, list):
            shape = arg1
            init_func = _dtype_to_tensor.get(dtype)
            if init_func is None:
                raise ValueError(_dtype_error)
            self._tensor = init_func(name, shape, fmt)
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
        init_func = _dtype_to_tensor.get(dtype)
        if init_func is None:
            raise ValueError(_dtype_error)
        return cls._fromCppTensor(init_func(x))

    @classmethod
    def from_tensor_base(cls, tensor_base):
        return cls._from_x(tensor_base, tensor_base.dtype())

    @property
    def order(self):
        """
        Order of tensor
        """
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
        if self.order < 2:
            return self
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
        """
            Packs a tensor
        """
        self._tensor.pack()

    def compile(self):
        """
            Compiles current expression
        """
        self._tensor.compile()

    def assemble(self):
        self._tensor.assemble()

    def evaluate(self):
        self._tensor.evaluate()

    def compute(self):
        self._tensor.compile()

    def __iter__(self):
        return iter(self._tensor)


    def __getitem__(self, index):
        return self._tensor[index]

    def __setitem__(self, key, value):
        self._tensor[key] = value

    def __repr__(self):
        return self._tensor.__repr__()

    def __add__(self, other):
        return tensor_add(self, other, default_mode)

    def __radd__(self, other):
        return tensor_add(other, self, default_mode)

    def __sub__(self, other):
        return tensor_sub(self, other, default_mode)

    def __rsub__(self, other):
        return tensor_sub(other, self, default_mode)

    def __mul__(self, other):
        return tensor_mul(self, other, default_mode)

    def __rmul__(self, other):
        return tensor_mul(other, self, default_mode)

    def __truediv__(self, other):
        return tensor_div(self, other, default_mode)

    def __rtruediv__(self, other):
        return tensor_div(other, self, default_mode)

    def __floordiv__(self, other):
        return tensor_floordiv(self, other, default_mode)

    def __rfloordiv__(self, other):
        return tensor_floordiv(other, self, default_mode)

    def __ge__(self, other):
        return tensor_ge(self, other, default_mode)

    def __gt__(self, other):
        return tensor_gt(self, other, default_mode)

    def __le__(self, other):
        return tensor_le(self, other, default_mode)

    def __lt__(self, other):
        return tensor_lt(self, other, default_mode)

    def __ne__(self, other):
        return tensor_ne(self, other, default_mode)

    def __eq__(self, other):
        return tensor_eq(self, other, default_mode)

    def __pow__(self, power, modulo=None):
        return tensor_pow(self, power, default_mode)

    def __abs__(self):
        return tensor_abs(self, default_mode)

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

    def to_array(self):
        return to_array(self)

    def toarray(self):
        return self.to_array()

    def to_sp_csr(self):
        return to_sp_csr(self)

    def to_sp_csc(self):
        return to_sp_csc(self)

    def copy(self):
        new_t = tensor(self.shape, self.format, dtype=self.dtype)
        idx_vars = _cm.get_index_vars(self.order)
        new_t[idx_vars] = self[idx_vars]
        return new_t

    def insert(self, coords, vals):
        self._tensor.insert(coords, vals)


def _from_matrix(inp_mat, copy, csr):
    matrix = inp_mat
    if not inp_mat.has_sorted_indices:
        matrix = inp_mat.sorted_indices()

    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    shape = matrix.shape
    return tensor._fromCppTensor(_cm.fromSpMatrix(indptr, indices, data, shape, copy, csr))


def from_sp_csr(matrix, copy=True):
    """
    Convert a sparse scipy matrix to a CSR taco tensor.

    Initializes a taco tensor from a scipy.sparse.csr_matrix object. This function copies the data by default.

    Parameters
    -----------
    matrix: scipy.sparse.csr_matrix
        A sparse scipy matrix to use to initialize the tensor.

    copy: boolean, optional
        If true, taco copies the data from scipy and stores it. Otherwise, taco points to the same data as scipy.

    Returns
    --------
    t: tensor
        A taco tensor pointing to the same underlying data as the scipy matrix if copy was set to False. Otherwise,
        returns a taco tensor containing data copied from the scipy matrix.
    """
    return _from_matrix(matrix, copy, True)  # true if csr false otherwise


def from_sp_csc(matrix, copy=True):
    """
    Convert a sparse scipy matrix to a CSC taco tensor.

    Initializes a taco tensor from a scipy.sparse.csc_matrix object. This function copies the data by default.

    Parameters
    -----------
    matrix: scipy.sparse.csc_matrix
        A sparse scipy matrix to use to initialize the tensor.

    copy: boolean, optional
        If true, taco copies the data from scipy and stores it. Otherwise, taco points to the same data as scipy.

    Returns
    --------
    t: tensor
        A taco tensor pointing to the same underlying data as the scipy matrix if copy was set to False. Otherwise,
        returns a taco tensor containing data copied from the scipy matrix.
    """
    return _from_matrix(matrix, copy, False)


def from_array(array, copy=True):

    """Convert a numpy array to a tensor.

    Initializes a taco tensor from a numpy array and copies the array by default. This always creates a dense
    tensor.

    Parameters
    ------------
    array: numpy.array
        A numpy array to convert to a taco tensor

    copy: boolean, optional
        If true, taco copies the data from numpy and stores its own copy. If false, taco points to the same
        underlying data as the numpy array.

    Warnings
    ---------
    Taco's changes to tensors may NOT be visible to numpy since taco places inserts in buffers may copy tensor data
    after inserting. See notes for details.

    Notes
    --------
    The copy flag is ignored if the input array is not C contiguous or F contiguous (so for most transposed views).
    If taco detects an array that is not contiguous, it will always copy the numpy array into a C contiguous format.
    This restriction will be lifted in future versions of taco.

    Taco is mainly intended to operate on sparse tensors. As a result, it buffers inserts since inserting into sparse
    structures is very costly. This means that when the full tensor structure is needed, taco will copy the tensor to
    another location and insert the new values as needed. This saves a lot of time when dealing with sparse structures
    but is not needed for dense tensors (like numpy arrays). Currently, taco does this copy for dense and sparse tensors.
    As a result, after inserting into a taco tensor numpy will not see the changes since taco will not be writing to
    the same memory location that numpy is referencing.


    See also
    ----------
    :func:`from_sp_csc`, :func:`from_sp_csr`

    Examples
    ----------
    If we choose not to copy, modifying the tensor also modifies the numpy array and vice-versa. An example of this is
    shown:

    .. doctest::

        >>> import numpy as np
        >>> import pytaco as pt
        >>> arr = np.array([0, 1, 2, 3]) # Note that this is contiguous so copy possible
        >>> t = pt.from_array(arr, copy=False)
        >>> arr[0] = 23
        >>> t[0]
        23



    Returns
    --------
    t: tensor
        A taco tensor pointing to the same underlying data as the numpy array if copy was set to False. Otherwise,
        returns a taco tensor containing data copied from the numpy array.
    """


    # For some reason disabling conversion in pybind11 still copies C and F style arrays unnecessarily.
    # Disabling the force convert parameter also seems to not work. This explicity calls the different functions
    # to get this working for now
    col_major = array.flags["F_CONTIGUOUS"]
    t = _cm.fromNpF(array, copy) if col_major else _cm.fromNpC(array, copy)
    return tensor._fromCppTensor(t)


def to_array(t):
    """
    Converts a taco tensor to a numpy array.

    This always copies the tensor. To avoid the copy for dense tensors, see the notes section.

    Parameters
    -----------
    t: tensor
        A taco tensor to convert to a numpy array.

    Notes
    -------
    Dense tensors export python's buffer interface. As a result, they can be converted to numpy arrays using
    ``np.array(tensor, copy=False)`` . Attempting to do this for sparse tensors throws an error. Note that as a result
    of exporting the buffer interface dense tensors can also be converted to eigen or any other library supporting this
    inferface.

    Also it is very important to note that if requesting a numpy view of data owned by taco, taco will mark the array as
    read only meaning the user cannot write to that data without using the taco reference. This is needed to avoid
    raising issues with taco's delayed execution mechanism.

    Examples
    ----------
    We first look at a simple use of to_array

    >>> import pytaco as pt
    >>> t = pt.tensor([2, 2], [pt.dense, pt.compressed])
    >>> t.insert([0, 0], 10)
    >>> t.to_array()[0, 0]
    10.0


    One could choose to use np.array if a copy is not needed


    >>> import pytaco as pt
    >>> import numpy as np
    >>> t = pt.tensor([2, 2], pt.dense)
    >>> t.insert([0, 0], 10)
    >>> a = np.array(t, copy=False)
    >>> a
    array([[10.,  0.],
           [ 0.,  0.]], dtype=float32)
    >>> t.insert([0, 0], 100) # Note that insert increments instead of setting!
    >>> t.to_array()[0, 0]
    110.0


    Returns
    ---------
    arr: numpy.array
        A numpy array containing a copy of the data in the tensor object t.

    """
    return np.array(t.to_dense(), copy=True)


def to_sp_csr(t):
    """

    Converts a taco tensor to a scipy csr_matrix.

    Takes a matrix from taco in any format and converts the matrix to a scipy sparse csr matrix. This method removes
    explicit zeros from the original taco tensor during the conversion.

    Parameters
    -----------
    t: tensor
        A taco tensor to convert to a scipy.csr_matrix array. The tensor must be of order 2 (i.e it must be a matrix).
        If the order of the tensor is not equal to 2, a value error is thrown.


    Notes
    -------
    The data and index values are always copied when making the scipy sparse array


    Returns
    ---------
    matrix: scipy.sparse.csr_matrix
        A matrix containing a copy of the data from the original order 2 tensor t.

    """
    arrs = _cm.to_sp_matrix(t._tensor, True)
    return csr_matrix((arrs[2], arrs[1], arrs[0]), shape=t.shape)


def to_sp_csc(t):
    """

    Converts a taco tensor to a scipy csc_matrix.

    Takes a matrix from taco in any format and converts the matrix to a scipy sparse csc matrix. This method removes
    explicit zeros from the original taco tensor during the conversion.

    Parameters
    -----------
    t: tensor
        A taco tensor to convert to a scipy.csc_matrix array. The tensor must be of order 2 (i.e it must be a matrix).
        If the order of the tensor is not equal to 2, a value error is thrown.


    Notes
    -------
    The data and index values are always copied when making the scipy sparse array


    Returns
    ---------
    matrix: scipy.sparse.csc_matrix
        A matrix containing a copy of the data from the original order 2 tensor t.

"""
    arrs = _cm.to_sp_matrix(t._tensor, False)
    return csc_matrix((arrs[2], arrs[1], arrs[0]), shape=t.shape)


def as_tensor(obj, copy=True):
    """
        Converts array_like or scipy csr and csr to tensors.

        Converts an array_like object (list of lists, etc..) or scipy csr and csc matrices to a taco tensor.

        Parameters
        ------------
        obj: array_like, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, tensor
            The object to convert to a taco tensor. If the object is a tensor, it will be copied depending on the copy
            flag.

        copy: boolean. optional
            If true, taco will attempt to take a reference to the input if possible. If false, taco will always
            copy the input.

        Notes
        ------
        This method internally uses :func:`from_array`, :func:`from_sp_csr` and :func:`from_sp_csc`. As a result the restrictions
        to those methods and their copy parameters apply here. For instance, non-contiguous arrays will always be copied
        regardless of the copy flag.

        Python objects will also always be copied since this internally uses np.array to create an array_like
        object.

        Returns
        ----------
        t: tensor
            A tensor initialized with the data from the object passed in.

    """

    if isinstance(obj, tensor):
        return obj.copy() if copy else obj

    if isinstance(obj, int) or isinstance(obj, float):
        return tensor(obj)

    if isinstance(obj, np.ndarray):
        return from_array(obj, copy)

    if isinstance(obj, csc_matrix):
        return from_sp_csc(obj, copy)

    if isinstance(obj, csr_matrix):
        return from_sp_csr(obj, copy)

    # Try converting object to numpy array. This will ignore the copy flag
    arr = np.array(obj)
    return from_array(arr, True)


def _is_broadcastable(shape1, shape2):
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a != b:  # for singleton dimension we would need && a != 1 and b != 1 but this isn't current supported
            return False
    return True


def _compute_elt_wise_out_shape(shape1, shape2):
    if not _is_broadcastable(shape1, shape2):
        raise ValueError("Shapes {} and {} cannot be broadcasted together".format(shape1, shape2))

    return shape1 if len(shape1) >= len(shape2) else shape2


def _get_indices_for_operands(result_indices, order1, order2):
    # This returns a tuple of the index variables that should be used from
    # result_indices to access shapeA and shapeB
    start_a = len(result_indices) - order1
    start_b = len(result_indices) - order2
    return result_indices[start_a:], result_indices[start_b:]


def _compute_bin_elt_wise_op(op, t1, t2, out_format, dtype=None):

    t1, t2 = as_tensor(t1, False), as_tensor(t2, False)
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


def tensor_add(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.add, t1, t2, out_format, dtype)


def tensor_mul(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.mul, t1, t2, out_format, dtype)


def tensor_sub(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.sub, t1, t2, out_format, dtype)


def tensor_div(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.truediv, t1, t2, out_format, dtype)


def tensor_floordiv(t1, t2, out_format, dtype=_cm.int64):
    if not dtype.is_int() or not dtype.is_uint():
        raise ValueError("Floor divide must have int data type as output")
    return _compute_bin_elt_wise_op(operator.floordiv, t1, t2, out_format, dtype)


def tensor_gt(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.gt, t1, t2, out_format, dtype)


def tensor_ge(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.ge, t1, t2, out_format, dtype)


def tensor_lt(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.lt, t1, t2, out_format, dtype)


def tensor_le(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.le, t1, t2, out_format, dtype)


def tensor_ne(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.ne, t1, t2, out_format, dtype)


def tensor_eq(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.eq, t1, t2, out_format, dtype)


def tensor_pow(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(operator.pow, t1, t2, out_format, dtype)


def tensor_max(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(_cm.max, t1, t2, out_format, dtype)


def tensor_min(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(_cm.min, t1, t2, out_format, dtype)


def tensor_heaviside(t1, t2, out_format, dtype=None):
    return _compute_bin_elt_wise_op(_cm.heaviside, t1, t2, out_format, dtype)


def _compute_unary_elt_eise_op(op, t1, out_format, dtype=None):

    t1 = as_tensor(t1, False)
    out_dtype = t1.dtype if dtype is None else dtype
    out_shape = t1.shape

    if out_shape:
        result = tensor(out_shape, out_format, dtype=out_dtype)
        index_var_list = _cm.get_index_vars(t1.order)
        result[index_var_list] = op(t1[index_var_list])
        return result
    else:
        result = tensor(dtype=out_dtype)
        result[None] = op(t1[None])
        return result


def tensor_logical_not(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.logical_not, t1, out_format, dtype)


def tensor_abs(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.abs, t1, out_format, dtype)


def tensor_square(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.square, t1, out_format, dtype)


def tensor_cube(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.cube, t1, out_format, dtype)


def tensor_sqrt(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.sqrt, t1, out_format, dtype)


def tensor_cube_root(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.cube_root, t1, out_format, dtype)


def tensor_exp(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.exp, t1, out_format, dtype)


def tensor_log(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.log, t1, out_format, dtype)


def tensor_log10(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.log10, t1, out_format, dtype)


def tensor_sin(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.sin, t1, out_format, dtype)


def tensor_cos(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.cos, t1, out_format, dtype)


def tensor_tan(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.tan, t1, out_format, dtype)


def tensor_asin(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.asin, t1, out_format, dtype)


def tensor_acos(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.acos, t1, out_format, dtype)


def tensor_atan(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.atan, t1, out_format, dtype)


def tensor_atan2(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.atan2, t1, out_format, dtype)


def tensor_sinh(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.sinh, t1, out_format, dtype)


def tensor_cosh(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.cosh, t1, out_format, dtype)


def tensor_tanh(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.tanh, t1, out_format, dtype)


def tensor_asinh(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.asinh, t1, out_format, dtype)


def tensor_acosh(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.acosh, t1, out_format, dtype)


def tensor_atanh(t1, out_format, dtype=None):
    return _compute_unary_elt_eise_op(_cm.atanh, t1, out_format, dtype)


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


def tensor_sum(t1, axis=None, out_format=default_mode, dtype=None):
    t1 = as_tensor(t1, False)

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


def matmul(t1, t2, out_format=default_mode, dtype=None):
    t1, t2 = as_tensor(t1, False), as_tensor(t2, False)

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


def inner(t1, t2, out_format=default_mode, dtype=None):
    t1, t2 = as_tensor(t1, False), as_tensor(t2, False)

    if t1.order == 0 or t2.order == 0:
        return tensor_mul(t1, t2, out_format, dtype)

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


def dot(t1, t2, out_format=default_mode, dtype=None):
    t1, t2 = as_tensor(t1, False), as_tensor(t2, False)
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


def outer(t1, t2, out_format=default_mode, dtype=None):
    t1, t2 = as_tensor(t1, False), as_tensor(t2, False)
    t1_order = t1.order
    t2_order = t2.order
    if t1_order == 0 or t2_order == 0:
        return tensor_mul(t1, t2, out_format, dtype)

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


def tensordot(t1, t2, axes=2, out_format=default_mode, dtype = None):

    # This is largely adapted from numpy's tensordot source code
    t1, t2 = as_tensor(t1, False), as_tensor(t2, False)
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


def evaluate(expr, *args, out_format=None, dtype=None):
    """
    Evaluates the index notation expression on the input operands.

    An output tensor may be optionally specified. In this case, the tensor should be given the expected output shape,
    format and dtype since the out_format and dtype fields will be ignored if an output tensor is seen.

    """

    args = [as_tensor(t) for t in args]
    if len(args) < 2:
        raise ValueError("Expression must have at least one operand on the LHS and one on the RHS.")

    out_dtype = args[0].dtype if dtype is None else dtype
    if dtype is None:
        for i in range(1, len(args)):
            out_dtype = _cm.max_type(out_dtype, args[i].dtype)

    tensor_base = _cm._parse(expr, [t._tensor for t in args], out_format, out_dtype)
    return tensor.from_tensor_base(tensor_base)


def einsum(expr, *operands, out_format=None, dtype=None):
    """
    Evaluates the Einstein summation convention on the input operands.

    The einsum summation convention employed here is very similar to `numpy's
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html#numpy.einsum>`_.

    Taco's einsum can express a wide variety of linear algebra expressions in a simple fashion. Einsum can be used
    in implicit mode where no output indices are specified. In this mode, it follows the usual einstein summation
    convention to compute an output. In explicit mode, the user can force summation over specified subscript variables.

    Note that this einsum parser is a subset of what taco can express. The full :func:`~evaluate` supports a much
    larger range of possible expressions.

    See the notes section for more details.

    Warnings
    ----------
    This differs from numpy's einsum in two important ways. The first is that the same subscript cannot appear more than
    once in a given operand.

    Parameters
    ------------
    expr: str
        Specifies the subscripts for summation as a comma separated list of subscript variables. An implicit (Classical
        Einstein summation) is calculation is performed unless there is an explicit indicator '->' included along with
        subscript labels specifying the output.

    operands: list of array_like, tensors, scipy csr and scipy csc matrices
        This specifies the operands for the computation. Taco will copy any numpy arrays that are not stored in
        row-major or column-major format.

    out_format: format, optional
        The storage :class:`format` of the output tensor.

     dtype: datatype, optional
        The datatype of the output tensor.


    See also
    ----------
    :func:`evaluate`

    Notes
    --------

    `einsum` provides a succint way to represent a large number of tensor algebra expressions. A list of some possible
    operations along with some examples is presented below:

    * Sum axes of tensor :func:`~sum`
    * Transpose tensors or transpose to dense tensors.
    * Matrix multiplication and dot products
    * Tensor contractions
    * Fused operations

    The expr string is a comma separated list of subscript labels where each label corresponds to a dimension in the
    tensor. In implicit mode, repeated subscripts are summed. This means that ``pt.einsum('ij,jk')`` is matrix
    multiplication since the j indices are implicitly summed over.

    """

    args = [as_tensor(t) for t in operands]
    out_dtype = args[0].dtype if dtype is None else dtype
    if dtype is None:
        for i in range(1, len(args)):
            out_dtype = _cm.max_type(out_dtype, args[i].dtype)

    ein = _cm._einsum(expr, [t._tensor for t in args], out_format, out_dtype)
    return tensor.from_tensor_base(ein)


# Change parser to execute
# Change pytaco.parse to pytaco.eval
# add from and to array functions in tensorio