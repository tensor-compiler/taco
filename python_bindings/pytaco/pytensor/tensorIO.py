from ..core import core_modules as _cm
from ..pytensor.taco_tensor import tensor


def write(filename, t):
    """
    Writes a tensor to a file.

    Writes the input tensor t to the file specified by filename using the extension given in the filename. This
    function only supports writing the extensions listed in the :ref:`io` section.

    Notes
    ---------
    This function forces tensor compilation.

    Parameters
    -------------
    filename: string
        The name of the file the tensor will be written to. The file must be given an extension of the .mtx, .tns or
        .rb.

    t: tensor
        The tensor to write to filename.


    Examples
    ----------
    To write to a .tns file we can do the following:

    .. doctest::

        >>> import pytaco as pt
        >>> a = pt.tensor([2, 2], [pt.dense, pt.compressed], dtype=pt.float32)
        >>> a.insert([0, 0], 10)
        >>> pt.write("simple_test.tns", a)

    """
    _cm._write(filename, t._tensor)


def read(filename, fmt, pack=True):
    """
    Reads a tensor from a file.

    Reads a tensor from the file filename. The extension must be one of the those supported by PyTaco listed in the
    :ref:`io` section.

    Parameters
    -------------
    filename: string
        The name of the file containing the tensor to read.

    fmt: pytaco.format
        The :class:`~format` that PyTaco should use to store the tensor.

    pack: boolean, optional
        If true, by taco will immediately pack the tensor in the specified format after reading the file. Otherwise,
        it will keep the data in a buffer until either the user explicitly calls :func:`~pytaco.tensor.pack` or
        the tensor is implicitly packed by taco.

    Examples
    ----------
    >>> import pytaco as pt
    >>> a = pt.tensor([2, 2], [pt.dense, pt.compressed], dtype=pt.float32)
    >>> a.insert([0, 0], 10)
    >>> pt.write("simple_test.tns", a)
    >>> t = pt.read("simple_test.tns", pt.csr)
    >>> t[0, 0]
    10.0

    Returns
    ---------
    tensor
        A :class:`tensor` of type double, with :class:`format` fmt containing the data read from the file filename.
    """
    cppTensor = _cm._read(filename, fmt, pack)
    return tensor._fromCppTensor(cppTensor)