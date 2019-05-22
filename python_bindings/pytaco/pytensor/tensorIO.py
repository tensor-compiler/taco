from ..core import core_modules as _cm
from ..pytensor.taco_tensor import tensor


def write(filename, t):
    _cm._write(filename, t._tensor)


def read(filename, fmt, pack=True):
    cppTensor = _cm._read(filename, fmt, pack)
    return tensor._fromCppTensor(cppTensor)