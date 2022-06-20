#include "pyDatatypes.h"
#include "pybind11/numpy.h"

namespace taco{
namespace pythonBindings{

namespace py = pybind11;


std::string getNpType(const taco::Datatype& dtype) {
  if (dtype.isBool()) return "bool_";
  else if (dtype.isInt()) return "int" + std::to_string(dtype.getNumBits());
  else if (dtype.isUInt()) return "uint" + std::to_string(dtype.getNumBits());
  else if (dtype.isFloat()) return "float" + std::to_string(dtype.getNumBits());
  else if (dtype.isComplex()) return "complex" + std::to_string(dtype.getNumBits());
  else throw py::type_error("Datatype must be defined for conversion");
}

py::object asNpDtype(const taco::Datatype &dtype){
  py::module np = py::module::import("numpy");
  return np.attr(getNpType(dtype).c_str());
}

void defineTacoTypes(py::module &m){
  py::options options;
  options.disable_function_signatures();
  m.def("as_np_dtype", &asNpDtype, R"//(
as_np_dtype(dtype)

Converts a :class:`pytaco.dtype` its equivalent NumPy data type.

Parameters
------------
dtype : :class:`pytaco.dtype`
    Any PyTaco data type object.

Returns
-------------
`numpy.dtype`
    The NumPy equivalent of the PyTaco data type passed in.

Examples
----------
>>> import pytaco as pt
>>> import numpy as np
>>> pt.as_np_dtype(pt.float32)
<class 'numpy.float32'>
)//");
  options.enable_function_signatures();
  m.def("max_type", &max_type, "Get the max data type");

  py::class_<taco::Datatype> dtype(m, "dtype", R"//(

A tensor contains elements describe by this dtype object.

PyTaco currently does not provide a way to construct your own data types but provides several common data types for users.

Methods
---------
is_bool
is_uint
is_int
is_float

Examples
----------
>>> import pytaco as pt
>>> pt.int32.is_uint()
False
>>> pt.uint32.is_int()
False
>>> pt.float32 == pt.float64
False
>>> pt.bool != pt.int64
True
>>> pt.int8
pytaco.int8_t

Notes
----------
PyTaco exports the following data types:

:attr:`pytaco.bool` - A True or False value.

:attr:`pytaco.int8` - An 8 bit signed integer.

:attr:`pytaco.int16`- A 16 bit signed integer.

:attr:`pytaco.int32` - A 32 bit signed integer.

:attr:`pytaco.int64` - A 64 bit signed integer.

:attr:`pytaco.uint8` - An 8 bit unsigned integer.

:attr:`pytaco.uint16` - A 16 bit unsigned integer.

:attr:`pytaco.uint32` - A 32 bit unsigned integer.

:attr:`pytaco.uint64` - A 64 bit unsigned integer.

:attr:`pytaco.float32` or :attr:`pytaco.float` - A 32 bit floating point number.

:attr:`pytaco.float64` or :attr:`pytaco.double` - A 64 bit floating point number.

PyTaco also overrides the equality operator of the data type class so users can compare types using == and != to check
if they are the same.

See also
-------------------
as_np_dtype : Convert to NumPy dtype
)//");

  dtype.def("is_bool",    &taco::Datatype::isBool, R"//(
Returns True if the data type is a boolean type and False otherwise.
)//")

       .def("is_uint",    &taco::Datatype::isUInt, R"//(
Returns True if the data type is an unsigned integer and False otherwise.
)//")

       .def("is_int",     &taco::Datatype::isInt, R"//(
Returns True if the data type is a signed integer and False otherwise.
)//")

       .def("is_float",   &taco::Datatype::isFloat, R"//(
Returns True if the data type is a float or double and False otherwise.
)//")
       .def("is_complex", &taco::Datatype::isComplex)

       .def("__repr__",   [](const taco::Datatype& dtype) -> std::string{
         std::ostringstream o;
         o << "pytaco."<< dtype;
         return o.str();
       }, py::is_operator())

       .def("__eq__", [](const taco::Datatype& dtype, const taco::Datatype other) -> bool{
         return dtype == other;
       }, py::is_operator())

       .def("__ne__", [](const taco::Datatype& dtype, const taco::Datatype& other) -> bool{
         return dtype != other;
       }, py::is_operator())

       .def("__hash__", [](const taco::Datatype &dtype) -> int{
         return (int) dtype.getKind();
       }, py::is_operator());


  m.attr("bool")       = Bool;
  m.attr("uint8")      = UInt8;
  m.attr("uint16")     = UInt16;
  m.attr("uint32")     = UInt32;
  m.attr("uint64")     = UInt64;
  m.attr("int8")       = Int8;
  m.attr("int16")      = Int16;
  m.attr("int32")      = Int32;
  m.attr("int64")      = Int64;
  m.attr("float")      = Float32;
  m.attr("float32")    = Float32;
  m.attr("float64")    = Float64;
  m.attr("double")     = Float64;
//  m.attr("complex64")  = Complex64;
//  m.attr("complex128") = Complex128;
}

}}

