#include "pyDatatypes.h"
#include "pybind11/numpy.h"

namespace taco{
namespace pythonBindings{

namespace py = pybind11;


std::string getNpType(const taco::Datatype& dtype) {
  if (dtype.isBool()) return "bool";
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

  m.def("as_np_dtype", &asNpDtype, "Convert taco datatype to its numpy equivalent.");
  m.def("max_type", &max_type, "Get the max datatype");

  py::class_<taco::Datatype> dtype(m, "dtype", R"//(

A tensor contains elements describe by this dtype object. PyTaco currently does not provide a way to construct
your own datatypes but provides several common datatypes for users.



)//");

  dtype.def("is_bool",    &taco::Datatype::isBool, R"//(
is_bool() -> bool

Returns
---------
bool
True if the datatype is a boolean type and False otherwise.

)//")
       .def("is_uint",    &taco::Datatype::isUInt)
       .def("is_int",     &taco::Datatype::isInt)
       .def("is_float",   &taco::Datatype::isFloat)
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

