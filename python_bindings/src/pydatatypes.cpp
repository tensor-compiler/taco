#include "pydatatypes.h"
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

  m.def("as_np_dtype", &asNpDtype, "Convert taco datatype to its numpy equivalent.");

  py::class_<taco::Datatype> dtype(m, "dtype");

  dtype.def(py::init<>())
       .def(py::init<taco::Datatype::Kind>())
       .def("get_kind",   &taco::Datatype::getKind)
       .def("is_bool",    &taco::Datatype::isBool)
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



  py::enum_<taco::Datatype::Kind>(dtype, "Kind")
          .value("bool",       taco::Datatype::Kind::Bool)
          .value("uint8",      taco::Datatype::Kind::UInt8)
          .value("uint16",     taco::Datatype::Kind::UInt16)
          .value("uint32",     taco::Datatype::Kind::UInt32)
          .value("uint64",     taco::Datatype::Kind::UInt64)
          .value("uint128",    taco::Datatype::Kind::UInt128)
          .value("int8",       taco::Datatype::Kind::Int8)
          .value("int16",      taco::Datatype::Kind::Int16)
          .value("int32",      taco::Datatype::Kind::Int32)
          .value("int64",      taco::Datatype::Kind::Int64)
          .value("int128",     taco::Datatype::Kind::Int128)
          .value("float32",    taco::Datatype::Kind::Float32)
          .value("float64",    taco::Datatype::Kind::Float64)
          .value("complex64",  taco::Datatype::Kind::Complex64)
          .value("complex128", taco::Datatype::Kind::Complex128)
          .value("undefined",  taco::Datatype::Kind::Undefined)
          .export_values();


  m.attr("bool")       = Bool;
  m.attr("uint8")      = UInt8;
  m.attr("uint16")     = UInt16;
  m.attr("uint32")     = UInt32;
  m.attr("uint64")     = UInt64;
  m.attr("int8")       = Int8;
  m.attr("int16")      = Int16;
  m.attr("int32")      = Int32;
  m.attr("int64")      = Int64;
  m.attr("float32")    = Float32;
  m.attr("float64")    = Float64;
  m.attr("complex64")  = Complex64;
  m.attr("complex128") = Complex128;
}

}}

