#include "pyTensor.h"
#include "taco/tensor.h"
#include "pybind11/operators.h"
#include "taco/type.h"
#include "pybind11/stl.h"

// Add Python dictionary initializer with {tuple(coordinate) : data} pairs


namespace taco{
namespace pythonBindings{

static void checkBounds(const std::vector<int>& dims, const std::vector<int>& indices){
  if(dims.size() != indices.size()){
    std::ostringstream o;
    o << "Incorrect number of dimensions when indexing. Tensor is order " << dims.size() << " but got index of "
                                                                                            "size " << indices.size();
    o << ". To index multiple dimensions only \"fancy\" notation is supported. For example to access the first "
         "element of a matrix, use A[0, 0] instead of A[0][0].";
    throw py::value_error(o.str());
  }

  for(size_t i = 0; i < dims.size(); ++i){
    if(indices[i] >= dims[i]){
      std::ostringstream o;
      o << "Index out of range for dimension " << i << ". Dimension shape is " << dims[i] << " but index value is "
           << indices[i];
      throw py::index_error(o.str());
    }
  }
}

template<typename CType>
static void declareTensor(py::module &m, std::string typestr) {

  using typedTensor = Tensor<CType>;

  std::string pyClassName = std::string("Tensor") + typestr;
  py::class_<typedTensor>(m, pyClassName.c_str())

          .def(py::init<>())

          .def(py::init<std::string>(), py::arg("name"))

          .def(py::init<CType>(), py::arg("value"))

          .def(py::init<std::vector<int>, ModeFormat>(), py::arg("shape"),
               py::arg("mode_type") = ModeFormat::compressed)

          .def(py::init<std::vector<int>, Format>(), py::arg("shape"), py::arg("format"))

          .def(py::init<std::string, std::vector<int>, ModeFormat>(), py::arg("name"), py::arg("shape"),
               py::arg("mode_type") = ModeFormat::compressed)

          .def(py::init<std::string, std::vector<int>, Format>(), py::arg("name"), py::arg("shape"),
               py::arg("format"))

          .def("set_name", &TensorBase::setName)

          .def("get_name", &TensorBase::getName)

          .def_property_readonly("order", &TensorBase::getOrder)

          .def_property("name", &TensorBase::getName, &TensorBase::setName)

          .def("get_shape", &TensorBase::getDimension, py::arg("axis"))

          .def_property_readonly("shape", &TensorBase::getDimensions)

          .def_property_readonly("dtype", &TensorBase::getComponentType)

          .def_property_readonly("format", &TensorBase::getFormat)

          .def("pack", &typedTensor::pack)

          .def("compile", &typedTensor::compile)

          .def("assemble", &typedTensor::assemble)

          .def("evaluate", &typedTensor::evaluate)

          .def("compute", &typedTensor::compute)

          // Set and get for indices
          .def("__getitem__", [](typedTensor& self, const int &index) -> CType {
              checkBounds(self.getDimensions(), {index});
              return self.at({index});
            }, py::is_operator())

          .def("__getitem__", [](typedTensor& self, const std::vector<int> &indices) -> CType {
              checkBounds(self.getDimensions(), indices);
              return self.at(indices);
            }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, const int &index, py::object value) -> void {
              checkBounds(self.getDimensions(), {index});
              self.insert({index}, static_cast<CType>(value.cast<double>()));
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, const std::vector<int> &indices, py::object value) -> void {
              checkBounds(self.getDimensions(), indices);
              self.insert(indices, static_cast<CType>(value.cast<double>()));
          }, py::is_operator())

          // Get and set item for using index vars
          .def("__getitem__", [](typedTensor& self, py::none) -> Access {
              if(self.getOrder() != 0){
                throw py::index_error("Can only use None with scalar tensors");
              }
              return self();
          }, py::is_operator())

          .def("__getitem__", [](typedTensor& self, const IndexVar indexVars) -> Access {
              return self(indexVars);
          }, py::is_operator())

          .def("__getitem__", [](typedTensor& self, const std::vector<IndexVar> indexVars) -> Access {
              return self(indexVars);
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, py::none, const IndexExpr expr) -> void {
              self() = expr;
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, py::none, const Access access) -> void {
              self() = access;
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, py::none, const TensorVar tensorVar) -> void {
              self() = tensorVar;
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, const IndexVar indexVar, const IndexExpr expr) -> void {
              self(indexVar) = expr;
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, const IndexVar indexVar, const Access access) -> void {
              self(indexVar) = access;
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, const IndexVar indexVar, const TensorVar tensorVar) -> void {
              self(indexVar) = tensorVar;
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, const std::vector<IndexVar> indexVars, const IndexExpr expr) -> void {
              self(indexVars) = expr;
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, const std::vector<IndexVar> indexVars, const Access access) -> void {
              self(indexVars) = access;
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, const std::vector<IndexVar> indexVars, const TensorVar tensorVar) -> void {
              self(indexVars) = tensorVar;
          }, py::is_operator())

          .def("__repr__",   [](typedTensor& self) -> std::string{
              std::ostringstream o;
              o << self;
              return o.str();
          }, py::is_operator())


          // This is a hack that exploits pybind11's resolution order. If we get here all other methods to resolve the
          // function failed on both passes and we throw an error. There may be better was to handle this in pybind.
          .def("__getitem__", [](typedTensor& self, const py::object &indices) -> void {
            throw py::index_error("Indices must be an iterable of integers or IndexVars");
          }, py::is_operator())

          .def("__setitem__", [](typedTensor& self, const py::object &indices, py::object value) -> void {
             throw py::index_error("Indices must be an iterable of integers or IndexVars");
          }, py::is_operator());
}

void defineTensor(py::module &m){
  declareTensor<int8_t>(m, "Int8");
  declareTensor<int16_t>(m, "Int16");
  declareTensor<int32_t>(m, "Int32");
  declareTensor<int64_t>(m, "Int64");
  declareTensor<uint8_t>(m, "UInt8");
  declareTensor<uint16_t>(m, "UInt16");
  declareTensor<uint32_t>(m, "UInt32");
  declareTensor<uint64_t>(m, "UInt64");
  declareTensor<float>(m, "Float");
  declareTensor<double>(m, "Double");
}

}}


