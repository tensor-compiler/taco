#include "pytensor.h"
#include "taco/tensor.h"
#include "taco/type.h"

// Add Python dictionary initializer with {tuple(coordinate) : data} pairs


namespace taco{
namespace pythonBindings{

template<typename CType>
void declareTensor(py::module &m, std::string &typestr) {
  using TypedTensor = taco::Tensor<CType>;

  std::string pyClassName = std::string("Tensor") + typestr;
  py::class_<TypedTensor, taco::TensorBase>(m, pyClassName.c_str())

          .def(py::init<>())

          .def(py::init<std::string>, py::arg("name"))

          .def(py::init<CType>, py::arg("value"))

          .def(py::init<std::vector<int>, ModeFormat>, py::arg("dimensions"),
               py::arg("mode_type") = ModeFormat::compressed)

          .def(py::init<std::vector<int>, Format>, py::arg("dimensions"), py::arg("format"))

          .def(py::init<std::string, std::vector<int>, ModeFormat>, py::arg("name"), py::arg("dimensions"),
               py::arg("mode_type") = ModeFormat::compressed)

          .def(py::init<std::string, std::vector<int>, Format>, py::arg("name"), py::arg("dimensions"),
               py::arg("format"))

          .def("set_name", &TensorBase::setName)

          .def("get_name", &TensorBase::getName)

          .def("order", &TensorBase::getOrder)

          .def_property("name", &TensorBase::getName, &TensorBase::setName)

          .def("get_shape", &TensorBase::getDimension, py::arg("axis"))

          .def_property_readonly("shape", &TensorBase::getDimensions)

          .def_property_readonly("dtype", &TensorBase::getComponentType)

          .def_property_readonly("format", &TensorBase::getFormat)


          .def("__repr__",   [](const taco::Tensor<CType>& self) -> std::string{
              std::ostringstream o;
              o << "pytaco.Tensor("<<self.getFormat()<<"," << self.getComponentType();
              return o.str();
          }, py::is_operator());
}

}}


