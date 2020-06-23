#include "../include/pyTensorIO.h"
#include "pybind11/stl.h"

namespace taco{
namespace pythonBindings {

template<typename T>
static Tensor<double> tensorRead(std::string filename, T modeType, bool pack = true){
  return Tensor<double>(taco::read(filename, modeType, pack));
}

void defineIOFuncs(py::module &m){
  m.def("_read", tensorRead<Format>, py::arg("filename"), py::arg("format").noconvert(),
          py::arg("pack")=true);

  m.def("_read", tensorRead<ModeFormat>, py::arg("filename"), py::arg("modeType").noconvert(),
          py::arg("pack")=true);

  m.def("_write",[](std::string s, TensorBase& t) -> void {
    // force tensor evaluation
    t.pack();
    if(t.needsCompute()) {
      t.evaluate();
    }
    write(s, t);
  }, py::arg("filename"), py::arg("tensor").noconvert());
}

}}
