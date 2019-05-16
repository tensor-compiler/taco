#include <python3.6/Python.h>
#include <pybind11/pybind11.h>
#include "pyFormat.h"
#include "pyDatatypes.h"
#include "pyIndexNotation.h"
#include "pyTensor.h"
#include "pyTensorIO.h"

PYBIND11_MODULE(pytaco, m){

  m.doc() = "A Python module for operating on Sparse Tensors.";
  using namespace taco::pythonBindings;
  m.def("get_taco_num_threads", &taco::get_taco_num_threads);
  m.def("set_taco_num_threads", &taco::set_taco_num_threads, py::arg("num_threads"));
  defineTacoTypes(m);
  defineModeFormats(m);
  defineModeFormatPack(m);
  defineFormat(m);
  defineIndexNotation(m);
  defineTensor(m);
  defineIOFuncs(m);

}
