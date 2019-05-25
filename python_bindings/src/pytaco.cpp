#include <Python.h>
#include <pybind11/pybind11.h>
#include "pyFormat.h"
#include "pyDatatypes.h"
#include "pyIndexNotation.h"
#include "pyTensor.h"
#include "pyTensorIO.h"
#include "pyParsers.h"


void addHelpers(py::module &m) {
  m.def("taco_get_num_threads", &taco::taco_get_num_threads);
  m.def("taco_set_num_threads", &taco::taco_set_num_threads, py::arg("num_threads"));
  m.def("unique_name", (std::string(*)(char)) &taco::util::uniqueName);
}

PYBIND11_MODULE(core_modules, m){

  m.doc() = "A Python module for operating on Sparse Tensors.";
  using namespace taco::pythonBindings;
  addHelpers(m);
  defineTacoTypes(m);
  defineModeFormats(m);
  defineModeFormatPack(m);
  defineFormat(m);
  defineIndexNotation(m);
  defineTensor(m);
  defineIOFuncs(m);
  defineParser(m);

}

