#include <python3.6/Python.h>
#include <pybind11/pybind11.h>
#include "pyFormat.h"
#include "pyDatatypes.h"
#include "pyIndexNotation.h"
#include "pyTensor.h"

PYBIND11_MODULE(pytaco, m){

  m.doc() = "A Python module for operating on Sparse Tensors.";
  using namespace taco::pythonBindings;
  defineTacoTypes(m);
  defineModeFormats(m);
  defineModeFormatPack(m);
  defineFormat(m);
  defineIndexNotation(m);
  defineTensor(m);

}
