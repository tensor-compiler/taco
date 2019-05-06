#ifndef TACO_PY_INDEX_NOTATION_H
#define TACO_PY_INDEX_NOTATION_H


#include "taco/tensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace taco{
namespace pythonBindings{

void defineIndexNotation(py::module &m);

}}


#endif //TACO_PY_INDEX_NOTATION_H
