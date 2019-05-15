#ifndef TACO_PYINDEXNOTATION_H
#define TACO_PYINDEXNOTATION_H


#include "taco/tensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace taco{
namespace pythonBindings{

void defineIndexNotation(py::module &m);

}}


#endif //TACO_PYINDEXNOTATION_H
