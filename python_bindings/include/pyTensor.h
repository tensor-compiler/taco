#ifndef TACO_PY_TENSOR_H
#define TACO_PY_TENSOR_H

#include "taco/tensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace taco{
namespace pythonBindings{

void defineTensor(py::module& m);

}}

#endif //TACO_PY_TENSOR_H
