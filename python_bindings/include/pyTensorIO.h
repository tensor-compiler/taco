#ifndef TACO_PYTENSORIO_H
#define TACO_PYTENSORIO_H

#include "taco/tensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace taco{
namespace pythonBindings{

void defineIOFuncs(py::module &m);


}}

#endif //TACO_PYTENSORIO_H
