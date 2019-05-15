#ifndef TACO_DATATYPES_H
#define TACO_DATATYPES_H
#include "taco/type.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace taco{
namespace pythonBindings{

std::string getNpType(const taco::Datatype &type);
py::object asNpDtype(const taco::Datatype& obj);
void defineTacoTypes(py::module &m);

}}

#endif //TACO_DATATYPES_H
