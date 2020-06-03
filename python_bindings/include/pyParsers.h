#ifndef TACO_PYPARSERS_H
#define TACO_PYPARSERS_H

#include "taco/parser/parser.h"
#include "taco/parser/einsum_parser.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace taco{
namespace pythonBindings{

void defineParser(py::module& m);

}}

#endif //TACO_PYPARSERS_H
