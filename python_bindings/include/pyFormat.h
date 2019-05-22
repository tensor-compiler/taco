#ifndef TACO_PYFORMAT_H
#define TACO_PYFORMAT_H

#include "taco/format.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace taco{
namespace pythonBindings{

std::size_t hashFormat(const taco::Format& format);
void defineModeFormats(py::module& m);
void defineModeFormatPack(py::module &m);
void defineFormat(py::module &m);

}}

#endif //TACO_PYFORMAT_H
