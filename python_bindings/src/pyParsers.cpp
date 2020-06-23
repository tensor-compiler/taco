#include <taco.h>
#include <taco/parser/lexer.h>
#include "pyParsers.h"
#include "pybind11/stl.h"

namespace py = pybind11;


namespace taco{

namespace pythonBindings{

static std::vector<std::string> extractNames(std::string& expr){

  parser::Parser nameGetter(expr, {}, {}, {}, {});
  try {
    nameGetter.parse();
  } catch(parser::ParseError & e) {
    throw py::value_error(e.getMessage());
  }
  return nameGetter.getNames();
}

static void resetNames(std::vector<std::string> oldNames, py::list &tensors){
  for(size_t i = 0; i < tensors.size(); ++i) {
    auto cpptensor = tensors[i];
    auto tensor = cpptensor.cast<TensorBase&>();
    tensor.setName(oldNames[i]);
  }
}

static TensorBase parseString(std::string& expr, py::list &tensors, py::object& fmt, Datatype dtype){

  auto tensorNames = extractNames(expr);
  bool found_output = true;
  std::map<std::string, Format> nameFormat;
  std::map<std::string, Datatype> nameDtype;
  std::map<std::string, std::vector<int>> nameDims;
  std::map<std::string, TensorBase> nameTensor;
  if(tensorNames.size() - 1 == tensors.size()){
    found_output = false;
    if(!fmt.is_none()) {
      nameFormat.insert({tensorNames[0], fmt.cast<Format>()});
    }
    nameDtype.insert({tensorNames[0], dtype});
    // Remove first tensor name from list to avoid adding it to the datastructures for the parser
    tensorNames = std::vector<std::string>(tensorNames.begin() + 1, tensorNames.end());
  }

  if(tensorNames.size() != tensors.size()) {
    throw py::value_error("The number of tensors in the expression and the number of tensors given mismatch.");
  }

  for(size_t i = 0; i < tensors.size(); ++i) {
    auto cpptensor = tensors[i];
    auto tensor = cpptensor.cast<TensorBase&>();
    std::string temp = tensor.getName();
    tensor.setName(tensorNames[i]);
    tensorNames[i] = temp;

    nameFormat.insert({tensor.getName(), tensor.getFormat()});
    nameDtype.insert({tensor.getName(), tensor.getComponentType()});
    nameDims.insert({tensor.getName(), tensor.getDimensions()});
    nameTensor.insert({tensor.getName(), tensor});
  }

  parser::Parser tensor_parser(expr, nameFormat, nameDtype, nameDims, nameTensor);
  try {
    tensor_parser.parse();
  } catch (const parser::ParseError& e){
    resetNames(tensorNames, tensors);
    throw py::value_error(e.getMessage());
  }

  resetNames(tensorNames, tensors);

  TensorBase result = tensor_parser.getResultTensor();
  if (found_output) {
    return tensors[0].cast<TensorBase>();
  }

  return result;
}

static TensorBase einsumParse(std::string& expr, py::list &tensors, py::object& fmt, Datatype dtype) {
  std::vector<TensorBase> cppTensors;
  for(auto &tensor: tensors){
    cppTensors.push_back(tensor.cast<TensorBase>());
  }

  Format format = fmt.is_none()? Format() : fmt.cast<Format>();
  parser::EinsumParser einsumParser(expr, cppTensors, format, dtype);
  try {
    einsumParser.parse();
  } catch (const parser::ParseError& e){
    throw py::value_error(e.getMessage());
  }

  TensorBase result = einsumParser.getResultTensor();
  return result;
}

void defineParser(py::module& m) {
  m.def("_parse", &parseString);
  m.def("_einsum", &einsumParse);
}

}}
