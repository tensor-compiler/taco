#ifndef TACO_EINSUM_PARSER_H
#define TACO_EINSUM_PARSER_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <set>

#include "taco/tensor.h"
#include "taco/util/uncopyable.h"
#include "taco/type.h"

namespace taco {
class TensorBase;
namespace parser {

class EinsumParser : public util::Uncopyable {

public:

  /// Create a parser object from einsum notation
  /// @throws ParserError is there is an error with parsing the einsum string
  EinsumParser(const std::string &expression, std::vector<TensorBase> &tensors,
               Format &format, Datatype outType);

  /// Returns true if the expression passed in has an output specified and false otherwise
  /// @throws ParserError if output is not specified correctly
  bool exprHasOutput(const std::string& subscripts);

  /// Parses the einsum expression and sets the result tensor to the result of that expression
  /// @throws ParserError is there is an error with parsing the einsum stirng
  void parse();

  /// Gets the result tensor after parsing is complete.
  TensorBase& getResultTensor();

private:
  std::string einsumSymbols;
  std::set<char> einSumSymbolsSet;
  std::string einsumPunctuation;
  Datatype outType;
  Format format;


  std::string subscripts;
  TensorBase resultTensor;
  std::vector<TensorBase> &tensors;

  /// Replaces the ellipses in an expression
  std::string replaceEllipse(std::string inp, std::string &newString);

  /// Returns a sorted string of unique elements for a given einsum expression in implicit notation
  std::string findUniqueIndices(const std::string &subscripts);

  /// Builds the result tensor given the explicit input and output substrings
  void buildResult(std::vector<std::string> inputAndOutput);

  /// splits a string by , but keeps the empty string
  std::vector<std::string> splitSubscriptInput(std::string &inp);

};


}}

#endif //TACO_EINSUM_PARSER_H
