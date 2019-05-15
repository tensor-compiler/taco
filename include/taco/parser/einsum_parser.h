#ifndef TACO_EINSUM_PARSER_H
#define TACO_EINSUM_PARSER_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <set>

#include "taco/util/uncopyable.h"
#include "taco/type.h"

namespace taco {
class TensorVar;
namespace parser {

class EinsumParser : public util::Uncopyable {


public:
  /// Create a parser object from einsum notation
  /// @throws ParserError is there is an error with parsing the einsum string
  EinsumParser(const std::string &expression, const std::vector<TensorVar> &tensors);

  /// Returns num_unused_symbols valid numpy einsum symbols that are not in used_symbols
  static std::vector<std::string> genUnusedSymbols(const std::set<char> &usedSymbols, int numUnusedSymbolsNeeded);

  /// Returns the output string for a given einsum expression in implicit notation
  static std::string findOutputString(const std::string &subscripts);

  /// Returns true if the expression passed in has an output specified and false otherwise
  /// @throws ParserError if output is not specified correctly
  static bool exprHasOutput(const std::string &subscripts);

  /// Converts an operand to taco notation
  /// @throws ParserError if ellipses incorrectly specified.
  static std::string convertToIndexExpr(const std::string &subscripts, const std::string &ellipsisReplacement,
                                        const std::string &tensorName);

  /// Replaces ellipsis in string with valid indices and returns a vector with the result tensor expression last
  /// and the input tensors preceding the result tensor in a vector.
  /// @throws ParserError is there is an error with parsing the einsum stirng
  static std::vector<std::string> parseToTaco(const std::string &subscripts, const std::vector<TensorVar>& tensors);

};


}}

#endif //TACO_EINSUM_PARSER_H
