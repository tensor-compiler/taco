#ifndef TACO_LINALG_PARSER_H
#define TACO_LINALG_PARSER_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <set>

#include "taco/tensor.h"
#include "taco/util/uncopyable.h"
#include "taco/type.h"
#include "taco/parser/parser.h"

namespace taco {
class TensorBase;
class Format;
class IndexVar;
class IndexExpr;
class Access;

namespace parser {
enum class Token;

class LinalgParser : public AbstractParser {

public:

  /// Create a parser object from linalg notation
  LinalgParser(std::string expression, const std::map<std::string,Format>& formats,
  const std::map<std::string, Datatype>& dataTypes,
  const std::map<std::string,std::vector<int>>& tensorDimensions,
  const std::map<std::string,TensorBase>& tensors,
  int defaultDimension=5);

  /// Parses the linalg expression and sets the result tensor to the result of that expression
  /// @throws ParserError if there is an error with parsing the linalg string
  void parse() override;

  /// Gets the result tensor after parsing is complete.
  const TensorBase& getResultTensor() const override;

  /// Gets all tensors
  const std::map<std::string,TensorBase>& getTensors() const override;

  /// Retrieve the tensor with the given name
  const TensorBase& getTensor(std::string name) const override;

  /// Returns true if the tensor appeared in the expression
  bool hasTensor(std::string name) const;

  /// Returns true if the index variable appeared in the expression
  bool hasIndexVar(std::string name) const;

  /// Retrieve the index variable with the given name
  IndexVar getIndexVar(std::string name) const;

private:
  Datatype outType;
  Format format;

  struct Content;
  std::shared_ptr<Content> content;
  std::vector<std::string> names;

  /// assign ::= var '=' expr
  TensorBase parseAssign();

  /// expr ::= term {('+' | '-') term}
  IndexExpr parseExpr();

  /// term ::= factor {('*' | '/') factor}
  IndexExpr parseTerm();

  /// factor ::= final
  ///          | '(' expr ')'
  ///          | '-' factor
  ///          | factor '^T'
  IndexExpr parseFactor();

  /// final ::= var
  ///         | scalar
  IndexExpr parseFinal();

  /// var ::= identifier
  Access parseVar();

  std::string currentTokenString();

  void consume(Token expected);

  /// Retrieve the next token from the lexer
  void nextToken();

  /// FIXME: REMOVE LATER, temporary workaround to use Tensor API and TensorBase
  std::vector<IndexVar> getUniqueIndices(size_t order);

  int idxcount;
};


}}
#endif //TACO_LINALG_PARSER_H
