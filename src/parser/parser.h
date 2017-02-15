#ifndef TACO_PARSER_H
#define TACO_PARSER_H

#include <string>
#include <memory>
#include <vector>
#include <map>

#include "util/uncopyable.h"

namespace taco {

class Format;
class Var;
class Expr;
struct Read;

namespace internal {
class Tensor;
}

namespace parser {
enum class Token;


/// A simple index expression parser. The parser can parse an index expression
/// string, where tensor access expressions are in the form (e.g.) `A(i,j)`.
/// A variable is taken to be free if it is used to index the lhs, and taken
/// to be a summation variable otherwise.
/// TODO: Support latex tensor index form `A_{i,j}`.
class Parser : public util::Uncopyable {
public:
  Parser(std::string expression, const std::map<std::string,Format>& formats);

  /// Parse the expression.
  /// @throws ParseError if there's a parser error
  void parse();

  /// Returns the result (lhs) tensor of the index expression.
  const internal::Tensor& getResultTensor() const;

  /// Retrieve the index variable with the given name
  Var getIndexVar(std::string name);

private:
  struct Content;
  std::shared_ptr<Content> content;

  /// assign ::= access '=' compute
  internal::Tensor parseAssign();

  /// expr ::= term {('+' | '-') term}
  Expr parseExpr();

  /// term ::= factor {'*' factor}
  Expr parseTerm();

  /// factor ::= access
  Expr parseFactor();

  /// access ::= identifier '(' varlist ')'
  Read parseAccess();

  /// varlist ::= var {, var}
  std::vector<Var> parseVarList();

  std::string currentTokenString();

  void consume(Token expected);

  /// Retrieve the next token from the lexer
  void nextToken();
};


/// An error that occurred during parsing. Thrown by the Parse::parse method. 
class ParseError {
public:
  ParseError(std::string msg) : msg(msg) {}
  std::string getMessage() const {return msg;}

private:
  std::string msg;

};

}}

#endif
