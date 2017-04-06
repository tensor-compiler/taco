#ifndef TACO_PARSER_H
#define TACO_PARSER_H

#include <string>
#include <memory>
#include <vector>
#include <map>

#include "taco/util/uncopyable.h"

namespace taco {

class TensorBase;
class Format;
class Var;
class Expr;
struct Read;

namespace parser {
enum class Token;


/// A simple index expression parser. The parser can parse an index expression
/// string, where tensor access expressions are in the form (e.g.) `A(i,j)`,
/// A_{i,j} or A_i. A variable is taken to be free if it is used to index the
/// lhs, and taken to be a summation variable otherwise.
class Parser : public util::Uncopyable {
public:
  Parser(std::string expression, const std::map<std::string,Format>& formats,
         const std::map<std::string,std::vector<int>>& dimensions,
         const std::map<std::string,TensorBase>& tensors,
         int dimensionDefault=5);

  /// Parse the expression.
  /// @throws ParseError if there's a parser error
  void parse();

  /// Returns the result (lhs) tensor of the index expression.
  const TensorBase& getResultTensor() const;

  /// Returns true if the index variable appeared in the expression
  bool hasIndexVar(std::string name) const;

  /// Retrieve the index variable with the given name
  Var getIndexVar(std::string name) const;

  /// Returns true if the tensor appeared in the expression
  bool hasTensor(std::string name) const;

  /// Retrieve the tensor with the given name
  const TensorBase& getTensor(std::string name) const;

  /// Retrieve a map from tensor names to tensors.
  const std::map<std::string,TensorBase>& getTensors() const;

private:
  struct Content;
  std::shared_ptr<Content> content;

  /// assign ::= access '=' compute
  TensorBase parseAssign();

  /// expr ::= term {('+' | '-') term}
  Expr parseExpr();

  /// term ::= factor {'*' factor}
  Expr parseTerm();

  /// factor ::= final 
  ///          | '(' expr ')'
  Expr parseFactor();

  /// final ::= access 
  ///         | scalar
  Expr parseFinal();

  /// access ::= identifier '(' varlist ')'
  ///          | identifier '_' '{' varlist '}'
  ///          | identifier '_' var
  ///          | identifier
  Read parseAccess();

  /// varlist ::= var {, var}
  std::vector<Var> parseVarList();

  /// var ::= identifier
  Var parseVar();

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
