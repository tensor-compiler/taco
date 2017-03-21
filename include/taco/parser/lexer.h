#ifndef TACO_LEXER_H
#define TACO_LEXER_H

#include <string>

namespace taco {
namespace parser {

enum class Token {
  identifier,
  scalar,
  comma,
  lparen,
  rparen,
  underscore,
  lcurly,
  rcurly,
  add,
  sub,
  mul,
  eq,
  eot,  // End of tokens
  error
};


// A simple index expression lexer.
class Lexer {
public:
  Lexer() {}
  Lexer(std::string expr) : expr(expr) {}

  /// Retrieve the next token.
  Token getToken();

  std::string getIdentifier() const;
  std::string getLastChar() const;

  /// Convert a token to a string.
  std::string tokenString(const Token& token);

private:
  std::string expr;
  int lastChar = ' ';
  int lastCharPos = -1;
  std::string identifier;

  int getNextChar();
};

}}
#endif
