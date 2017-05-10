#include "taco/parser/lexer.h"

#include "taco/error.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace parser {

Token Lexer::getToken() {
  while (isspace(lastChar)) {
    lastChar = getNextChar();
  }

  // identifiers
  if(isalpha(lastChar)) {
    identifier = lastChar;
    while (isalnum(lastChar = getNextChar())) {
      identifier += lastChar;
    }
    return Token::identifier;
  }
  if(isdigit(lastChar)) {
    identifier = lastChar;
    while (isdigit(lastChar = getNextChar())) {
      identifier += lastChar;
    }
    if (lastChar == '.') {
      identifier += lastChar;
      while (isdigit(lastChar = getNextChar())) {
        identifier += lastChar;
      }
    }
    return Token::scalar;
  }

  Token token;
  switch (lastChar) {
    case ',':
      token = Token::comma;
      break;
    case '(':
      token = Token::lparen;
      break;
    case ')':
      token = Token::rparen;
      break;
    case '_':
      token = Token::underscore;
      break;
    case '{':
      token = Token::lcurly;
      break;
    case '}':
      token = Token::rcurly;
      break;
    case '+':
      token = Token::add;
      break;
    case '-':
      token = Token::sub;
      break;
    case '*':
      token = Token::mul;
      break;
    case '=':
      token = Token::eq;
      break;
    case EOF:
      token = Token::eot;
      break;
    default:
      token = Token::error;
      break;
  }

  lastChar = getNextChar();
  return token;
}

std::string Lexer::getIdentifier() const {
  return identifier;
}

std::string Lexer::getLastChar() const {
  return util::toString(char(lastChar));
}

std::string Lexer::tokenString(const Token& token) {
  string str;
  switch (token) {
    case Token::identifier:
      str = "identifier";
      break;
    case Token::scalar:
      str = "scalar";
      break;
    case Token::comma:
      str = ",";
      break;
    case Token::lparen:
      str = "(";
      break;
    case Token::rparen:
      str = ")";
      break;
    case Token::underscore:
      str = "_";
    break;
    case Token::lcurly:
      str = "{";
    break;
    case Token::rcurly:
      str = "}";
    break;
    case Token::add:
      str = "+";
      break;
    case Token::sub:
      str = "-";
      break;
    case Token::mul:
      str = "*";
      break;
    case Token::eq:
      str = "=";
      break;
    case Token::error:
      str = "error";
      break;
    case Token::eot:
    default:
      taco_ierror;
      break;
  }

  return str;
}

int Lexer::getNextChar() {
  if (lastCharPos+1 == (int)expr.size()) {
    return EOF;
  }
  return expr[++lastCharPos];
}

}}
