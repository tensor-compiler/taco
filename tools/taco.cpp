#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "tensor.h"
#include "internal_tensor.h"
#include "expr.h"
#include "operator.h"
#include "error.h"
#include "util/strings.h"

using namespace std;
using namespace taco;

/// Lexer
enum class Token {
  identifier,
  comma,
  lparen,
  rparen,
  add,
  sub,
  mul,
  eq,
  eot,  // End of tokens
  error
};

class Lexer {
public:
  Lexer(string expr) : expr(expr) {}

  Token getToken() {
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

  string getIdentifier() const {
    return identifier;
  }

  string getLastChar() {
    return util::toString(char(lastChar));
  }

private:
  string expr;
  int lastChar = ' ';
  int lastCharPos = -1;
  string identifier;

  int getNextChar() {
    if (lastCharPos+1 == (int)expr.size()) {
      return EOF;
    }
    return expr[++lastCharPos];
  }
};


/// Parser
struct ParseError {
  ParseError(string msg) : msg(msg) {}
  string msg;
};

class Parser {
public:
  Parser(Lexer lexer) : lexer(lexer) {
    nextToken();
  }

  /// varlist ::= var {, var}
  vector<Var> parseVarList() {
    if (currentToken != Token::identifier) {
      throw ParseError("Expected index variable");
    }

    vector<Var> varlist;

    varlist.push_back(getIndexVar(lexer.getIdentifier()));
    consume(Token::identifier);

    while (currentToken == Token::comma) {
      consume(Token::comma);
      if (currentToken != Token::identifier) {
        throw ParseError("Expected index variable");
      }
      varlist.push_back(getIndexVar(lexer.getIdentifier()));
      consume(Token::identifier);
    }

    return varlist;
  }

  /// access ::= identifier '(' varlist ')'
  Read parseAccess() {
    if(currentToken != Token::identifier) {
      throw ParseError("Expected tensor name");
    }
    string tensorName = lexer.getIdentifier();
    consume(Token::identifier);
    consume(Token::lparen);
    vector<Var> varlist = parseVarList();
    consume(Token::rparen);

    Format::LevelTypes      levelTypes;
    Format::DimensionOrders dimensions;
    vector<int>                   dimensionSizes;
    size_t order = varlist.size();
    for (size_t i = 0; i < order; i++) {
      // defaults
      levelTypes.push_back(LevelType::Dense);
      dimensions.push_back(i);
      dimensionSizes.push_back(3);

    }
    Format format(levelTypes, dimensions);
    internal::Tensor tensor(tensorName, dimensionSizes,
                                  format, internal::ComponentType::Double,
                                  DEFAULT_ALLOC_SIZE);
    return Read(tensor, varlist);
  }

  /// factor ::= access
  Expr parseFactor() {
    return parseAccess();
  }

  /// term ::= factor {'*' factor}
  Expr parseTerm() {
    Expr term = parseFactor();
    while (currentToken == Token::mul) {
      switch (currentToken) {
        case Token::mul:
          consume(Token::mul);
          term = term * parseFactor();
          break;
        default:
          unreachable;
      }
    }
    return term;
  }

  /// expr ::= term {('+' | '-') term}
  Expr parseExpr() {
    Expr expr = parseTerm();
    while (currentToken == Token::add || currentToken == Token::sub) {
      switch (currentToken) {
        case Token::add:
          consume(Token::add);
          expr = expr + parseTerm();
          break;
        case Token::sub:
          consume(Token::sub);
          expr = expr - parseTerm();
          break;
        default:
          unreachable;
      }
    }
    return expr;
  }

  /// assign ::= access '=' compute
  internal::Tensor parseAssign() {
    parsingLhs = true;
    Read lhs = parseAccess();
    parsingLhs = false;
    consume(Token::eq);
    Expr rhs = parseExpr();
    lhs = rhs;
    return lhs.getTensor();
  }

private:
  Lexer lexer;
  Token currentToken;

  bool parsingLhs = false;
  map<string,Var> indexVars;
  Var getIndexVar(string name) {
    if (!util::contains(indexVars, name)) {
      indexVars.insert({name, Var(name, (parsingLhs ? Var::Free : Var::Sum))});
    }
    return indexVars.at(name);
  }

  void nextToken() {
    currentToken = lexer.getToken();
  }

  string tokenString(Token token) {
    string str;
    switch (token) {
      case Token::identifier:
        str = "identifier";
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
        ierror;
        break;
    }

    return str;
  }

  string currentTokenString() {
    return (currentToken == Token::identifier) ? lexer.getIdentifier()
                                               : tokenString(currentToken);
  }

  void consume(Token expected) {
    if(currentToken != expected) {
      string error = "Expected \'" + tokenString(expected) + "\' but got \'" +
                     currentTokenString() + "\'";
      throw ParseError(error);
    }
    nextToken();
  }
};



static void printFlag(string flag, string text) {
  // TODO: Clean print with nice text wrapping
  cout << "  -" << flag << "          " << text << endl;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "Usage: taco [options] <index expression>" << endl;
    cout << endl;
    cout << "Options:" << endl;
    printFlag("f",
              "Specify the format of a tensor in the expression. Formats are "
              "specified per dimension using D (dense) and S (sparse). "
              "For example, A:DS, b:D and D:SSS. "
              "All formats default to dense.");
    return 1;
  }

  string exprStr = argv[1];

  internal::Tensor tensor;
  try {
    Lexer lexer(exprStr);
    Parser parser(lexer);
    tensor = parser.parseAssign();
  } catch (ParseError& e) {
    cerr << e.msg << endl;
  }

  tensor.compile();
  tensor.printComputeIR(cout);

  return 0;
}
