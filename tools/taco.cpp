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
  Parser(Lexer lexer, map<string,Format> formats)
      : lexer(lexer), formats(formats) {
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

    Format format;
    if (util::contains(formats, tensorName)) {
      format = formats.at(tensorName);
    }
    else {
      Format::LevelTypes      levelTypes;
      Format::DimensionOrders dimensions;
      size_t order = varlist.size();
      for (size_t i = 0; i < order; i++) {
        // defaults
        levelTypes.push_back(LevelType::Dense);
        dimensions.push_back(i);
      }
      format = Format(levelTypes, dimensions);
    }

    vector<int> dimensionSizes;
    for (size_t i = 0; i < format.getLevels().size(); i++) {
      dimensionSizes.push_back(3);
    }
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
  map<string,Format> formats;

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
  const size_t descriptionStart = 15;
  const size_t columnEnd        = 80;
  string flagString = "  -" + flag +
                      util::repeat(" ",descriptionStart-(flag.size()+3));
  cout << flagString;
  size_t column = flagString.size();
  vector<string> words = util::split(text, " ");
  for (auto& word : words) {
    if (column + word.size()+1 >= columnEnd) {
      cout << endl << util::repeat(" ", descriptionStart);
      column = descriptionStart;
    }
    column += word.size()+1;
    cout << word << " ";
  }
  cout << endl;
}

static void printUsageInfo() {
  cout << "Usage: taco [options] <index expression>" << endl;
  cout << endl;
  cout << "Options:" << endl;
  printFlag("f=<format>",
            "Specify the format of a tensor in the expression. Formats are "
            "specified per dimension using d (dense) and s (sparse). "
            "All formats default to dense. "
            "Examples: A:ds, b:d and D:sss.");
  cout << endl;
  cout << "Options planned for the future:" << endl;
  printFlag("c", "Print compute IR (default).");
  cout << endl;
  printFlag("a", "Print assembly IR.");
  cout << endl;
  printFlag("g",
            "Generate random data for a given tensor. (e.g. B).");
  cout << endl;
  printFlag("i",
            "Initialize a tensor from an input file (e.g. B:\"myfile.txt\"). "
            "If all the tensors have been initialized then the expression is "
            "evaluated.");
  cout << endl;
  printFlag("o",
            "Write the result of evaluating the expression to the given file");
  cout << endl;
  printFlag("t", "Time compilation, assembly and computation.");
}

static int reportError(string errorMessage, int errorCode) {
  cerr << "Error: " << errorMessage << endl << endl;
  printUsageInfo();
  return errorCode;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printUsageInfo();
    return 1;
  }

  string exprStr;
  map<string,Format> formats;
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if ("-f=" == arg.substr(0,3)) {
      vector<string> descriptor = util::split(arg.substr(3,string::npos), ":");
      if (descriptor.size() != 2) {
        return reportError("Incorrect format descriptor", 3);
      }
      string tensorName   = descriptor[0];
      string formatString = descriptor[1];
      Format::LevelTypes      levelTypes;
      Format::DimensionOrders dimensions;
      for (size_t i = 0; i < formatString.size(); i++) {
        switch (formatString[i]) {
          case 'd':
            levelTypes.push_back(LevelType::Dense);
            break;
          case 's':
            levelTypes.push_back(LevelType::Sparse);
            break;
          default:
            return reportError("Incorrect format descriptor", 3);
            break;
        }
        dimensions.push_back(i);
        formats.insert({tensorName, Format(levelTypes, dimensions)});
      }
    }
    else {
      if (exprStr.size() != 0) {
        printUsageInfo();
        return 2;
      }
      exprStr = argv[i];
    }
  }

  internal::Tensor tensor;
  try {
    Lexer lexer(exprStr);
    Parser parser(lexer, formats);
    tensor = parser.parseAssign();
  } catch (ParseError& e) {
    cerr << e.msg << endl;
  }

  tensor.compile();
  tensor.printComputeIR(cout);

  return 0;
}
