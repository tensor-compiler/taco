#include "parser.h"

#include "var.h"
#include "expr.h"
#include "tensor.h"
#include "internal_tensor.h"
#include "operator.h"
#include "lexer.h"
#include "format.h"
#include "util/collections.h"

using namespace std;

namespace taco {
namespace parser {

struct Parser::Content {
  /// Tensor formats
  map<string,Format> formats;

  internal::Tensor resultTensor;

  Lexer lexer;
  Token currentToken;
  bool parsingLhs = false;
  map<string,Var> indexVars;
};

Parser::Parser(string expression, const map<string,Format>& formats)
    : content(new Parser::Content) {
  content->lexer = Lexer(expression);
  content->formats = formats;
  nextToken();
}

void Parser::parse() {
  content->resultTensor = parseAssign();
}

const internal::Tensor& Parser::getResultTensor() const {
  return content->resultTensor;
}

internal::Tensor Parser::parseAssign() {
  content->parsingLhs = true;
  Read lhs = parseAccess();
  content->parsingLhs = false;
  consume(Token::eq);
  Expr rhs = parseExpr();
  lhs = rhs;
  return lhs.getTensor();
}

Expr Parser::parseExpr() {
  Expr expr = parseTerm();
  while (content->currentToken == Token::add ||
         content->currentToken == Token::sub) {
    switch (content->currentToken) {
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

Expr Parser::parseTerm() {
  Expr term = parseFactor();
  while (content->currentToken == Token::mul) {
    switch (content->currentToken) {
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

Expr Parser::parseFactor() {
  return parseAccess();
}

Read Parser::parseAccess() {
  if(content->currentToken != Token::identifier) {
    throw ParseError("Expected tensor name");
  }
  string tensorName = content->lexer.getIdentifier();
  consume(Token::identifier);
  consume(Token::lparen);
  vector<Var> varlist = parseVarList();
  consume(Token::rparen);

  Format format;
  if (util::contains(content->formats, tensorName)) {
    format = content->formats.at(tensorName);
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

vector<Var> Parser::parseVarList() {
  if (content->currentToken != Token::identifier) {
    throw ParseError("Expected index variable");
  }

  vector<Var> varlist;

  varlist.push_back(getIndexVar(content->lexer.getIdentifier()));
  consume(Token::identifier);

  while (content->currentToken == Token::comma) {
    consume(Token::comma);
    if (content->currentToken != Token::identifier) {
      throw ParseError("Expected index variable");
    }
    varlist.push_back(getIndexVar(content->lexer.getIdentifier()));
    consume(Token::identifier);
  }

  return varlist;
}

Var Parser::getIndexVar(string name) {
  if (!util::contains(content->indexVars, name)) {
    Var var(name, (content->parsingLhs ? Var::Free : Var::Sum));
    content->indexVars.insert({name, var});
  }
  return content->indexVars.at(name);
}


string Parser::currentTokenString() {
  return (content->currentToken == Token::identifier)
      ? content->lexer.getIdentifier()
      : content->lexer.tokenString(content->currentToken);
}

void Parser::consume(Token expected) {
  if(content->currentToken != expected) {
    string error = "Expected \'" + content->lexer.tokenString(expected) +
                   "\' but got \'" + currentTokenString() + "\'";
    throw ParseError(error);
  }
  nextToken();
}

void Parser::nextToken() {
  content->currentToken = content->lexer.getToken();
}

}}
