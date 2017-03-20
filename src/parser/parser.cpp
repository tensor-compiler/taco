#include "taco/parser/parser.h"

#include "taco/parser/lexer.h"
#include "taco/tensor_base.h"
#include "taco/format.h"
#include "taco/expr.h"
#include "taco/operator.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace parser {

struct Parser::Content {
  /// Tensor formats
  map<string,Format> formats;

  TensorBase resultTensor;

  Lexer lexer;
  Token currentToken;
  bool parsingLhs = false;
  map<string,Var> indexVars;
  map<string,TensorBase> tensors;
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

const TensorBase& Parser::getResultTensor() const {
  return content->resultTensor;
}

TensorBase Parser::parseAssign() {
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
        taco_unreachable;
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
        taco_unreachable;
    }
  }
  return term;
}

Expr Parser::parseFactor() {
  if (content->currentToken == Token::lparen) {
    consume(Token::lparen);
    Expr factor = parseExpr();
    consume(Token::rparen);
    return factor;
  }
  return parseFinal();
}

Expr Parser::parseFinal() {
  if(content->currentToken == Token::scalar) {
    string value=content->lexer.getIdentifier();
    consume(Token::scalar);
    return Expr(atof(value.c_str()));
  }
  else
    return parseAccess();
}

Read Parser::parseAccess() {
  if(content->currentToken != Token::identifier) {
    throw ParseError("Expected tensor name");
  }
  string tensorName = content->lexer.getIdentifier();
  consume(Token::identifier);

  vector<Var> varlist;
  if (content->currentToken == Token::underscore) {
    consume(Token::underscore);
    if (content->currentToken == Token::lcurly) {
      consume(Token::lcurly);
      varlist = parseVarList();
      consume(Token::rcurly);
    }
    else {
      varlist.push_back(parseVar());
    }
  }
  else {
    consume(Token::lparen);
    varlist = parseVarList();
    consume(Token::rparen);
  }

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
  TensorBase tensor(tensorName, ComponentType::Double,
                    dimensionSizes, format, DEFAULT_ALLOC_SIZE);
  content->tensors.insert({tensorName,tensor});
  return Read(tensor, varlist);
}

vector<Var> Parser::parseVarList() {
  vector<Var> varlist;
  varlist.push_back(parseVar());
  while (content->currentToken == Token::comma) {
    consume(Token::comma);
    varlist.push_back(parseVar());
  }
  return varlist;
}

Var Parser::parseVar() {
  if (content->currentToken != Token::identifier) {
    throw ParseError("Expected index variable");
  }
  Var var = getIndexVar(content->lexer.getIdentifier());
  consume(Token::identifier);
  return var;
}

bool Parser::hasIndexVar(std::string name) const {
  return util::contains(content->indexVars, name);
}

Var Parser::getIndexVar(string name) const {
  taco_iassert(name != "");
  if (!hasIndexVar(name)) {
    Var var(name, (content->parsingLhs ? Var::Free : Var::Sum));
    content->indexVars.insert({name, var});
  }
  return content->indexVars.at(name);
}

bool Parser::hasTensor(std::string name) const {
  return util::contains(content->tensors, name);
}

const TensorBase& Parser::getTensor(string name) const {
  taco_iassert(name != "");
  if (!hasTensor(name)) {
    taco_uerror << "Parser error: Tensor name " << name <<
        " not found in expression" << endl;
  }
  return content->tensors.at(name);
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
