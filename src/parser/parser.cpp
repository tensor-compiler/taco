#include "taco/parser/parser.h"

#include "taco/parser/lexer.h"
#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/expr.h"

#include "taco/expr_nodes/expr_nodes.h"
#include "taco/expr_nodes/expr_rewriter.h"

#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace parser {

struct Parser::Content {
  /// Tensor formats
  map<string,Format> formats;

  /// Tensor dimensions
  map<string,std::vector<int>> dimensionSizes;
  map<IndexVar, int>           indexVarSizes;

  int dimensionDefault;

  /// Track which dimensions has default values, so that we can change them
  /// to values inferred from other tensors (read from files).
  set<pair<TensorBase,size_t>> defaultDimension;

  Lexer lexer;
  Token currentToken;
  bool parsingLhs = false;

  map<string,IndexVar> indexVars;

  TensorBase             resultTensor;
  map<string,TensorBase> tensors;
};

Parser::Parser(string expression, const map<string,Format>& formats,
               const map<string,std::vector<int>>& dimensionSizes,
               const std::map<std::string,TensorBase>& tensors,
               int dimensionDefault)
    : content(new Parser::Content) {
  content->lexer = Lexer(expression);
  content->formats = formats;
  content->dimensionSizes = dimensionSizes;
  content->dimensionDefault = dimensionDefault;
  content->tensors = tensors;
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
  Access lhs = parseAccess();
  content->parsingLhs = false;
  consume(Token::eq);
  IndexExpr rhs = parseExpr();

  // Collect all index var dimension sizes
  struct Visitor : expr_nodes::ExprVisitor {
    using ExprVisitor::visit;
    set<pair<TensorBase,size_t>> defaultDimension;
    map<IndexVar, int>* indexVarSizes;

    void visit(const expr_nodes::ReadNode* op) {
      for (size_t i = 0; i < op->indexVars.size(); i++) {
        IndexVar indexVar = op->indexVars[i];
        if (!util::contains(defaultDimension, {op->tensor,i})) {
          int dimension = op->tensor.getDimension(i);
          if (util::contains(*indexVarSizes, indexVar)) {
            taco_uassert(indexVarSizes->at(indexVar) == dimension) <<
                "Incompatible dimensions";
          }
          else {
            indexVarSizes->insert({indexVar, dimension});
          }
        }
      }
    }
  };
  Visitor visitor;
  visitor.indexVarSizes = &content->indexVarSizes;
  visitor.defaultDimension = content->defaultDimension;
  rhs.accept(&visitor);

  // Rewrite expression to new index sizes
  struct Rewriter : expr_nodes::ExprRewriter {
    using ExprRewriter::visit;
    map<IndexVar, int>* indexVarSizes;
    map<string,TensorBase> tensors;

    void visit(const expr_nodes::ReadNode* op) {
      bool dimensionChanged = false;
      vector<int> dimensions = op->tensor.getDimensions();

      taco_uassert(op->indexVars.size() == dimensions.size()) <<
          "The order of " << op->tensor.getName() << " is inconsistent " <<
          "between tensor accesses or options. Is it order " <<
          dimensions.size() << " or " << op->indexVars.size() << "?";

      for (size_t i=0; i < dimensions.size(); i++) {
        IndexVar indexVar = op->indexVars[i];
        if (util::contains(*indexVarSizes, indexVar)) {
          int dimSize = indexVarSizes->at(indexVar);
          if (dimSize != dimensions[i]) {
            dimensions[i] = dimSize;
            dimensionChanged = true;
          }
        }
      }
      if (dimensionChanged) {
        TensorBase tensor;
        if (util::contains(tensors, op->tensor.getName())) {
          tensor = tensors.at(op->tensor.getName());
        }
        else {
          tensor = TensorBase(op->tensor.getName(),
                              op->tensor.getComponentType(), dimensions,
                              op->tensor.getFormat());
          tensors.insert({tensor.getName(), tensor});
        }
        expr = new expr_nodes::ReadNode(tensor, op->indexVars);
      }
      else {
        expr = op;
      }
    }
  };
  Rewriter rewriter;
  rewriter.indexVarSizes = visitor.indexVarSizes;
  rhs = rewriter.rewrite(rhs);

  IndexExpr rewrittenLhs = rewriter.rewrite(lhs);

  for (auto& tensor : rewriter.tensors) {
    content->tensors.at(tensor.first) = tensor.second;
  }
  content->resultTensor = content->tensors.at(lhs.getTensor().getName());
  content->resultTensor.setExpr(lhs.getIndexVars(), rhs);
  return content->resultTensor;
}

IndexExpr Parser::parseExpr() {
  IndexExpr expr = parseTerm();
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

IndexExpr Parser::parseTerm() {
  IndexExpr term = parseFactor();
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

IndexExpr Parser::parseFactor() {
  switch (content->currentToken) {
    case Token::lparen: {
      consume(Token::lparen);
      IndexExpr factor = parseExpr();
      consume(Token::rparen);
      return factor;
    }
    case Token::sub:
      consume(Token::sub);
      return new expr_nodes::NegNode(parseFactor());
    default:
      break;
  }
  return parseFinal();
}

IndexExpr Parser::parseFinal() {
  if(content->currentToken == Token::scalar) {
    string value=content->lexer.getIdentifier();
    consume(Token::scalar);
    return IndexExpr(atof(value.c_str()));
  }
  else
    return parseAccess();
}

Access Parser::parseAccess() {
  if(content->currentToken != Token::identifier) {
    throw ParseError("Expected tensor name");
  }
  string tensorName = content->lexer.getIdentifier();
  consume(Token::identifier);

  vector<IndexVar> varlist;
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
  else if (content->currentToken == Token::lparen) {
    consume(Token::lparen);
    varlist = parseVarList();
    consume(Token::rparen);
  }

  size_t order = varlist.size();

  Format format;
  if (util::contains(content->formats, tensorName)) {
    format = content->formats.at(tensorName);
  }
  else {
    std::vector<ModeType> levelTypes;
    std::vector<int> dimensionOrder;
    for (size_t i = 0; i < order; i++) {
      // defaults
      levelTypes.push_back(ModeType::Dense);
      dimensionOrder.push_back(i);
    }
    format = Format(levelTypes, dimensionOrder);
  }

  TensorBase tensor;
  if (util::contains(content->tensors, tensorName)) {
    tensor = content->tensors.at(tensorName);
  }
  else {
    vector<int> dimensionSizes(order);
    vector<bool> dimensionDefault(order, false);
    for (size_t i = 0; i < dimensionSizes.size(); i++) {
      if (util::contains(content->dimensionSizes, tensorName)) {
        dimensionSizes[i] = content->dimensionSizes.at(tensorName)[i];
      }
      else if (util::contains(content->indexVarSizes, varlist[i])) {
        dimensionSizes[i] = content->indexVarSizes.at(varlist[i]);
      }
      else {
        dimensionSizes[i] = content->dimensionDefault;
        dimensionDefault[i] = true;
      }
    }
    tensor = TensorBase(tensorName,Float(64),dimensionSizes,format);

    for (size_t i = 0; i < dimensionSizes.size(); i++) {
      if (dimensionDefault[i]) {
        content->defaultDimension.insert({tensor, i});
      }
    }

    content->tensors.insert({tensorName,tensor});
  }
  return Access(tensor, varlist);
}

vector<IndexVar> Parser::parseVarList() {
  vector<IndexVar> varlist;
  varlist.push_back(parseVar());
  while (content->currentToken == Token::comma) {
    consume(Token::comma);
    varlist.push_back(parseVar());
  }
  return varlist;
}

IndexVar Parser::parseVar() {
  if (content->currentToken != Token::identifier) {
    throw ParseError("Expected index variable");
  }
  IndexVar var = getIndexVar(content->lexer.getIdentifier());
  consume(Token::identifier);
  return var;
}

bool Parser::hasIndexVar(std::string name) const {
  return util::contains(content->indexVars, name);
}

IndexVar Parser::getIndexVar(string name) const {
  taco_iassert(name != "");
  if (!hasIndexVar(name)) {
    IndexVar var(name);
    content->indexVars.insert({name, var});

    // dimensionSizes can also store index var sizes
    if (util::contains(content->dimensionSizes, name)) {
      content->indexVarSizes.insert({var, content->dimensionSizes.at(name)[0]});
    }
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

const std::map<std::string,TensorBase>& Parser::getTensors() const {
  return content->tensors;
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
