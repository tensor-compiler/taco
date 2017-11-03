#include "taco/parser/parser.h"

#include <climits>

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
  map<string,std::vector<int>> tensorDimensions;
  map<IndexVar, int>           indexVarDimensions;

  int defaultDimension;

  /// Track which modes have default values, so that we can change them
  /// to values inferred from other tensors (read from files).
  set<pair<TensorBase,size_t>> modesWithDefaults;

  Lexer lexer;
  Token currentToken;
  bool parsingLhs = false;

  map<string,IndexVar> indexVars;

  TensorBase             resultTensor;
  map<string,TensorBase> tensors;
};

Parser::Parser(string expression, const map<string,Format>& formats,
               const map<string,std::vector<int>>& tensorDimensions,
               const std::map<std::string,TensorBase>& tensors,
               int defaultDimension)
    : content(new Parser::Content) {
  content->lexer = Lexer(expression);
  content->formats = formats;
  content->tensorDimensions = tensorDimensions;
  content->defaultDimension = defaultDimension;
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

  // TODO: This is a quick hack to support '+='. A better solution is to create
  // a new token for it and to handle it in the lexer.
  bool accumulate = false;
  if (content->currentToken == Token::add) {
    consume(Token::add);
    accumulate = true;
  }

  consume(Token::eq);
  IndexExpr rhs = parseExpr();

  // Collect all index var dimensions
  struct Visitor : expr_nodes::ExprVisitor {
    using ExprVisitor::visit;
    set<pair<TensorBase,size_t>> modesWithDefaults;
    map<IndexVar, int>* indexVarDimensions;

    void visit(const expr_nodes::AccessNode* op) {
      for (size_t i = 0; i < op->indexVars.size(); i++) {
        IndexVar indexVar = op->indexVars[i];
        if (!util::contains(modesWithDefaults, {op->tensor,i})) {
          int dimension = op->tensor.getDimension(i);
          if (util::contains(*indexVarDimensions, indexVar)) {
            taco_uassert(indexVarDimensions->at(indexVar) == dimension) <<
                "Incompatible dimensions";
          }
          else {
            indexVarDimensions->insert({indexVar, dimension});
          }
        }
      }
    }
  };
  Visitor visitor;
  visitor.indexVarDimensions = &content->indexVarDimensions;
  visitor.modesWithDefaults = content->modesWithDefaults;
  rhs.accept(&visitor);

  // Rewrite expression to new index dimensions
  struct Rewriter : expr_nodes::ExprRewriter {
    using ExprRewriter::visit;
    map<IndexVar, int>* indexVarDimensions;
    map<string,TensorBase> tensors;

    void visit(const expr_nodes::AccessNode* op) {
      bool dimensionChanged = false;
      vector<int> dimensions = op->tensor.getDimensions();

      taco_uassert(op->indexVars.size() == dimensions.size()) <<
          "The order of " << op->tensor.getName() << " is inconsistent " <<
          "between tensor accesses or options. Is it order " <<
          dimensions.size() << " or " << op->indexVars.size() << "?";

      for (size_t i=0; i < dimensions.size(); i++) {
        IndexVar indexVar = op->indexVars[i];
        if (util::contains(*indexVarDimensions, indexVar)) {
          int dimension = indexVarDimensions->at(indexVar);
          if (dimension != dimensions[i]) {
            dimensions[i] = dimension;
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
        expr = new expr_nodes::AccessNode(tensor, op->indexVars);
      }
      else {
        expr = op;
      }
    }
  };
  Rewriter rewriter;
  rewriter.indexVarDimensions = visitor.indexVarDimensions;
  rhs = rewriter.rewrite(rhs);

  IndexExpr rewrittenLhs = rewriter.rewrite(lhs);

  for (auto& tensor : rewriter.tensors) {
    content->tensors.at(tensor.first) = tensor.second;
  }
  content->resultTensor = content->tensors.at(lhs.getTensor().getName());
  content->resultTensor.setExpr(lhs.getIndexVars(), rhs, accumulate);
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

  taco_iassert(varlist.size() <= INT_MAX);
  size_t order = varlist.size();

  Format format;
  if (util::contains(content->formats, tensorName)) {
    format = content->formats.at(tensorName);
  }
  else {
    std::vector<ModeType> levelTypes;
    std::vector<size_t> modeOrdering;
    for (size_t i = 0; i < order; i++) {
      // defaults
      levelTypes.push_back(ModeType::Dense);
      modeOrdering.push_back(i);
    }
    format = Format(levelTypes, modeOrdering);
  }

  TensorBase tensor;
  if (util::contains(content->tensors, tensorName)) {
    tensor = content->tensors.at(tensorName);
  }
  else {
    vector<int> tensorDimensions(order);
    vector<bool> modesWithDefaults(order, false);
    for (size_t i = 0; i < tensorDimensions.size(); i++) {
      if (util::contains(content->tensorDimensions, tensorName)) {
        tensorDimensions[i] = content->tensorDimensions.at(tensorName)[i];
      }
      else if (util::contains(content->indexVarDimensions, varlist[i])) {
        tensorDimensions[i] = content->indexVarDimensions.at(varlist[i]);
      }
      else {
        tensorDimensions[i] = content->defaultDimension;
        modesWithDefaults[i] = true;
      }
    }
    tensor = TensorBase(tensorName,Float(64),tensorDimensions,format);

    for (size_t i = 0; i < tensorDimensions.size(); i++) {
      if (modesWithDefaults[i]) {
        content->modesWithDefaults.insert({tensor, i});
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

    // tensorDimensions can also store index var dimensions
    if (util::contains(content->tensorDimensions, name)) {
      content->indexVarDimensions.insert({var, content->tensorDimensions.at(name)[0]});
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
