#include "taco/parser/linalg_parser.h"
#include "taco/parser/parser.h"

#include <climits>

#include "taco/parser/lexer.h"
#include "taco/tensor.h"
#include "taco/format.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_rewriter.h"

#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace parser {

struct LinalgParser::Content {
  /// Tensor formats
  map<string,Format> formats;
  map<string,Datatype> dataTypes;

  /// Tensor dimensions
  map<string,std::vector<int>> tensorDimensions;
  map<IndexVar, int>           indexVarDimensions;

  int defaultDimension;

  /// Track which modes have default values, so that we can change them
  /// to values inferred from other tensors (read from files).
  set<pair<TensorVar,size_t>> modesWithDefaults;

  Lexer lexer;
  Token currentToken;
  bool parsingLhs = false;

  map<string,IndexVar> indexVars;

  TensorBase             resultTensor;
  map<string,TensorBase> tensors;
};

  LinalgParser::LinalgParser(string expression, const map<string,Format>& formats,
                 const map<string,Datatype>& dataTypes,
                 const map<string,std::vector<int>>& tensorDimensions,
                 const std::map<std::string,TensorBase>& tensors,
                 int defaultDimension)
    : content(new LinalgParser::Content) {
    content->lexer = Lexer(expression);
    content->formats = formats;
    content->tensorDimensions = tensorDimensions;
    content->defaultDimension = defaultDimension;
    content->tensors = tensors;
    content->dataTypes = dataTypes;

    idxcount = 0;

    nextToken();
  }

void LinalgParser::parse() {
    content->resultTensor = parseAssign();
}

const TensorBase& LinalgParser::getResultTensor() const {
    return content->resultTensor;
}

TensorBase LinalgParser::parseAssign() {
  content->parsingLhs = true;
  cout << "parsing lhs" << endl;
  Access lhs = parseVar();
  cout << "Result of parsing LHS" << endl;
  cout << lhs << endl;
  content->parsingLhs = false;

  cout << "parsing rhs" << endl;
  consume(Token::eq);
  IndexExpr rhs = parseExpr();
  cout << "Result of parsing RHS" << endl;
  cout << rhs << endl;

  // Collect all index var dimensions
  struct Visitor : IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    set<pair<TensorVar, size_t>> modesWithDefaults;
    map<IndexVar, int> *indexVarDimensions;

    void visit(const AccessNode *op) {
      for (size_t i = 0; i < op->indexVars.size(); i++) {
        IndexVar indexVar = op->indexVars[i];
        if (!util::contains(modesWithDefaults, {op->tensorVar, i})) {
          auto dimension = op->tensorVar.getType().getShape().getDimension(i);
          if (util::contains(*indexVarDimensions, indexVar)) {
            taco_uassert(indexVarDimensions->at(indexVar) == dimension) <<
                                                                        "Incompatible dimensions";
          } else {
            indexVarDimensions->insert({indexVar, dimension.getSize()});
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
  struct Rewriter : IndexNotationRewriter {
    using IndexNotationRewriter::visit;
    map<IndexVar, int> *indexVarDimensions;
    map<string, TensorBase> tensors;

    void visit(const AccessNode *op) {
      bool dimensionChanged = false;
      Shape shape = op->tensorVar.getType().getShape();
      vector<int> dimensions;
      for (auto &dimension : shape) {
        taco_iassert(dimension.isFixed());
        dimensions.push_back((int) dimension.getSize());
      }

      taco_uassert(op->indexVars.size() == dimensions.size()) <<
                                                              "The order of " << op->tensorVar.getName()
                                                              << " is inconsistent " <<
                                                              "between tensor accesses or options. Is it order " <<
                                                              dimensions.size() << " or " << op->indexVars.size()
                                                              << "?";

      for (size_t i = 0; i < dimensions.size(); i++) {
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
        if (util::contains(tensors, op->tensorVar.getName())) {
          tensor = tensors.at(op->tensorVar.getName());
        } else {
          tensor = TensorBase(op->tensorVar.getName(),
                              op->tensorVar.getType().getDataType(), dimensions,
                              op->tensorVar.getFormat());
          tensors.insert({tensor.getName(), tensor});
        }
        expr = tensor(op->indexVars);
      } else {
        expr = op;
      }
    }
  };
  Rewriter rewriter;
  rewriter.indexVarDimensions = visitor.indexVarDimensions;
  rhs = rewriter.rewrite(rhs);

  IndexExpr rewrittenLhs = rewriter.rewrite(lhs);

  for (auto &tensor : rewriter.tensors) {
    content->tensors.at(tensor.first) = tensor.second;
  }
  content->resultTensor = content->tensors.at(lhs.getTensorVar().getName());

  content->resultTensor(lhs.getIndexVars()) = rhs;
  return content->resultTensor;
}

IndexExpr LinalgParser::parseExpr() {
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

IndexExpr LinalgParser::parseTerm() {
  IndexExpr term = parseFactor();
  while (content->currentToken == Token::mul ||
         content->currentToken == Token::div) {
    switch (content->currentToken) {
      case Token::mul: {
        consume(Token::mul);
        term = term * parseFactor();
        break;
      }
      case Token::div: {
        consume(Token::div);
        term = term / parseFactor();
        break;
      }
      default:
        taco_unreachable;
    }
  }
  return term;
}

IndexExpr LinalgParser::parseFactor() {
  switch (content->currentToken) {
    case Token::lparen: {
      consume(Token::lparen);
      IndexExpr factor = parseExpr();
      consume(Token::rparen);
      return factor;
    }
    case Token::sub:
      consume(Token::sub);
      return new NegNode(parseFactor());
    default:
      break;
  }

  IndexExpr final = parseFinal();

  if (content->currentToken == Token::caretT) {
    consume(Token::caretT);
    return new TransposeNode(final);
  }
  return final;
}

IndexExpr LinalgParser::parseFinal() {
  std::istringstream value (content->lexer.getIdentifier());
  switch (content->currentToken) {
    case Token::complex_scalar:
    {
      consume(Token::complex_scalar);
      std::complex<double> complex_value;
      value >> complex_value;
      return IndexExpr(complex_value);
    }
    case Token::int_scalar:
    {
      consume(Token::int_scalar);
      int64_t int_value;
      value >> int_value;
      return IndexExpr(int_value);
    }
    case Token::uint_scalar:
    {
      consume(Token::uint_scalar);
      uint64_t uint_value;
      value >> uint_value;
      return IndexExpr(uint_value);
    }
    case Token::float_scalar:
    {
      consume(Token::float_scalar);
      double float_value;
      value >> float_value;
      return IndexExpr(float_value);
    }
    default:
      return parseVar();
  }
}

Access LinalgParser::parseVar() {
  if(content->currentToken != Token::identifier) {
    throw ParseError("Expected linalg name");
  }
  string tensorName = content->lexer.getIdentifier();
  cout << tensorName << endl;
  consume(Token::identifier);
  names.push_back(tensorName);

  size_t order = 0;
  // LinalgParser: By default assume capital variables are Matrices and lower case variables are vectors
  if (isupper(tensorName.at(0))) {
    order = 2;
  } else {
    order = 1;
  }
  cout << order << endl;

  Format format;
  if (util::contains(content->formats, tensorName)) {
    format = content->formats.at(tensorName);
  }
  else {
    format = Format(std::vector<ModeFormatPack>(order, Dense));
  }
  cout << format << endl;

  TensorBase tensor;
  if (util::contains(content->tensors, tensorName)) {
    tensor = content->tensors.at(tensorName);
  }
  else {
    vector<int> tensorDimensions(order);
    vector<bool> modesWithDefaults(order, false);
    for (size_t i = 0; i < tensorDimensions.size(); i++) {
      cout << i << endl;
      if (util::contains(content->tensorDimensions, tensorName)) {
        tensorDimensions[i] = content->tensorDimensions.at(tensorName)[i];
      }
      else {
        cout << "default" << endl;
        tensorDimensions[i] = content->defaultDimension;
        modesWithDefaults[i] = true;
      }
    }
    Datatype dataType = Float();
    if (util::contains(content->dataTypes, tensorName)) {
      dataType = content->dataTypes.at(tensorName);
    }
    tensor = TensorBase(tensorName,dataType,tensorDimensions,format);
    cout << tensor << endl;
    for (size_t i = 0; i < tensorDimensions.size(); i++) {
      if (modesWithDefaults[i]) {
        content->modesWithDefaults.insert({tensor.getTensorVar(), i});
      }
    }

    content->tensors.insert({tensorName,tensor});
  }

  cout << order << endl;
  vector<IndexVar> idxlist = getUniqueIndices(order);
  cout << "Idxlist";
  for (auto i : idxlist)
    cout << i << ", ";

  return tensor(idxlist);
}

vector<IndexVar> LinalgParser::getUniqueIndices(size_t order) {
    vector<IndexVar> result;
    for (int i = idxcount; i < (idxcount + (int)order); i++) {
      cout << i << ": ";
      string name = "i" + to_string(i);
      cout << name << " ";
      IndexVar indexVar = getIndexVar(name);
      cout << indexVar << endl;
      result.push_back(indexVar);
    }
    idxcount += order;
  return result;
}

IndexVar LinalgParser::getIndexVar(string name) const {
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

bool LinalgParser::hasIndexVar(std::string name) const {
  return util::contains(content->indexVars, name);
}

void LinalgParser::consume(Token expected) {
  if(content->currentToken != expected) {
    string error = "Expected \'" + content->lexer.tokenString(expected) +
                   "\' but got \'" + currentTokenString() + "\'";
    throw ParseError(error);
  }
  nextToken();
}

const std::map<std::string,TensorBase>& LinalgParser::getTensors() const {
  return content->tensors;
}

// FIXME: Remove this redundancy and try to add it to abstract parser class...
void LinalgParser::nextToken() {
  content->currentToken = content->lexer.getToken();
}

string LinalgParser::currentTokenString() {
  return (content->currentToken == Token::identifier)
         ? content->lexer.getIdentifier()
         : content->lexer.tokenString(content->currentToken);
}

const TensorBase& LinalgParser::getTensor(string name) const {
  taco_iassert(name != "");
  if (!hasTensor(name)) {
    taco_uerror << "Parser error: Tensor name " << name <<
                " not found in expression" << endl;
  }
  return content->tensors.at(name);
}

bool LinalgParser::hasTensor(std::string name) const {
  return util::contains(content->tensors, name);
}
}   // namespace parser
}   // namespace taco
