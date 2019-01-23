#include "lower_codegen.h"

#include <set>

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "iterators.h"
#include "iteration_graph.h"
#include "taco/ir/ir.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;
using namespace taco::ir;

namespace taco {
namespace old {

static vector<TensorVar> getOperands(const IndexExpr& expr) {
  struct GetOperands : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    set<TensorVar> inserted;
    vector<TensorVar> operands;
    void visit(const AccessNode* node) {
      TensorVar tensor = node->tensorVar;
      if (!util::contains(inserted, tensor)) {
        inserted.insert(tensor);
        operands.push_back(tensor);
      }
    }
  };
  GetOperands getOperands;
  expr.accept(&getOperands);
  return getOperands.operands;
}

std::tuple<std::vector<ir::Expr>,         // parameters
           std::vector<ir::Expr>,         // results
           std::map<TensorVar,ir::Expr>>  // mapping
getTensorVars(Assignment assignment) {
  vector<ir::Expr> parameters;
  vector<ir::Expr> results;
  map<TensorVar, ir::Expr> mapping;

  TensorVar tensor = assignment.getLhs().getTensorVar();

  // Pack result tensor into output parameter list
  ir::Expr tensorVarExpr = ir::Var::make(tensor.getName(),
                                         tensor.getType().getDataType(), true, 
                                         true);
  mapping.insert({tensor, tensorVarExpr});
  results.push_back(tensorVarExpr);

  // Pack operand tensors into input parameter list
  for (TensorVar operand : getOperands(assignment.getRhs())) {
    ir::Expr operandVarExpr = ir::Var::make(operand.getName(),
                                           operand.getType().getDataType(), 
                                           true, true);
    taco_iassert(!util::contains(mapping, operand));
    mapping.insert({operand, operandVarExpr});
    parameters.push_back(operandVarExpr);
  }

  return std::tuple<std::vector<ir::Expr>, std::vector<ir::Expr>,
      std::map<TensorVar,ir::Expr>> {parameters, results, mapping};
}

ir::Expr lowerToScalarExpression(const IndexExpr& indexExpr,
                                 const Iterators& iterators,
                                 const IterationGraph& iterationGraph,
                                 const map<TensorVar,ir::Expr>& temporaries) {

  class ScalarCode : public IndexExprVisitorStrict {
    using IndexExprVisitorStrict::visit;

  public:
    const Iterators& iterators;
    const IterationGraph& iterationGraph;
    const map<TensorVar,ir::Expr>& temporaries;
    ScalarCode(const Iterators& iterators,
               const IterationGraph& iterationGraph,
               const map<TensorVar,ir::Expr>& temporaries)
        : iterators(iterators), iterationGraph(iterationGraph),
          temporaries(temporaries) {}

    ir::Expr expr;
    ir::Expr lower(const IndexExpr& indexExpr) {
      indexExpr.accept(this);
      auto e = expr;
      expr = ir::Expr();
      return e;
    }

    void visit(const AccessNode* op) {
      if (util::contains(temporaries, op->tensorVar)) {
        expr = temporaries.at(op->tensorVar);
        return;
      }
      TensorPath path = iterationGraph.getTensorPath(op);
      Type type = op->tensorVar.getType();
      Iterator iterator = (type.getShape().getOrder() == 0)
          ? iterators.getRoot(path)
          : iterators[path.getLastStep()];
      ir::Expr pos = iterator.getPosVar();
      ir::Expr values = GetProperty::make(iterator.getTensor(),
                                          TensorProperty::Values);
      ir::Expr loadValue = Load::make(values, pos);
      expr = loadValue;
    }

    void visit(const LiteralNode* op) {
      switch (op->getDataType().getKind()) {
        case Datatype::Bool:
          taco_not_supported_yet;
          break;
        case Datatype::UInt8:
          expr = ir::Expr((unsigned long long)op->getVal<uint8_t>());
          break;
        case Datatype::UInt16:
          expr = ir::Expr((unsigned long long)op->getVal<uint16_t>());
          break;
        case Datatype::UInt32:
          expr = ir::Expr((unsigned long long)op->getVal<uint32_t>());
          break;
        case Datatype::UInt64:
          expr = ir::Expr((unsigned long long)op->getVal<uint64_t>());
          break;
        case Datatype::UInt128:
          taco_not_supported_yet;
          break;
        case Datatype::Int8:
          expr = ir::Expr((long long)op->getVal<int8_t>());
          break;
        case Datatype::Int16:
          expr = ir::Expr((long long)op->getVal<int16_t>());
          break;
        case Datatype::Int32:
          expr = ir::Expr((long long)op->getVal<int32_t>());
          break;
        case Datatype::Int64:
          expr = ir::Expr((long long)op->getVal<int64_t>());
          break;
        case Datatype::Int128:
          taco_not_supported_yet;
          break;
        case Datatype::Float32:
          expr = ir::Expr(op->getVal<float>());
          break;
        case Datatype::Float64:
          expr = ir::Expr(op->getVal<double>());
          break;
        case Datatype::Complex64:
          expr = ir::Expr(op->getVal<std::complex<float>>());
          break;
        case Datatype::Complex128:
          expr = ir::Expr(op->getVal<std::complex<double>>());
          break;
        case Datatype::Undefined:
          break;
      }
    }

    void visit(const NegNode* op) {
      expr = ir::Neg::make(lower(op->a));
    }

    void visit(const SqrtNode* op) {
      expr = ir::Sqrt::make(lower(op->a));
    }

    void visit(const AddNode* op) {
      expr = ir::Add::make(lower(op->a), lower(op->b));
    }

    void visit(const SubNode* op) {
      expr = ir::Sub::make(lower(op->a), lower(op->b));
    }

    void visit(const MulNode* op) {
      expr = ir::Mul::make(lower(op->a), lower(op->b));
    }

    void visit(const DivNode* op) {
      expr = ir::Div::make(lower(op->a), lower(op->b));
    }

    void visit(const ReductionNode* op) {
      expr = lower(op->a);
    }
  };
  return ScalarCode(iterators,iterationGraph,temporaries).lower(indexExpr);
}

ir::Stmt mergePathIndexVars(ir::Expr var, vector<ir::Expr> pathVars){
  return ir::Assign::make(var, ir::Min::make(pathVars));
}

ir::Expr min(const std::string resultName,
             const std::vector<Iterator>& iterators,
             std::vector<Stmt>* statements) {
  taco_iassert(iterators.size() > 0);
  taco_iassert(statements != nullptr);

  if (iterators.size() == 1) {
    return iterators[0].getCoordVar();
  }

  for (const auto& iterator : iterators) {
    if (iterator.isFull()) {
      return iterator.getCoordVar();
    }
  }

  ir::Expr minVar = ir::Var::make(resultName, Int());
  ir::Expr minExpr = ir::Min::make(getIdxVars(iterators));
  ir::Stmt initIdxStmt = ir::VarDecl::make(minVar, minExpr);
  statements->push_back(initIdxStmt);
  
  return minVar;
}

std::pair<ir::Expr,ir::Expr>
minWithIndicator(const std::string resultName,
                 const std::vector<Iterator>& iterators,
                 std::vector<Stmt>* statements) {
  taco_iassert(iterators.size() >= 2 && 
               (int)iterators.size() <= UInt().getNumBits());
  taco_iassert(statements != nullptr);
  ir::Expr minVar = ir::Var::make(resultName, Int());
  ir::Expr minInd = ir::Var::make(std::string("c") + resultName, UInt());
 
  ir::Stmt initMinIdx = ir::VarDecl::make(minVar, iterators[0].getCoordVar());
  ir::Stmt initMinInd = ir::VarDecl::make(minInd, 1ull);
  statements->push_back(initMinIdx);
  statements->push_back(initMinInd);

  for (size_t i = 1; i < iterators.size(); ++i) {
    ir::Expr idxVar = iterators[i].getCoordVar();
    
    ir::Expr checkLt = ir::Lt::make(idxVar, minVar);
    ir::Stmt replaceMinVar = ir::Assign::make(minVar, idxVar);
    ir::Stmt replaceMinInd = ir::Assign::make(minInd, 1ull << i);
    ir::Stmt replaceStmts = ir::Block::make({replaceMinVar, replaceMinInd});
    
    ir::Expr checkEq = ir::Eq::make(idxVar, minVar);
    ir::Expr newBit = ir::Mul::make(1ull << i, ir::Cast::make(checkEq, UInt()));
    ir::Expr newInd = ir::BitOr::make(minInd, newBit);
    ir::Stmt updateMinInd = ir::Assign::make(minInd, newInd);

    ir::Stmt checkIdxVar = ir::IfThenElse::make(checkLt, replaceStmts, 
                                                updateMinInd);
    statements->push_back(checkIdxVar);
  }
  return std::make_pair(minVar, minInd);
}

vector<ir::Stmt> printCoordinate(const vector<ir::Expr>& indexVars) {
  vector<string> indexVarNames;
  indexVarNames.reserve((indexVars.size()));
  for (auto& indexVar : indexVars) {
    indexVarNames.push_back(util::toString(indexVar));
  }

  vector<string> fmtstrings(indexVars.size(), "%d");
  string format = util::join(fmtstrings, ",");
  vector<ir::Expr> printvars = indexVars;
  return {ir::Print::make("("+util::join(indexVarNames)+") = "
                          "("+format+")\\n", printvars)};
}

}}
