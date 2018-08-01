#include "taco/lower/lowerer_impl.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/ir/ir.h"
#include "iterator.h"
#include "merge_lattice.h"
#include "mode_access.h"
#include "taco/util/collections.h"

using namespace std;
using namespace taco::ir;
using taco::util::combine;

namespace taco {

class LowererImpl::Visitor : public IndexNotationVisitorStrict {
public:
  Visitor(LowererImpl* impl) : impl(impl) {}
  Stmt lower(IndexStmt stmt) {
    IndexStmtVisitorStrict::visit(stmt);
    return this->stmt;
  }
  Expr lower(IndexExpr expr) {
    IndexExprVisitorStrict::visit(expr);
    return this->expr;
  }
private:
  LowererImpl* impl;
  Expr expr;
  Stmt stmt;
  using IndexNotationVisitorStrict::visit;
  void visit(const AssignmentNode* node) { stmt = impl->lowerAssignment(node); }
  void visit(const ForallNode* node)     { stmt = impl->lowerForall(node); }
  void visit(const WhereNode* node)      { stmt = impl->lowerWhere(node); }
  void visit(const MultiNode* node)      { stmt = impl->lowerMulti(node); }
  void visit(const SequenceNode* node)   { stmt = impl->lowerSequence(node); }
  void visit(const AccessNode* node)     { expr = impl->lowerAccess(node); }
  void visit(const LiteralNode* node)    { expr = impl->lowerLiteral(node); }
  void visit(const NegNode* node)        { expr = impl->lowerNeg(node); }
  void visit(const AddNode* node)        { expr = impl->lowerAdd(node); }
  void visit(const SubNode* node)        { expr = impl->lowerSub(node); }
  void visit(const MulNode* node)        { expr = impl->lowerMul(node); }
  void visit(const DivNode* node)        { expr = impl->lowerDiv(node); }
  void visit(const SqrtNode* node)       { expr = impl->lowerSqrt(node); }
  void visit(const ReductionNode* node)  {
    taco_ierror << "Reduction nodes not supported in concrete index notation";
  }
};

LowererImpl::LowererImpl() : visitor(new Visitor(this)) {
}

/// Convert index notation tensor variables to IR pointer variables.
static vector<Expr> createVars(const vector<TensorVar>& tensorVars,
                               map<TensorVar, Expr>* vars) {
  taco_iassert(vars != nullptr);
  vector<Expr> irVars;
  for (auto& var : tensorVars) {
    Expr irVar = Var::make(var.getName(),
                           var.getType().getDataType(),
                           true, true);
    irVars.push_back(irVar);
    vars->insert({var, irVar});
  }
  return irVars;
}

/// Replace scalar tensor pointers with stack scalar for lowering
static Stmt declareScalarArgumentVar(TensorVar var, bool zero,
                                     map<TensorVar, Expr>* tensorVars) {
  Datatype type = var.getType().getDataType();
  Expr varValueIR = Var::make(var.getName() + "_val", type, false, false);
  Expr init = (zero) ? ir::Literal::zero(type)
                     : Load::make(GetProperty::make(tensorVars->at(var),
                                                    TensorProperty::Values));
  tensorVars->find(var)->second = varValueIR;
  return VarAssign::make(varValueIR, init, true);
}

Stmt LowererImpl::lower(IndexStmt stmt, string name, bool assemble,
                        bool compute) {
  this->assemble = assemble;
  this->compute = compute;

  // Create result and parameter variables
  vector<TensorVar> results = getResultTensorVars(stmt);
  vector<TensorVar> arguments = getInputTensorVars(stmt);
  vector<TensorVar> temporaries = getTemporaryTensorVars(stmt);

  // Convert tensor results, arguments and temporaries to IR variables
  map<TensorVar, Expr> resultVars;
  vector<Expr> resultsIR = createVars(results, &resultVars);
  tensorVars.insert(resultVars.begin(), resultVars.end());
  vector<Expr> argumentsIR = createVars(arguments, &tensorVars);
  vector<Expr> temporariesIR = createVars(temporaries, &tensorVars);

  // Create iterators
  createIterators(stmt, tensorVars, &iterators, &indexVars, &coordVars___);

  map<TensorVar, Expr> scalars;
  vector<Stmt> headerStmts;
  vector<Stmt> footerStmts;

  // Declare and initialize dimension variables
  vector<IndexVar> indexVars = getIndexVars(stmt);
  for (auto& ivar : indexVars) {
    Expr dimension;
    match(stmt,
      function<void(const AssignmentNode*,Matcher*)>([&](
          const AssignmentNode* n, Matcher* m) {
        m->match(n->rhs);
        if (!dimension.defined()) {
          auto ivars = n->lhs.getIndexVars();
          int loc = distance(ivars.begin(),
                             find(ivars.begin(),ivars.end(), ivar));
          dimension = GetProperty::make(tensorVars.at(n->lhs.getTensorVar()),
                                        TensorProperty::Dimension, loc);
        }
      }),
      function<void(const AccessNode*)>([&](const AccessNode* n) {
        auto ivars = n->indexVars;
        int loc = distance(ivars.begin(),
                                find(ivars.begin(),ivars.end(), ivar));
        dimension = GetProperty::make(tensorVars.at(n->tensorVar),
                                      TensorProperty::Dimension, loc);
      })
    );
    Expr ivarIR = Var::make(ivar.getName() + "_size", type<int32_t>());
    Stmt decl = VarAssign::make(ivarIR, dimension, true);
    dimensions.insert({ivar, ivarIR});
    headerStmts.push_back(decl);
  }

  // Declare and initialize scalar results and arguments
  if (generateComputeCode()) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(!util::contains(scalars, result));
        taco_iassert(util::contains(tensorVars, result));
        scalars.insert({result, tensorVars.at(result)});
        headerStmts.push_back(declareScalarArgumentVar(result, true,
                                                       &tensorVars));
      }
    }
    for (auto& argument : arguments) {
      if (isScalar(argument.getType())) {
        taco_iassert(!util::contains(scalars, argument));
        taco_iassert(util::contains(tensorVars, argument));
        scalars.insert({argument, tensorVars.at(argument)});
        headerStmts.push_back(declareScalarArgumentVar(argument, false,
                                                       &tensorVars));
      }
    }
  }

  // Declare, allocate, and initialize temporaries.
  if (generateComputeCode()) {
    for (auto& temporary : temporaries) {
      if (isScalar(temporary.getType())) {
        taco_iassert(!util::contains(scalars, temporary)) << temporary;
        taco_iassert(util::contains(tensorVars, temporary));
        scalars.insert({temporary, tensorVars.at(temporary)});
        headerStmts.push_back(declareScalarArgumentVar(temporary, true,
                                                       &tensorVars));
      }
    }
  }

  // Allocate memory for dense results up front.
  if (generateAssembleCode()) {
    for (auto& result : results) {
      Format format = result.getFormat();
      if (isDense(format)) {
        Expr resultIR = resultVars.at(result);
        Expr vals = GetProperty::make(resultIR, TensorProperty::Values);

        // Compute size from dimension sizes
        // TODO: If dimensions are constant then emit constants here
        Expr size = (result.getOrder() > 0)
                    ? GetProperty::make(resultIR, TensorProperty::Dimension, 0)
                    : 1;
        for (int i = 1; i < result.getOrder(); i++) {
          size = ir::Mul::make(size,
                               GetProperty::make(resultIR,
                                                 TensorProperty::Dimension, i));
        }
        headerStmts.push_back(Allocate::make(vals, size));
      }
    }
  }

  // Lower the index statement to compute and/or assemble.
  Stmt body = lower(stmt);

  // Store scalar stack variables back to results.
  if (generateComputeCode()) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(util::contains(scalars, result));
        taco_iassert(util::contains(tensorVars, result));
        Expr resultIR = scalars.at(result);
        Expr varValueIR = tensorVars.at(result);
        Expr valuesArrIR = GetProperty::make(resultIR, TensorProperty::Values);
        footerStmts.push_back(Store::make(valuesArrIR, 0, varValueIR));
      }
    }
  }

  // Create function.
  Stmt header = (headerStmts.size() > 0)
                ? Block::make(util::combine(headerStmts, {BlankLine::make()}))
                : Stmt();
  Stmt footer = (footerStmts.size() > 0)
                ? Block::make(util::combine({BlankLine::make()}, footerStmts))
                : Stmt();
  return Function::make(name, resultsIR, argumentsIR,
                        Block::make({header, body, footer}));
}

Stmt LowererImpl::lowerAssignment(Assignment assignment) {
  TensorVar result = assignment.getLhs().getTensorVar();

  if (generateComputeCode()) {
    Expr varIR = getTensorVar(result);
    Expr rhs = lower(assignment.getRhs());

    // Assignment to scalar variables.
    if (isScalar(result.getType())) {
      if (!assignment.getOperator().defined()) {
        return VarAssign::make(varIR, rhs);
      }
      else {
        taco_iassert(isa<taco::Add>(assignment.getOperator()));
        return VarAssign::make(varIR, ir::Add::make(varIR,rhs));
      }
    }
    // Assignments to tensor variables (non-scalar).
    else {
      Expr valueArray = GetProperty::make(varIR, TensorProperty::Values);
      return ir::Store::make(valueArray, generateValueLocExpr(assignment.getLhs()),
                           rhs);
      // When we're assembling while computing we need to allocate more
      // value memory as we write to the values array.
      if (generateAssembleCode()) {
        // TODO
      }
    }
  }
  // We're only assembling so defer allocating value memory to the end when
  // we'll know exactly how much we need.
  else if (generateAssembleCode()) {
    // TODO
    return Block::make();
  }
  // We're neither assembling or computing so we emit nothing.
  else {
    return Block::make();
  }
  taco_unreachable;
  return Stmt();
}

Stmt LowererImpl::lowerForall(Forall forall) {
  MergeLattice lattice = MergeLattice::make(forall, getIterators());

  // Emit a loop that iterates over over a single iterator (optimization)
  if (lattice.getRangeIterators().size() == 1) {
           Iterator  rangeIterator   = lattice.getRangeIterators()[0];
    vector<Iterator> locateIterators = lattice.getMergeIterators();
    vector<Iterator> insertIterators = lattice.getInsertIterators();
    vector<Iterator> appendIterators = lattice.getAppendIterators();

    // Emit dimension coordinate iteration loop
    if (rangeIterator.isFull() && rangeIterator.hasLocate()) {
      return lowerForallDimension(forall, locateIterators,
                                  insertIterators, appendIterators);
    }
    // Emit position iteration loop
    else if (rangeIterator.hasPosIter()) {
      locateIterators.erase(remove(locateIterators.begin(),
                                   locateIterators.end(),
                                   rangeIterator), locateIterators.end());
      return lowerForallPosition(forall, rangeIterator, locateIterators,
                                 insertIterators, appendIterators);
    }
    // Emit coordinate iteration loop
    else {
      taco_iassert(rangeIterator.hasCoordIter());
      taco_not_supported_yet;
      return Stmt();
    }
  }
  // Emit general loops to merge multiple iterators
  else {
    return lowerForallMerge(forall, lattice);
  }
}

Stmt LowererImpl::lowerForallDimension(Forall forall,
                                       vector<Iterator> locateIterators,
                                       vector<Iterator> insertIterators,
                                       vector<Iterator> appendIterators) {
  Stmt body = generateLoopBody(forall, locateIterators, insertIterators,
                               appendIterators);
  IndexVar indexVar  = forall.getIndexVar();
  return For::make(getCoordinateVar(indexVar), 0, getDimension(indexVar), 1,
                   body);
}

Stmt LowererImpl::lowerForallCoordinate(Forall forall, Iterator iterator,
                                        vector<Iterator> locateIterators,
                                        vector<Iterator> insertIterators,
                                        vector<Iterator> appendIterators) {
  taco_not_supported_yet;
  return Stmt();
}

Stmt LowererImpl::lowerForallPosition(Forall forall, Iterator iterator,
                                      vector<Iterator> locateIterators,
                                      vector<Iterator> insertIterators,
                                      vector<Iterator> appendIterators) {
  Stmt coordVarDecl =
      VarAssign::make(getCoordinateVar(forall.getIndexVar()),
                      iterator.posAccess(getCoords(iterator)).getResults()[0],
                      true);
  Stmt body = generateLoopBody(forall, locateIterators, insertIterators,
                               appendIterators);
  ModeFunction bounds = iterator.posBounds();
  return Block::make({bounds.getBody(),
                      For::make(iterator.getPosVar(),
                                bounds.getResults()[0],
                                bounds.getResults()[1], 1,
                                Block::make({coordVarDecl, body}))});
}

Stmt LowererImpl::lowerForallMerge(Forall forall, MergeLattice lattice) {
  IndexVar  indexVar  = forall.getIndexVar();
  IndexStmt indexStmt = forall.getStmt();
  Expr coordVar = getCoordinateVar(indexVar);

  // Emit merge position variables

  // Emit a loop for each merge lattice point lp

  // Emit merge coordinate variables

  // Emit coordinate variable

  // Emit located position variables

  // Emit a case for each child lattice point lq of lp

  // Emit loop body

  // Emit code to increment merged position variables

  return Stmt();
}

Stmt LowererImpl::lowerWhere(Where where) {
  // TODO: Either initialise or re-initialize temporary memory
  Stmt producer = lower(where.getProducer());
  Stmt consumer = lower(where.getConsumer());
  return Block::make({producer, consumer});
}

Stmt LowererImpl::lowerSequence(Sequence sequence) {
  Stmt definition = lower(sequence.getDefinition());
  Stmt mutation = lower(sequence.getMutation());
  return Block::make({definition, mutation});
}

Stmt LowererImpl::lowerMulti(Multi multi) {
  Stmt stmt1 = lower(multi.getStmt1());
  Stmt stmt2 = lower(multi.getStmt2());
  return Block::make({stmt1, stmt2});
}

Expr LowererImpl::lowerAccess(Access access) {
  TensorVar var = access.getTensorVar();
  Expr varIR = getTensorVar(var);
  return (isScalar(var.getType()))
         ? varIR
         : Load::make(GetProperty::make(varIR, TensorProperty::Values),
                      generateValueLocExpr(access));
}

Expr LowererImpl::lowerLiteral(Literal) {
  taco_not_supported_yet;
  return Expr();
}

Expr LowererImpl::lowerNeg(Neg neg) {
  return ir::Neg::make(lower(neg.getA()));
}

Expr LowererImpl::lowerAdd(Add add) {
  return ir::Add::make(lower(add.getA()), lower(add.getB()));
}

Expr LowererImpl::lowerSub(Sub sub) {
  return ir::Sub::make(lower(sub.getA()), lower(sub.getB()));
}

Expr LowererImpl::lowerMul(Mul mul) {
  return ir::Mul::make(lower(mul.getA()), lower(mul.getB()));
}

Expr LowererImpl::lowerDiv(Div div) {
  return ir::Div::make(lower(div.getA()), lower(div.getB()));
}

Expr LowererImpl::lowerSqrt(Sqrt sqrt) {
  return ir::Sqrt::make(lower(sqrt.getA()));
}

Stmt LowererImpl::lower(IndexStmt stmt) {
  return visitor->lower(stmt);
}

Expr LowererImpl::lower(IndexExpr expr) {
  return visitor->lower(expr);
}

bool LowererImpl::generateAssembleCode() const {
  return this->assemble;
}

bool LowererImpl::generateComputeCode() const {
  return this->compute;
}

Expr LowererImpl::getTensorVar(TensorVar tensorVar) const {
  taco_iassert(util::contains(this->tensorVars, tensorVar)) << tensorVar;
  return this->tensorVars.at(tensorVar);
}

Expr LowererImpl::getDimension(IndexVar indexVar) const {
  taco_iassert(util::contains(this->dimensions, indexVar)) << indexVar;
  return this->dimensions.at(indexVar);
}

Iterator LowererImpl::getIterator(ModeAccess modeAccess) const {
  taco_iassert(util::contains(getIterators(), modeAccess)) << modeAccess;
  return getIterators().at(modeAccess);
}

const map<ModeAccess, Iterator>& LowererImpl::getIterators() const {
  return this->iterators;
}

Expr LowererImpl::getCoordinateVar(IndexVar indexVar) const {
  taco_iassert(util::contains(this->coordVars___, indexVar)) << indexVar;
  return this->coordVars___.at(indexVar);
}

Expr LowererImpl::getCoordinateVar(Iterator iterator) const {
  taco_iassert(util::contains(this->indexVars, iterator)) << iterator;
  auto& indexVar = this->indexVars.at(iterator);
  return this->getCoordinateVar(indexVar);
}

vector<Expr> LowererImpl::getCoords(Iterator iterator) {
  vector<Expr> coords;
  do {
    coords.push_back(getCoordinateVar(iterator));
    iterator = iterator.getParent();
  } while (iterator.getParent().defined());
  util::reverse(coords);
  return coords;
}


Expr LowererImpl::generateValueLocExpr(Access access) const {
  if (isScalar(access.getTensorVar().getType())) {
    return ir::Literal::make(0);
  }
  size_t loc = access.getIndexVars().size();
  Iterator it = getIterator(ModeAccess(access, loc));
  return it.getPosVar();
}

Stmt LowererImpl::generatePosVarLocateDecls(vector<Iterator> locateIterators) {
  vector<Stmt> posVarDecls;
  for (Iterator& locateIterator : locateIterators) {
    ModeFunction locate = locateIterator.locate(getCoords(locateIterator));
    taco_iassert(isValue(locate.getResults()[1], true));
    Stmt posVarDecl = VarAssign::make(locateIterator.getPosVar(),
                                      locate.getResults()[0],
                                      true);
    posVarDecls.push_back(posVarDecl);
  }
  return Block::make(posVarDecls);
}

ir::Stmt LowererImpl::generateLoopBody(Forall forall,
                                       vector<Iterator> locateIterators,
                                       vector<Iterator> insertIterators,
                                       vector<Iterator> appendIterators) {
  Expr coordVar = getCoordinateVar(forall.getIndexVar());
  Stmt posVarDeclarations = generatePosVarLocateDecls(combine(locateIterators,
                                                              insertIterators));
  Stmt body = lower(forall.getStmt());
  return Block::make({posVarDeclarations, body});
}

}
