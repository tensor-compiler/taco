#include "taco/lower/lowerer_impl.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/ir/ir.h"
#include "ir/ir_generators.h"
#include "taco/ir/simplify.h"
#include "taco/lower/iterator.h"
#include "taco/lower/merge_lattice.h"
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
    this->stmt = Stmt();
    IndexStmtVisitorStrict::visit(stmt);
    return this->stmt;
  }
  Expr lower(IndexExpr expr) {
    this->expr = Expr();
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
  return VarDecl::make(varValueIR, init);
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
  iterators = Iterators::make(stmt, tensorVars, &indexVars);

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
          int loc = (int)distance(ivars.begin(),
                                  find(ivars.begin(),ivars.end(), ivar));
          dimension = GetProperty::make(tensorVars.at(n->lhs.getTensorVar()),
                                        TensorProperty::Dimension, loc);
        }
      }),
      function<void(const AccessNode*)>([&](const AccessNode* n) {
        auto ivars = n->indexVars;
        int loc = (int)distance(ivars.begin(),
                                find(ivars.begin(),ivars.end(), ivar));
        dimension = GetProperty::make(tensorVars.at(n->tensorVar),
                                      TensorProperty::Dimension, loc);
      })
    );
    dimensions.insert({ivar, dimension});
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

  // Allocate memory for scalar results
  if (generateAssembleCode()) {
    for (auto& result : results) {
      if (result.getOrder() == 0) {
        Expr resultIR = resultVars.at(result);
        Expr vals = GetProperty::make(resultIR, TensorProperty::Values);
        Expr valsSize = GetProperty::make(resultIR, TensorProperty::ValuesSize);
        headerStmts.push_back(Assign::make(valsSize, 1));
        headerStmts.push_back(Allocate::make(vals, valsSize));
      }
    }
  }

  // Allocate and initialize append and insert mode indices
  Stmt initializeResults = initResultArrays(getResultAccesses(stmt));

  // Declare, allocate, and initialize temporaries
  Stmt declareTemporaries = declTemporaries(temporaries, scalars);

  // Lower the index statement to compute and/or assemble
  Stmt body = lower(stmt);

  // Post-process result modes and allocate memory for values if necessary
  Stmt finalizeResults = finalizeResultArrays(getResultAccesses(stmt));

  // Store scalar stack variables back to results
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

  // Create function
  Stmt header = (headerStmts.size() > 0) ? Block::make(headerStmts) : Stmt();
  Stmt footer = (footerStmts.size() > 0) ? Block::make(footerStmts) : Stmt();
  return Function::make(name, resultsIR, argumentsIR,
                        Block::blanks(header,
                                      declareTemporaries,
                                      initializeResults,
                                      body,
                                      finalizeResults,
                                      footer));
}


Stmt LowererImpl::lowerAssignment(Assignment assignment) {
  TensorVar result = assignment.getLhs().getTensorVar();

  if (generateComputeCode()) {
    Expr var = getTensorVar(result);
    Expr rhs = lower(assignment.getRhs());

    // Assignment to scalar variables.
    if (isScalar(result.getType())) {
      if (!assignment.getOperator().defined()) {
        return Assign::make(var, rhs);
      }
      else {
        taco_iassert(isa<taco::Add>(assignment.getOperator()));
        return Assign::make(var, ir::Add::make(var,rhs));
      }
    }
    // Assignments to tensor variables (non-scalar).
    else {
      Expr values = GetProperty::make(var, TensorProperty::Values);
      Expr size = GetProperty::make(var, TensorProperty::ValuesSize);
      Expr loc = generateValueLocExpr(assignment.getLhs());

      // When we're assembling while computing we need to allocate more
      // value memory as we write to the values array.
      Iterator lastIterator = getIterators(assignment.getLhs()).back();
      Stmt resizeValueArray;
      if (generateAssembleCode() && lastIterator.hasAppend()) {
        resizeValueArray = doubleSizeIfFull(values, size, loc);
      }

      Stmt computeStmt = Store::make(values, loc, rhs);

      return resizeValueArray.defined()
             ? Block::make(resizeValueArray,  computeStmt)
             : computeStmt;
    }
  }
  // We're only assembling so defer allocating value memory to the end when
  // we'll know exactly how much we need.
  else if (generateAssembleCode()) {
    // TODO
    return Stmt();
  }
  // We're neither assembling or computing so we emit nothing.
  else {
    return Stmt();
  }
  taco_unreachable;
  return Stmt();
}


static pair<vector<Iterator>, vector<Iterator>>
splitAppenderAndInserters(const vector<Iterator>& results) {
  vector<Iterator> appenders;
  vector<Iterator> inserters;

  // TODO: Choose insert when the current forall is nested inside a reduction
  for (auto& result : results) {
    taco_iassert(result.hasAppend() || result.hasInsert())
        << "Results must support append or insert";

    if (result.hasAppend()) {
      appenders.push_back(result);
    }
    else {
      taco_iassert(result.hasInsert());
      inserters.push_back(result);
    }
  }

  return {appenders, inserters};
}


Stmt LowererImpl::lowerForall(Forall forall)
{
  MergeLattice lattice = MergeLattice::make(forall, iterators);

  // Pre-allocate/initialize memory of value arrays that are full below this
  // loops index variable
  Stmt preInitValues = initResultArrays(forall.getIndexVar(),
                                        getResultAccesses(forall));

  Stmt loops;
  // Emit a loop that iterates over over a single iterator (optimization)
  if (lattice.points().size() == 1 && lattice.iterators().size() == 1) {
    MergePoint point = lattice.points()[0];
    Iterator iterator = lattice.iterators()[0];

    vector<Iterator> locaters = point.locators();
    vector<Iterator> appenders;
    vector<Iterator> inserters;
    tie(appenders, inserters) = splitAppenderAndInserters(point.results());

    // Emit dimension coordinate iteration loop
    if (iterator.isDimensionIterator()) {
      loops = lowerForallDimension(forall, point.locators(),
                                  inserters, appenders);
    }
    // Emit position iteration loop
    else if (iterator.hasPosIter()) {
      loops = lowerForallPosition(forall, iterator, locaters,
                                 inserters, appenders);
    }
    // Emit coordinate iteration loop
    else {
      taco_iassert(iterator.hasCoordIter());
      taco_not_supported_yet;
      loops = Stmt();
    }
  }
  // Emit general loops to merge multiple iterators
  else {
    loops = lowerMergeLattice(lattice, getCoordinateVar(forall.getIndexVar()),
                              forall.getStmt());
  }
  taco_iassert(loops.defined());

  return Block::blanks(preInitValues,
                       loops);
}


Stmt LowererImpl::lowerForallDimension(Forall forall,
                                       vector<Iterator> locators,
                                       vector<Iterator> inserters,
                                       vector<Iterator> appenders)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders);

  Stmt posAppend = generateAppendPositions(appenders);

  // Emit loop with preamble and postamble
  Expr dimension = getDimension(forall.getIndexVar());
  return Block::blanks(For::make(coordinate, 0, dimension, 1, body,
                                 LoopKind::Serial, false),
                       posAppend);
}


Stmt LowererImpl::lowerForallCoordinate(Forall forall, Iterator iterator,
                                        vector<Iterator> locaters,
                                        vector<Iterator> inserters,
                                        vector<Iterator> appenders) {
  taco_not_supported_yet;
  return Stmt();
}

Stmt LowererImpl::lowerForallPosition(Forall forall, Iterator iterator,
                                      vector<Iterator> locators,
                                      vector<Iterator> inserters,
                                      vector<Iterator> appenders)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  Expr coordinateArray= iterator.posAccess(coordinates(iterator)).getResults()[0];
  Stmt declareCoordinate = VarDecl::make(coordinate, coordinateArray);

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders);

  // Code to append positions
  Stmt posAppend = generateAppendPositions(appenders);

  // Loop with preamble and postamble
  ModeFunction bounds = iterator.posBounds();
  return Block::blanks(bounds.compute(),
                       For::make(iterator.getPosVar(), bounds[0], bounds[1], 1,
                                 Block::make(declareCoordinate, body),
                                 LoopKind::Serial, false),
                       posAppend);
}

Stmt LowererImpl::lowerMergeLattice(MergeLattice lattice, Expr coordinate,
                                    IndexStmt statement)
{
  vector<Iterator> appenders = filter(lattice.results(),
                                      [](Iterator it){return it.hasAppend();});

  // TODO: this is fishy, will not this memory be initialized again at recursive
  //       loop lowering?)
  Stmt iteratorVarInits = codeToInitializeIteratorVars(lattice.iterators());

  vector<Stmt> mergeLoopsVec;
  for (MergePoint point : lattice.points()) {
    // Each iteration of this loop generates a while loop for one of the merge
    // points in the merge lattice.
    IndexStmt zeroedStmt = zero(statement, getExhaustedAccesses(point,lattice));
    MergeLattice sublattice = lattice.subLattice(point);
    Stmt mergeLoop = lowerMergePoint(sublattice, coordinate, zeroedStmt);
    mergeLoopsVec.push_back(mergeLoop);
  }
  Stmt mergeLoops = Block::make(mergeLoopsVec);

  // Append position to the pos array
  Stmt appendPositions = generateAppendPositions(appenders);

  return Block::blanks(iteratorVarInits,
                       mergeLoops,
                       appendPositions);
}

Stmt LowererImpl::lowerMergePoint(MergeLattice pointLattice,
                                  ir::Expr coordinate, IndexStmt statement)
{
  MergePoint point = pointLattice.points().front();

  vector<Iterator> iterators = point.iterators();
  vector<Iterator> mergers = point.mergers();
  vector<Iterator> rangers = point.rangers();
  vector<Iterator> locators = point.locators();

  taco_iassert(iterators.size() > 0);
  taco_iassert(mergers.size() > 0);
  taco_iassert(rangers.size() > 0);

  // Load coordinates from position iterators
  Stmt loadPosIterCoordinates;
  if (iterators.size() > 1) {
    vector<Stmt> loadPosIterCoordinateStmts;
    auto posIters = filter(iterators, [](Iterator it){return it.hasPosIter();});
    for (auto& posIter : posIters) {
      taco_tassert(posIter.hasPosIter());
      ModeFunction posAccess = posIter.posAccess(coordinates(posIter));
      loadPosIterCoordinateStmts.push_back(posAccess.compute());
      loadPosIterCoordinateStmts.push_back(VarDecl::make(posIter.getCoordVar(),
                                                          posAccess[0]));
    }
    loadPosIterCoordinates = Block::make(loadPosIterCoordinateStmts);
  }

  // Merge iterator coordinate variables
  Stmt resolveCoordinate;
  if (mergers.size() == 1) {
    Iterator merger = mergers[0];
    if (merger.hasPosIter()) {
      // Just one position iterator so it is the resolved coordinate
      ModeFunction posAccess = merger.posAccess(coordinates(merger));
      resolveCoordinate = Block::make(posAccess.compute(),
                                          VarDecl::make(coordinate,
                                                        posAccess[0]));
    }
    else if (merger.hasCoordIter()) {
      taco_not_supported_yet;
    }
    else if (merger.isDimensionIterator()) {
      // Just one dimension iterator so resolved coordinate already exist and we
      // do nothing
    }
    else {
      taco_ierror << "Unexpected type of single iterator " << merger;
    }
  }
  else {
    // Multiple position iterators so the smallest is the resolved coordinate
    resolveCoordinate = VarDecl::make(coordinate,
                                          Min::make(coordinates(mergers)));
  }

  // Locate positions
  Stmt loadLocatorPosVars = declLocatePosVars(locators);

  // One case for each child lattice point lp
  Stmt caseStmts = lowerMergeCases(coordinate, statement, pointLattice);

  // Increment iterator position variables
  Stmt incIteratorVarStmts = codeToIncIteratorVars(coordinate, iterators);

  /// While loop over rangers
  return While::make(checkThatNoneAreExhausted(rangers),
                     Block::make(loadPosIterCoordinates,
                                 resolveCoordinate,
                                 loadLocatorPosVars,
                                 caseStmts,
                                 incIteratorVarStmts));
}

Stmt LowererImpl::lowerMergeCases(ir::Expr coordinate, IndexStmt stmt,
                                  MergeLattice lattice)
{
  vector<Stmt> result;

  vector<Iterator> appenders;
  vector<Iterator> inserters;
  tie(appenders, inserters) = splitAppenderAndInserters(lattice.results());

  // Just one iterator so no conditionals
  if (lattice.iterators().size() == 1) {
    Stmt body = lowerForallBody(coordinate, stmt, {}, inserters, appenders);
    result.push_back(body);
  }
  else {
    vector<pair<Expr,Stmt>> cases;
    for (MergePoint point : lattice.points()) {

      // Construct case expression
      vector<Expr> coordComparisons;
      for (Iterator iterator : point.rangers()) {
        coordComparisons.push_back(Eq::make(iterator.getCoordVar(),coordinate));
      }

      // Construct case body
      IndexStmt zeroedStmt = zero(stmt, getExhaustedAccesses(point, lattice));
      Stmt body = lowerForallBody(coordinate, zeroedStmt, {},
                                  inserters, appenders);

      cases.push_back({conjunction(coordComparisons), body});
    }
    result.push_back(Case::make(cases, lattice.exact()));
  }

  return Block::make(result);
}


Stmt LowererImpl::lowerForallBody(Expr coordinate, IndexStmt stmt,
                                  vector<Iterator> locators,
                                  vector<Iterator> inserters,
                                  vector<Iterator> appenders) {
  // Inserter positions
  Stmt declInserterPosVars = declLocatePosVars(inserters);

  // Locate positions
  Stmt declLocatorPosVars = declLocatePosVars(locators);

  // Code of loop body statement
  Stmt body = lower(stmt);

  // Code to append coordinates
  Stmt appendCoordinate = generateAppendCoordinate(appenders, coordinate);

  // Code to increment append position variables
  Stmt incrementAppendPositionVars = generateAppendPosVarIncrements(appenders);

  return Block::make(declInserterPosVars,
                     declLocatorPosVars,
                     body,
                     appendCoordinate,
                     incrementAppendPositionVars);
}

Stmt LowererImpl::lowerWhere(Where where) {
  // TODO: Either initialise or re-initialize temporary memory
  Stmt producer = lower(where.getProducer());
  Stmt consumer = lower(where.getConsumer());
  return Block::make(producer, consumer);
}


Stmt LowererImpl::lowerSequence(Sequence sequence) {
  Stmt definition = lower(sequence.getDefinition());
  Stmt mutation = lower(sequence.getMutation());
  return Block::make(definition, mutation);
}


Stmt LowererImpl::lowerMulti(Multi multi) {
  Stmt stmt1 = lower(multi.getStmt1());
  Stmt stmt2 = lower(multi.getStmt2());
  return Block::make(stmt1, stmt2);
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


std::vector<Iterator> LowererImpl::getIterators(Access access) const {
  vector<Iterator> result;
  TensorVar tensor = access.getTensorVar();
  for (int i = 0; i < tensor.getOrder(); i++) {
    int mode = tensor.getFormat().getModeOrdering()[i];
    result.push_back(iterators.levelIterator(ModeAccess(access, mode+1)));
  }
  return result;
}


set<Access> LowererImpl::getExhaustedAccesses(MergePoint point,
                                              MergeLattice lattice) const
{
  set<Access> exhaustedAccesses;
  for (auto& iterator : lattice.exhausted(point)) {
    exhaustedAccesses.insert(iterators.modeAccess(iterator).getAccess());
  }
  return exhaustedAccesses;
}


Expr LowererImpl::getCoordinateVar(IndexVar indexVar) const {
  return this->iterators.modeIterator(indexVar).getCoordVar();
}


Expr LowererImpl::getCoordinateVar(Iterator iterator) const
{
  if (iterator.isDimensionIterator()) {
    return iterator.getCoordVar();
  }
  taco_iassert(util::contains(this->indexVars, iterator))
      << "Could not find a coordinate for " << iterator << " from "
      << util::join(this->indexVars);
  auto& indexVar = this->indexVars.at(iterator);
  return this->getCoordinateVar(indexVar);
}


vector<Expr> LowererImpl::coordinates(Iterator iterator) const
{
  taco_iassert(iterator.defined());

  vector<Expr> coords;
  do {
    coords.push_back(getCoordinateVar(iterator));
    iterator = iterator.getParent();
  } while (iterator.getParent().defined());
  auto reverse = util::reverse(coords);
  return vector<Expr>(reverse.begin(), reverse.end());
}

vector<Expr> LowererImpl::coordinates(vector<Iterator> iterators)
{
  taco_iassert(all(iterators, [](Iterator iter){ return iter.defined(); }));
  vector<Expr> result;
  for (auto& iterator : iterators) {
    result.push_back(iterator.getCoordVar());
  }
  return result;
}

Stmt LowererImpl::initResultArrays(vector<Access> writes)
{
  std::vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0) continue;

    std::vector<Stmt> initArrays;

    const auto iterators = getIterators(write);
    taco_iassert(!iterators.empty());

    Expr tensor = getTensorVar(write.getTensorVar());
    Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);

    Expr parentSize = 1;
    if (generateAssembleCode()) {
      for (const auto& iterator : iterators) {
        Expr size;
        Stmt init;
        if (iterator.hasAppend()) {
          size = 0;
          init = iterator.getAppendInitLevel(parentSize, size);
        } else if (iterator.hasInsert()) {
          size = simplify(ir::Mul::make(parentSize, iterator.getWidth()));
          init = iterator.getInsertInitLevel(parentSize, size);
        } else {
          taco_ierror << "Write iterator supports neither append nor insert";
        }
        initArrays.push_back(init);

        // Declare position variable of append modes
        if (iterator.hasAppend()) {
          initArrays.push_back(VarDecl::make(iterator.getPosVar(), 0));
        }

        parentSize = size;
      }

      // Pre-allocate memory for the value array if computing while assembling
      if (generateComputeCode()) {
        taco_iassert(!iterators.empty());
        
        Expr valsSize = GetProperty::make(tensor, TensorProperty::ValuesSize);
        Expr allocSize = (isa<ir::Literal>(parentSize) &&
                          to<ir::Literal>(parentSize)->equalsScalar(0)) 
                         ? DEFAULT_ALLOC_SIZE : parentSize;
        Stmt assignValsSize = Assign::make(valsSize, allocSize);
        Stmt allocVals = Allocate::make(valuesArr, valsSize);
        initArrays.push_back(Block::make(assignValsSize, allocVals));
      }

      taco_iassert(!initArrays.empty());
      result.push_back(Block::make(initArrays));
    }
    // Declare position variable for the last level
    else if (generateComputeCode()) {
      Iterator lastAppendIterator;
      for (auto& iterator : iterators) {
        if (iterator.hasAppend()) {
          lastAppendIterator = iterator;
          parentSize = iterator.getSize(parentSize);
        } else if (iterator.hasInsert()) {
          parentSize = ir::Mul::make(parentSize, iterator.getWidth());
        } else {
          taco_ierror << "Write iterator supports neither append nor insert";
        }
        parentSize = simplify(parentSize);
      }

      if (lastAppendIterator.defined()) {
        result.push_back(VarDecl::make(lastAppendIterator.getPosVar(), 0));
      }
    }

    // TODO: Do more checks to make sure that vals array actually needs to 
    //       be zero-initialized.
    if (generateComputeCode() && iterators.back().hasInsert() && 
        (!isa<ir::Literal>(parentSize) ||
         !to<ir::Literal>(parentSize)->equalsScalar(0))) {
      result.push_back(zeroInitValues(tensor, 0, parentSize));
    }
  }
  return result.empty() ? Stmt() : Block::blanks(result);
}


ir::Stmt LowererImpl::finalizeResultArrays(std::vector<Access> writes) 
{
  if (!generateAssembleCode()) {
    return Stmt();
  }

  std::vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0) continue;

    const auto iterators = getIterators(write);
    taco_iassert(!iterators.empty());
      
    Expr parentSize = 1;
    for (const auto& iterator : iterators) {
      Expr size;
      Stmt finalize;
      if (iterator.hasAppend()) {
        size = iterator.getPosVar();
        finalize = iterator.getAppendFinalizeLevel(parentSize, size);
      } else if (iterator.hasInsert()) {
        size = simplify(ir::Mul::make(parentSize, iterator.getWidth()));
        finalize = iterator.getInsertFinalizeLevel(parentSize, size);
      } else {
        taco_ierror << "Write iterator supports neither append nor insert";
      }
      result.push_back(finalize);
      parentSize = size;
    }

    if (!generateComputeCode()) {
      Expr tensor = getTensorVar(write.getTensorVar());

      Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);
      Expr valuesSize = GetProperty::make(tensor, TensorProperty::ValuesSize);

      result.push_back(Assign::make(valuesSize, parentSize));
      result.push_back(Allocate::make(valuesArr, valuesSize));
    }
  }
  return result.empty() ? Stmt() : Block::blanks(result);
}


Stmt LowererImpl::declTemporaries(vector<TensorVar> temporaries,
                                  map<TensorVar, Expr> scalars) {
  vector<Stmt> result;
  if (generateComputeCode()) {
    for (auto& temporary : temporaries) {
      if (isScalar(temporary.getType())) {
        taco_iassert(!util::contains(scalars, temporary)) << temporary;
        taco_iassert(util::contains(tensorVars, temporary));
        scalars.insert({temporary, tensorVars.at(temporary)});
        result.push_back(declareScalarArgumentVar(temporary,true,&tensorVars));
      }
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


static
vector<Iterator> getIteratorsFrom(IndexVar var, 
                                  const vector<Iterator>& iterators) {
  vector<Iterator> result;
  bool found = false;
  for (Iterator iterator : iterators) {
    if (var == iterator.getIndexVar()) found = true;
    if (found) {
      result.push_back(iterator);
    }
  }
  return result;
}


Stmt LowererImpl::initResultArrays(IndexVar var, vector<Access> writes) {
  if (!generateAssembleCode()) {
    return Stmt();
  }

  vector<Stmt> result;
  for (auto& write : writes) {
    Expr tensor = getTensorVar(write.getTensorVar());
    Expr values = GetProperty::make(tensor, TensorProperty::Values);
    Expr valuesSizeVar = GetProperty::make(tensor, TensorProperty::ValuesSize);

    vector<Iterator> iterators = getIteratorsFrom(var, getIterators(write));

    if (iterators.empty()) {
      continue;
    }

    Iterator resultIterator = iterators.front();

    // Initialize begin var
    if (resultIterator.hasAppend() && !resultIterator.isBranchless()) {
      Expr begin = resultIterator.getBeginVar();
      Stmt initBegin = VarDecl::make(begin, resultIterator.getPosVar());
      result.push_back(initBegin);
    }

    const bool isTopLevel = (iterators.size() == write.getIndexVars().size());
    if (resultIterator.getParent().hasAppend() || isTopLevel) {
      Expr resultParentPos = resultIterator.getParent().getPosVar();
      Expr initBegin = resultParentPos;
      Expr initEnd = simplify(ir::Add::make(resultParentPos, 1));
      Expr initSize = 1;

      Iterator initIterator;
      for (Iterator iterator : iterators) {
        if (!iterator.hasInsert()) {
          initIterator = iterator;
          break;
        }

        initBegin = simplify(ir::Mul::make(initBegin, iterator.getWidth()));
        initEnd = simplify(ir::Mul::make(initEnd, iterator.getWidth()));
        result.push_back(iterator.getInsertInitCoords(initBegin, initEnd));

        initSize = ir::Mul::make(initSize, iterator.getWidth());
      }

      if (initIterator.defined()) {
        taco_iassert(initIterator.hasAppend());
        result.push_back(initIterator.getAppendInitEdges(initBegin, initEnd));
      } else if (generateComputeCode() && !isTopLevel) {
        result.push_back(doubleSizeIfFull(values, valuesSizeVar, initEnd));
        result.push_back(zeroInitValues(tensor, resultParentPos, initSize));
      }
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::zeroInitValues(Expr tensor, Expr begin, Expr size) {
  std::vector<Stmt> stmts;

  Expr stride = simplify(size);
  LoopKind parallel = (isa<ir::Literal>(stride) && 
                       to<ir::Literal>(stride)->getIntValue() < (1 << 10))
                      ? LoopKind::Serial : LoopKind::Static;

  if (isa<ir::Mul>(stride) && (!isa<ir::Literal>(begin) || 
      !to<ir::Literal>(begin)->equalsScalar(0))) {
    Expr strideVar = Var::make(util::toString(tensor) + "_stride", Int());
    stmts.push_back(VarDecl::make(strideVar, stride));
    stride = strideVar;
  } 

  begin = simplify(ir::Mul::make(begin, stride));
  Expr end = simplify(ir::Mul::make(ir::Add::make(begin, 1), stride));
  Expr p = Var::make("p" + util::toString(tensor), Int());
  Expr values = GetProperty::make(tensor, TensorProperty::Values);
  Stmt zeroInit = Store::make(values, p, ir::Literal::zero(tensor.type()));
  stmts.push_back(For::make(p, begin, end, 1, zeroInit, parallel, false));

  return Block::make(stmts);
}


Stmt LowererImpl::declLocatePosVars(vector<Iterator> locaters) {
  vector<Stmt> result;
  for (Iterator& locateIterator : locaters) {
    ModeFunction locate = locateIterator.locate(coordinates(locateIterator));
    taco_iassert(isValue(locate.getResults()[1], true));
    Stmt declarePosVar = VarDecl::make(locateIterator.getPosVar(),
                                       locate.getResults()[0]);
    result.push_back(declarePosVar);
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::codeToInitializeIteratorVars(vector<Iterator> iterators)
{
  vector<Stmt> result;
  for (Iterator iterator : iterators) {
    taco_iassert(iterator.hasPosIter() || iterator.hasCoordIter() ||
                 iterator.isDimensionIterator());

    if (iterator.hasPosIter()) {
      // E.g. a compressed mode
      ModeFunction bounds = iterator.posBounds();
      result.push_back(bounds.compute());
      result.push_back(VarDecl::make(iterator.getIteratorVar(), bounds[0]));
      result.push_back(VarDecl::make(iterator.getEndVar(), bounds[1]));
    }
    else if (iterator.hasCoordIter()) {
      // E.g. a hasmap mode
      vector<Expr> coords = coordinates(iterator);
      coords.erase(coords.begin());
      ModeFunction bounds = iterator.coordBounds(coords);
      result.push_back(bounds.compute());
      result.push_back(VarDecl::make(iterator.getIteratorVar(), bounds[0]));
      result.push_back(VarDecl::make(iterator.getEndVar(), bounds[1]));
    }
    else if (iterator.isDimensionIterator()) {
      // A dimension
      Expr coord = coordinates(vector<Iterator>({iterator}))[0];
      result.push_back(VarDecl::make(coord, 0));
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::codeToIncIteratorVars(Expr coordinate, vector<Iterator> iterators) {
  if (iterators.size() == 1) {
    Expr ivar = iterators[0].getIteratorVar();
    return Assign::make(ivar, ir::Add::make(ivar, 1));
  }

  vector<Stmt> result;

  // We emit the level iterators before the mode iterator because the coordinate
  // of the mode iterator is used to conditionally advance the level iterators.

  auto levelIterators =
      filter(iterators, [](Iterator it){return !it.isDimensionIterator();});
  for (auto& iterator : levelIterators) {
    Expr ivar = iterator.getIteratorVar();
    Expr increment = (iterator.isFull())
                   ? 1
                   : Cast::make(Eq::make(iterator.getCoordVar(), coordinate),
                                ivar.type());
    result.push_back(Assign::make(ivar, ir::Add::make(ivar, increment)));
  }

  auto modeIterators =
      filter(iterators, [](Iterator it){return it.isDimensionIterator();});
  for (auto& iterator : modeIterators) {
    taco_iassert(iterator.isFull());
    Expr ivar = iterator.getIteratorVar();
    result.push_back(Assign::make(ivar, ir::Add::make(ivar, 1)));
  }

  return Block::make(result);
}


Stmt LowererImpl::generateAppendCoordinate(vector<Iterator> appenders,
                                            Expr coord) {
  vector<Stmt> result;
  if (generateAssembleCode()) {
    for (Iterator appender : appenders) {
      Expr pos = appender.getPosVar();
      Stmt appendCoord = appender.getAppendCoord(pos, coord);
      result.push_back(appendCoord);
    }
  }
  return (result.size() > 0) ? Block::make(result) : Stmt();
}


Stmt LowererImpl::generateAppendPositions(vector<Iterator> appenders) {
  vector<Stmt> result;
  if (generateAssembleCode()) {
    for (Iterator appender : appenders) {
      if (!appender.isBranchless()) {
        Expr pos = appender.getPosVar();
        Expr beginPos = appender.getBeginVar();
        Expr parentPos = appender.getParent().getPosVar();
        Stmt appendPos = appender.getAppendEdges(parentPos, beginPos, pos);
        result.push_back(appendPos);
      }
    }
  }
  return (result.size() > 0) ? Block::make(result) : Stmt();
}


Stmt LowererImpl::generateAppendPosVarIncrements(vector<Iterator> appenders) {
  vector<Stmt> result;
  for (auto& appender : appenders) {
    Expr increment = ir::Add::make(appender.getPosVar(), 1);
    Stmt incrementPos = ir::Assign::make(appender.getPosVar(), increment);
    result.push_back(incrementPos);
  }
  return Block::make(result);
}


Expr LowererImpl::generateValueLocExpr(Access access) const {
  if (isScalar(access.getTensorVar().getType())) {
    return ir::Literal::make(0);
  }
  Iterator it = getIterators(access).back();
  return it.getPosVar();
}


Expr LowererImpl::checkThatNoneAreExhausted(std::vector<Iterator> iterators)
{
  taco_iassert(!iterators.empty());
  if (iterators.size() == 1 && iterators[0].isFull()) {
    Expr dimension = getDimension(iterators[0].getIndexVar());
    return Lt::make(iterators[0].getIteratorVar(), dimension);
  }

  vector<Expr> result;
  for (const auto& iterator : iterators) {
    taco_iassert(!iterator.isFull()) << iterator
        << " - full iterators do not need to partake in merge loop bounds";
    Expr iterUnexhausted = Lt::make(iterator.getIteratorVar(),
                                    iterator.getEndVar());
    result.push_back(iterUnexhausted);
  }

  return (!result.empty())
         ? conjunction(result)
         : Lt::make(iterators[0].getIteratorVar(), iterators[0].getEndVar());
}

}
