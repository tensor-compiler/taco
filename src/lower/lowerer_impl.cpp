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
  iterators = createIterators(stmt, tensorVars, &indexVars, &coordVars);

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
  Stmt initResultArrays = generateInitResultArrays(getResultAccesses(stmt));

  // Declare, allocate, and initialize temporaries
  Stmt declareTemporaries = generateTemporaryDecls(temporaries, scalars);

  // Lower the index statement to compute and/or assemble
  Stmt body = lower(stmt);

  // Post-process result modes.
  Stmt finalizeResultModes = generateModeFinalizes(getResultAccesses(stmt));

  // If assembling without computing then allocate value memory at the end
  Stmt postAllocValues = generatePostAllocValues(getResultAccesses(stmt));

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

  // Create function
  Stmt header = (headerStmts.size() > 0) ? Block::make(headerStmts) : Stmt();
  Stmt footer = (footerStmts.size() > 0) ? Block::make(footerStmts) : Stmt();
  return Function::make(name, resultsIR, argumentsIR,
                        Block::blanks({header,
                                       declareTemporaries,
                                       initResultArrays,
                                       body,
                                       finalizeResultModes,
                                       postAllocValues,
                                       footer}));
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
             ? Block::make({resizeValueArray,  computeStmt})
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

Stmt LowererImpl::lowerForall(Forall forall) {
  MergeLattice lattice = MergeLattice::make(forall, getIteratorMap());

  // Emit a loop that iterates over over a single iterator (optimization)
  if (lattice.getPoints().size() == 1 && lattice.getIterators().size() == 1) {
    MergePoint point = lattice.getPoints()[0];
    Iterator iterator = lattice.getIterators()[0];

    vector<Iterator> locaters = point.getLocators();
    vector<Iterator> appenders;
    vector<Iterator> inserters;
    tie(appenders, inserters) = splitAppenderAndInserters(point.getResults());

    // Emit dimension coordinate iteration loop
    if (iterator.isDimensionIterator()) {
      return lowerForallDimension(forall, point.getLocators(),
                                  inserters, appenders);
    }
    // Emit position iteration loop
    else if (iterator.hasPosIter()) {
      return lowerForallPosition(forall, iterator, locaters,
                                 inserters, appenders);
    }
    // Emit coordinate iteration loop
    else {
      taco_iassert(iterator.hasCoordIter());
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
                                       vector<Iterator> locators,
                                       vector<Iterator> inserters,
                                       vector<Iterator> appenders) {
  Expr coordinate = getCoordinateVar(forall.getIndexVar());

  Stmt header = lowerForallHeader(forall, locators, inserters, appenders);
  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders);
  Stmt footer = lowerForallFooter(forall, locators, inserters, appenders);

  // Emit loop with preamble and postamble
  Expr dimension = getDimension(forall.getIndexVar());
  return Block::blanks({header,
                        For::make(coordinate, 0, dimension, 1, body),
                        footer
                       });
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
                                      vector<Iterator> appenders) {
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  Expr coordinateArray= iterator.posAccess(getCoords(iterator)).getResults()[0];
  Stmt declareCoordinate = VarDecl::make(coordinate, coordinateArray);

  Stmt header = lowerForallHeader(forall, locators, inserters, appenders);
  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders);
  Stmt footer = lowerForallFooter(forall, locators, inserters, appenders);

  // Loop with preamble and postamble
  ModeFunction bounds = iterator.posBounds();
  return Block::blanks({header,
                        bounds.compute(),
                        For::make(iterator.getPosVar(), bounds[0], bounds[1], 1,
                                  Block::make({declareCoordinate, body})),
                        footer
                       });
}

Stmt LowererImpl::lowerForallMerge(Forall forall, MergeLattice lattice) {
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  vector<Iterator> iterators = lattice.getIterators();

  // Declare and initialize range iterator position variables
  Stmt declPosVarIterators = generateDeclPosVarIterators(iterators);

  // One loop for each merge lattice point lp
  Stmt loops = lowerMergeLoops(coordinate, forall.getStmt(), lattice);

  return Block::make({declPosVarIterators, loops});
}

Stmt LowererImpl::lowerMergeLoops(ir::Expr coordinate, IndexStmt stmt,
                                   MergeLattice lattice) {
  vector<Stmt> result;
  for (MergePoint point : lattice.getPoints()) {
    MergeLattice sublattice = lattice.subLattice(point);
    Stmt mergeLoop = lowerMergeLoop(coordinate, stmt, sublattice);
    result.push_back(mergeLoop);
  }
  return Block::make(result);
}

Stmt LowererImpl::lowerMergeLoop(ir::Expr coordinate, IndexStmt stmt,
                                 MergeLattice lattice) {

  vector<Iterator> iterators = lattice.getIterators();

  // Merge range iterator coordinate variables
  Stmt mergeCoordinates = generateMergeCoordinates(coordinate, iterators);

  // Emit located position variables
  // TODO

  // One case for each child lattice point lp
  Stmt cases = lowerMergeCases(coordinate, stmt, lattice);

  /// While loop over rangers
  return While::make(generateNoneExhausted(iterators),
                     Block::make({mergeCoordinates, cases}));
}

Stmt LowererImpl::lowerMergeCases(ir::Expr coordinate, IndexStmt stmt,
                                  MergeLattice lattice) {
  vector<Stmt> result;

  // Just one iterator so no conditionals
  if (lattice.getIterators().size() == 1) {
    Stmt body = Comment::make("...");
    vector<Iterator> appenders;
    vector<Iterator> inserters;
    tie(appenders, inserters) = splitAppenderAndInserters(lattice.getResults());
    result.push_back(lowerForallBody(coordinate, stmt, {},inserters,appenders));
  }
  else {
    vector<pair<Expr,Stmt>> cases;
    for (MergePoint point : lattice.getPoints()) {
      vector<Iterator> iterators = point.getIterators();

      Stmt body = Stmt();

      if (iterators.size() == 1) {
        cases.push_back({true, body});
      }
      else {
        // Conditionals to execute code for the intersection cases
        vector<Expr> coordComparisons;
        for (Iterator iterator : iterators) {
          coordComparisons.push_back(Eq::make(iterator.getCoordVar(), coordinate));
        }
        Expr expr = conjunction(coordComparisons);
        cases.push_back({expr, body});
      }
    }
  }

  return Block::make(result);
}

Stmt LowererImpl::lowerForallBody(Expr coordinate, IndexStmt stmt,
                                  vector<Iterator> locators,
                                  vector<Iterator> inserters,
                                  vector<Iterator> appenders) {

  // Insert positions
  Stmt declInserterPosVars = generateDeclLocatePosVars(inserters);

  // Locate positions
  Stmt declLocatorPosVars = generateDeclLocatePosVars(locators);

  // Code of loop body statement
  Stmt body = lower(stmt);

  // Code to append coordinates
  Stmt appendCoordinate = generateAppendCoordinate(appenders, coordinate);

  // Code to increment append position variables
  Stmt incrementAppendPositionVars = generateAppendPosVarIncrements(appenders);

  return Block::make({declInserterPosVars,
                      declLocatorPosVars,
                      body,
                      appendCoordinate,
                      incrementAppendPositionVars
                     });
}

Stmt LowererImpl::lowerForallHeader(Forall forall,
                                    vector<Iterator> locaters,
                                    vector<Iterator> inserters,
                                    vector<Iterator> appenders) {
  // Pre-allocate/initialize memory of value arrays that are full below this
  // loops index variable
  Stmt preInitValues = generatePreInitValues(forall.getIndexVar(),
                                             getResultAccesses(forall));
  return preInitValues;
}

  /// Lower a forall loop footer.
Stmt LowererImpl::lowerForallFooter(Forall forall,
                                    vector<Iterator> locaters,
                                    vector<Iterator> inserters,
                                    vector<Iterator> appenders) {
  // Code to append positions
  Stmt appendPositions = generateAppendPositions(appenders);
  return appendPositions;
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
  return getIteratorMap().at(modeAccess);
}

std::vector<Iterator> LowererImpl::getIterators(Access access) const {
  vector<Iterator> result;
  TensorVar tensor = access.getTensorVar();
  for (int i = 0; i < tensor.getOrder(); i++) {
    int mode = tensor.getFormat().getModeOrdering()[i];
    result.push_back(getIterator(ModeAccess(access, mode+1)));
  }
  return result;
}

const map<ModeAccess, Iterator>& LowererImpl::getIteratorMap() const {
  return this->iterators;
}

Expr LowererImpl::getCoordinateVar(IndexVar indexVar) const {
  taco_iassert(util::contains(this->coordVars, indexVar)) << indexVar;
  return this->coordVars.at(indexVar);
}

Expr LowererImpl::getCoordinateVar(Iterator iterator) const {
  taco_iassert(util::contains(this->indexVars, iterator)) << iterator;
  auto& indexVar = this->indexVars.at(iterator);
  return this->getCoordinateVar(indexVar);
}

vector<Expr> LowererImpl::getCoords(Iterator iterator) const {
  vector<Expr> coords;
  do {
    coords.push_back(getCoordinateVar(iterator));
    iterator = iterator.getParent();
  } while (iterator.getParent().defined());
  util::reverse(coords);
  return coords;
}

vector<Expr> LowererImpl::getCoords(vector<Iterator> iterators) {
  vector<Expr> result;
  for (auto& iterator : iterators) {
    result.push_back(iterator.getCoordVar());
  }
  return result;
}

Stmt LowererImpl::generateInitResultArrays(vector<Access> writes) {
  vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0) continue;

    vector<Stmt> initArrays;
    Expr parentSize = 1;
    auto iterators = getIterators(write);
    if (generateAssembleCode()) {
      for (auto& iterator : iterators) {
        Expr size;
        Stmt init;
        if (iterator.hasAppend()) {
          size = 0;
          init = iterator.getAppendInitLevel(parentSize, size);
        }
        else if (iterator.hasInsert()) {
          size = ir::Mul::make(parentSize, iterator.getSize());
          init = iterator.getInsertInitLevel(parentSize, size);
        }
        else {
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
      if (generateComputeCode() && !isDense(write.getTensorVar().getFormat())) {
        taco_iassert(iterators.size() > 0);
        Iterator lastIterator = iterators.back();

        Expr tensor = getTensorVar(write.getTensorVar());

        Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);
        Expr valsSize = GetProperty::make(tensor, TensorProperty::ValuesSize);

        Stmt assignValsSize = Assign::make(valsSize, DEFAULT_ALLOC_SIZE);
        Stmt allocVals = Allocate::make(valuesArr, valsSize);

        initArrays.push_back(Block::make({assignValsSize, allocVals}));
      }

      taco_iassert(initArrays.size() > 0);
      result.push_back(Block::make(initArrays));
    }
    // Declare position variable for the last level
    else if (generateComputeCode()) {
      result.push_back(VarDecl::make(iterators.back().getPosVar(), 0));
    }
  }
  return (result.size() > 0) ? Block::blanks(result) : Stmt();
}

ir::Stmt LowererImpl::generateModeFinalizes(std::vector<Access> writes) {
  vector<Stmt> result;
  return (result.size() > 0) ? Block::make(result) : Stmt();
}

Stmt LowererImpl::generateTemporaryDecls(vector<TensorVar> temporaries,
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
  return (result.size() > 0) ? Block::make(result) : Stmt();
}

static
vector<Iterator> getIteratorsFrom(IndexVar var, vector<Iterator> iterators) {
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

static bool allInsert(vector<Iterator> iterators) {
  for (Iterator iterator : iterators) {
    if (!iterator.hasInsert()) {
      return false;
    }
  }
  return true;
}

Stmt LowererImpl::generatePreInitValues(IndexVar var, vector<Access> writes) {
  vector<Stmt> result;

  for (auto& write : writes) {
    Expr tensor = getTensorVar(write.getTensorVar());
    Expr values = GetProperty::make(tensor, TensorProperty::Values);
    Expr valuesSizeVar = GetProperty::make(tensor, TensorProperty::ValuesSize);

    vector<Iterator> iterators = getIteratorsFrom(var, getIterators(write));
    taco_iassert(iterators.size() > 0);
    if (!allInsert(iterators)) continue;

    Expr size = iterators[0].getSize();
    for (size_t i = 1; i < iterators.size(); i++) {
      size = ir::Mul::make(size, iterators[i].getSize());
    }

    if (generateAssembleCode()) {
      // Allocate value memory
      result.push_back(Assign::make(valuesSizeVar, size));
      result.push_back(Allocate::make(values, valuesSizeVar));
    }

    if (generateComputeCode()) {
      Expr i = Var::make(var.getName() + "z", Int());
      result.push_back(For::make(i, 0,size,1, Store::make(values, i, 0.0)));
    }
  }

  return (result.size() > 0) ? Block::make(result) : Stmt();
}

Stmt LowererImpl::generateDeclLocatePosVars(vector<Iterator> locaters) {
  vector<Stmt> result;
  for (Iterator& locateIterator : locaters) {
    ModeFunction locate = locateIterator.locate(getCoords(locateIterator));
    taco_iassert(isValue(locate.getResults()[1], true));
    Stmt declarePosVar = VarDecl::make(locateIterator.getPosVar(),
                                       locate.getResults()[0]);
    result.push_back(declarePosVar);
  }
  return (result.size() > 0) ? Block::make(result) : Stmt();
}

Stmt LowererImpl::generateDeclPosVarIterators(vector<Iterator> iterators) {
  vector<Stmt> result;
  for (Iterator iterator : iterators) {
    taco_iassert(iterator.hasPosIter());
    ModeFunction bounds = iterator.posBounds();
    result.push_back(bounds.compute());
    result.push_back(VarDecl::make(iterator.getIteratorVar(), bounds[0]));
    result.push_back(VarDecl::make(iterator.getEndVar(), bounds[1]));
  }
  return (result.size() > 0) ? Block::make(result) : Stmt();
}

Stmt LowererImpl::generateMergeCoordinates(Expr coordinate,
                                           vector<Iterator> iterators) {
  taco_iassert(iterators.size() > 0);

  /// Just one iterator so it's coordinate var is the resolved coordinate.
  if (iterators.size() == 1) {
    ModeFunction posAccess = iterators[0].posAccess(getCoords(iterators[0]));
    return Block::make({posAccess.compute(),
                        VarDecl::make(coordinate, posAccess[0])
                       });
  }

  // Multiple iterators so we compute the min of their coordinate variables.
  vector<Stmt> result;
  vector<Expr> iteratorCoordVars;
  for (Iterator iterator : iterators) {
    taco_iassert(iterator.hasPosIter());
    ModeFunction posAccess = iterator.posAccess(getCoords(iterator));
    result.push_back(posAccess.compute());
    result.push_back(VarDecl::make(iterator.getCoordVar(), posAccess[0]));
  }
  result.push_back(VarDecl::make(coordinate, Min::make(getCoords(iterators))));
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
      Expr pos = appender.getPosVar();
      Expr parentPos = appender.getParent().getPosVar();
      Stmt appendPos = appender.getAppendEdges(parentPos, ir::Sub::make(pos,1),
                                               pos);
      result.push_back(appendPos);
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

Stmt LowererImpl::generatePostAllocValues(vector<Access> writes) {
  if (generateComputeCode() || !generateAssembleCode()) {
    return Stmt();
  }

  vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0) continue;

    auto iterators = getIterators(write);
    taco_iassert(iterators.size() > 0);
    Iterator lastIterator = iterators[0];

    if (lastIterator.hasAppend()) {
      Expr tensor = getTensorVar(write.getTensorVar());

      Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);
      Expr valuesSize = GetProperty::make(tensor, TensorProperty::ValuesSize);

      result.push_back(Assign::make(valuesSize, lastIterator.getPosVar()));
      result.push_back(Allocate::make(valuesArr, valuesSize));
    }
  }
  return Block::make(result);
}


Expr LowererImpl::generateValueLocExpr(Access access) const {
  if (isScalar(access.getTensorVar().getType())) {
    return ir::Literal::make(0);
  }
  int loc = (int)access.getIndexVars().size();
  Iterator it = getIterator(ModeAccess(access, loc));
  return it.getPosVar();
}

Expr LowererImpl::generateNoneExhausted(std::vector<Iterator> iterators) {
  taco_iassert(!iterators.empty());

  vector<Expr> result;
  for (const auto& iterator : iterators) {
    taco_iassert(!iterator.isFull());
    Expr iterUnexhausted = Lt::make(iterator.getIteratorVar(),
                                    iterator.getEndVar());
    result.push_back(iterUnexhausted);
  }
  return (!result.empty())
         ? conjunction(result)
         : Lt::make(iterators[0].getIteratorVar(), iterators[0].getEndVar());
}

}
