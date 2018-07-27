#include "taco/lower/lower.h"

#include <algorithm>
#include <vector>
#include <list>
#include <set>
#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"

#include "taco/ir/ir.h"
#include "taco/ir/simplify.h"
#include "ir/ir_generators.h"

#include "taco/lower/lowerer_impl.h"
#include "iterator.h"
#include "mode_access.h"
#include "merge_lattice.h"

#include "error/error_checks.h"
#include "taco/error/error_messages.h"
#include "taco/util/name_generator.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {


// class Lowerer
Lowerer::Lowerer() : impl(new LowererImpl()) {
}

Lowerer::Lowerer(LowererImpl* impl) : impl(impl) {
}

std::shared_ptr<LowererImpl> Lowerer::getLowererImpl() {
  return impl;
}

struct Context {
  // Configuration options
  bool assemble;
  bool compute;

  // Map from tensor variables in index notation to variables in the IR
  map<TensorVar, Expr> tensorVars;

  // Map from index variables to their ranges, currently [0, expr>.
  map<IndexVar, Expr> ranges;

  // Map from mode accesses to iterators.
  map<ModeAccess, Iterator> iterators;

  // Map from iterators to the index variables they contribute to.
  map<Iterator, IndexVar> indexVars;

  // Map from index variables to corresponding resolved coordinate variable.
  map<IndexVar, Expr> coordVars;
};


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
static Stmt declareScalarArgumentVar(TensorVar var, bool zero, Context* ctx) {
  Datatype type = var.getType().getDataType();
  Expr varValueIR = Var::make(var.getName() + "_val", type, false, false);
  Expr init = (zero) ? ir::Literal::zero(type)
                     : Load::make(GetProperty::make(ctx->tensorVars.at(var),
                                                    TensorProperty::Values));
  ctx->tensorVars.find(var)->second = varValueIR;
  return VarAssign::make(varValueIR, init, true);
}


/// Create an expression to index into a tensor value array.
static Expr locExpr(const AccessNode* node, Context* ctx) {
  if (isScalar(node->tensorVar.getType())) {
    return ir::Literal::make(0);
  }
  taco_iassert(util::contains(ctx->iterators,
                              ModeAccess(node, node->indexVars.size())));
  Iterator it = ctx->iterators.at(ModeAccess(Access(node),
                                             node->indexVars.size()));
  return it.getPosVar();
}


/// Lower an index expression to IR.
static Expr lower(const IndexExpr& expr, Context* ctx) {
  struct Lower : IndexExprVisitorStrict {
    using IndexExprVisitorStrict::visit;
    Context* ctx;
    Expr ir;
    Lower(Context* ctx) : ctx(ctx) {}
    Expr rewrite(IndexExpr expr) {
      visit(expr);
      return ir;
    }

    void visit(const AccessNode* node) {
      taco_iassert(util::contains(ctx->tensorVars, node->tensorVar));
      TensorVar var = node->tensorVar;
      Expr varIR = ctx->tensorVars.at(node->tensorVar);
      if (isScalar(var.getType())) {
        ir = varIR;
      }
      else {
        ir = Load::make(GetProperty::make(varIR, TensorProperty::Values),
                        locExpr(node, ctx));
      }
    }

    void visit(const LiteralNode* node) {
      taco_not_supported_yet;
    }

    void visit(const NegNode* node) {
      ir = ir::Neg::make(rewrite(node->a));
    }

    void visit(const AddNode* node) {
      ir = ir::Add::make(rewrite(node->a), rewrite(node->b));
    }

    void visit(const SubNode* node) {
      ir = ir::Sub::make(rewrite(node->a), rewrite(node->b));
    }

    void visit(const MulNode* node) {
      ir = ir::Mul::make(rewrite(node->a), rewrite(node->b));
    }

    void visit(const DivNode* node) {
      ir = ir::Div::make(rewrite(node->a), rewrite(node->b));
    }

    void visit(const SqrtNode* node) {
      ir = ir::Sqrt::make(rewrite(node->a));
    }

    void visit(const ReductionNode* node) {
      taco_ierror << "Reduction nodes not supported in concrete index notation";
    }
  };
  return Lower(ctx).rewrite(expr);
}


/// Lower an index statement to IR.
static Stmt lower(const IndexStmt& stmt, Context* ctx) {
  struct Lowerer : IndexStmtVisitorStrict {
    using IndexStmtVisitorStrict::visit;

    Context* ctx;
    Stmt ir;

    Lowerer(Context* ctx) : ctx(ctx) {}

    Stmt rewrite(IndexStmt stmt) {
      visit(stmt);
      return ir;
    }

    void visit(const AssignmentNode* node) {
      TensorVar result = node->lhs.getTensorVar();
      if (ctx->compute) {
        taco_iassert(util::contains(ctx->tensorVars, node->lhs.getTensorVar()))
            << node->lhs.getTensorVar();
        ir::Expr varIR = ctx->tensorVars.at(result);
        ir::Expr rhs = lower(node->rhs, ctx);

        // Assignment to scalar variables.
        if (isScalar(result.getType())) {
          if (!node->op.defined()) {
            ir = VarAssign::make(varIR, rhs);
          }
          else {
            taco_iassert(isa<taco::Add>(node->op));
            ir = VarAssign::make(varIR, ir::Add::make(varIR,rhs));
          }
        }
        // Assignments to tensor variables (non-scalar).
        else {
          Expr valueArray = GetProperty::make(varIR, TensorProperty::Values);
          ir = ir::Store::make(valueArray,
                               locExpr(to<AccessNode>(node->lhs.ptr),ctx),
                               rhs);
          // When we're assembling while computing we need to allocate more
          // value memory as we write to the values array.
          if (ctx->assemble) {
            // TODO
          }
        }
      }
      // We're only assembling so defer allocating value memory to the end when
      // we'll know exactly how much we need.
      else if (ctx->assemble) {
        // TODO
        ir = Block::make();
      }
      // We're neither assembling or computing so we emit nothing.
      else {
        ir = Block::make();
      }
      taco_iassert(ir.defined()) << Assignment(node);
    }

    vector<Expr> getCoords(Iterator iterator) {
      vector<Expr> coords;
      do {
        coords.push_back(ctx->coordVars.at(ctx->indexVars.at(iterator)));
        iterator = iterator.getParent();
      } while (iterator.getParent().defined());
      util::reverse(coords);
      return coords;
    }

    Stmt makePosVarLocateDecls(vector<Iterator> iterators) {
      vector<Stmt> posVarDecls;
      for (Iterator& iterator : iterators) {
        ModeFunction locate = iterator.locate(getCoords(iterator));
        taco_iassert(isValue(locate.getResults()[1], true));
        Stmt posVarDecl = VarAssign::make(iterator.getPosVar(),
                                          locate.getResults()[0],
                                          true);
        posVarDecls.push_back(posVarDecl);
      }
      return Block::make(posVarDecls);
    }

    /// Make a loop that iterates over all the coordinates in the dimension
    /// of the forall's index variable.  Positions of tensors are located from
    /// the locate iterators.
    Stmt makeDimensionLoop(Forall forall,
                           const vector<Iterator>& locateIterators) {
      IndexVar  indexVar  = forall.getIndexVar();
      IndexStmt indexStmt = forall.getStmt();
      Expr coordVar = ctx->coordVars.at(indexVar);
      Stmt posVarDecls = makePosVarLocateDecls(locateIterators);
      Stmt body = rewrite(indexStmt);
      return For::make(coordVar, 0, ctx->ranges.at(indexVar), 1,
                       Block::make({posVarDecls, body}));
    }

    /// Make a loop that
    Stmt makePosLoop(Forall forall, Iterator iterator,
                     vector<Iterator> locateIterators) {
      IndexVar  indexVar  = forall.getIndexVar();
      IndexStmt indexStmt = forall.getStmt();
      Expr coordVar = ctx->coordVars.at(indexVar);
      ModeFunction access = iterator.posAccess(getCoords(iterator));
      Stmt coordVarDecl = VarAssign::make(coordVar, access.getResults()[0], true);
      Stmt posVarDecls = makePosVarLocateDecls(locateIterators);
      Stmt body = rewrite(indexStmt);
      ModeFunction bounds = iterator.posBounds();
      return Block::make({bounds.getBody(),
                          For::make(iterator.getPosVar(),
                                    bounds.getResults()[0],
                                    bounds.getResults()[1], 1,
                                    Block::make({coordVarDecl,
                                                 posVarDecls,
                                                 body}))});
    }

    Stmt makeMergeLoops(Forall forall, MergeLattice lattice) {
      IndexVar  indexVar  = forall.getIndexVar();
      IndexStmt indexStmt = forall.getStmt();
      Expr coordVar = ctx->coordVars.at(indexVar);

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

    void visit(const ForallNode* node) {
      MergeLattice lattice = MergeLattice::make(node, ctx->iterators);

      // Emit a loop that iterates over over a single iterator (optimization)
      if (lattice.getRangeIterators().size() == 1) {
        Iterator rangeIterator = lattice.getMergeIterators()[0];
        // Emit dimension coordinate iteration loop
        if (rangeIterator.isFull() && rangeIterator.hasLocate()) {
          ir = makeDimensionLoop(node,
                                 util::combine(lattice.getMergeIterators(),
                                               lattice.getResultIterators()));
        }
        // Emit position iteration loop
        else if (rangeIterator.hasPosIter()) {
          auto locateIterators = util::combine(lattice.getMergeIterators(),
                                               lattice.getResultIterators());
          locateIterators.erase(remove(locateIterators.begin(),
                                       locateIterators.end(),
                                       rangeIterator), locateIterators.end());
          ir = makePosLoop(node, rangeIterator, locateIterators);
        }
        // Emit coordinate iteration loop
        else {
          taco_iassert(rangeIterator.hasCoordIter());
          taco_not_supported_yet;
        }
      }
      // Emit general loops to merge multiple iterators
      else {
        ir = makeMergeLoops(node, lattice);
      }
    }

    void visit(const WhereNode* node) {
      ir::Stmt producer = rewrite(node->producer);
      ir::Stmt consumer = rewrite(node->consumer);
      ir = Block::make({producer, consumer});
      // TODO: Either initialise or re-initialize temporary memory
    }

    void visit(const SequenceNode* node) {
      ir::Stmt definition = rewrite(node->definition);
      ir::Stmt mutation = rewrite(node->mutation);
      ir = Block::make({definition, mutation});
    }

    void visit(const MultiNode* node) {
      ir::Stmt stmt1 = rewrite(node->stmt1);
      ir::Stmt stmt2 = rewrite(node->stmt2);
      ir = Block::make({stmt1, stmt2});
    }
  };
  return Lowerer(ctx).rewrite(stmt);
}


ir::Stmt lower(IndexStmt stmt, std::string name, bool assemble, bool compute,
               Lowerer lowerer) {
  taco_iassert(isLowerable(stmt));

  // Create context
  Context ctx;
  ctx.assemble = assemble;
  ctx.compute  = compute;

  // Create result and parameter variables
  vector<TensorVar> results = getResultTensorVars(stmt);
  vector<TensorVar> arguments = getInputTensorVars(stmt);
  vector<TensorVar> temporaries = getTemporaryTensorVars(stmt);

  // Convert tensor results, arguments and temporaries to IR variables
  map<TensorVar, Expr> resultVars;
  vector<Expr> resultsIR = createVars(results, &resultVars);
  ctx.tensorVars.insert(resultVars.begin(), resultVars.end());
  vector<Expr> argumentsIR = createVars(arguments, &ctx.tensorVars);
  vector<Expr> temporariesIR = createVars(temporaries, &ctx.tensorVars);

  // Create iterators
  createIterators(stmt, ctx.tensorVars,
                  &ctx.iterators, &ctx.indexVars, &ctx.coordVars);

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
          int loc = std::distance(ivars.begin(),
                                  std::find(ivars.begin(),ivars.end(), ivar));
          dimension = GetProperty::make(ctx.tensorVars.at(n->lhs.getTensorVar()),
                                        TensorProperty::Dimension, loc);
        }
      }),
      function<void(const AccessNode*)>([&](const AccessNode* n) {
        auto ivars = n->indexVars;
        int loc = std::distance(ivars.begin(),
                                std::find(ivars.begin(),ivars.end(), ivar));
        dimension = GetProperty::make(ctx.tensorVars.at(n->tensorVar),
                                      TensorProperty::Dimension, loc);
      })
    );
    Expr ivarIR = Var::make(ivar.getName() + "_size", type<int32_t>());
    Stmt decl = VarAssign::make(ivarIR, dimension, true);
    ctx.ranges.insert({ivar, ivarIR});
    headerStmts.push_back(decl);
  }

  // Declare and initialize scalar results and arguments
  if (ctx.compute) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(!util::contains(scalars, result));
        taco_iassert(util::contains(ctx.tensorVars, result));
        scalars.insert({result, ctx.tensorVars.at(result)});
        headerStmts.push_back(declareScalarArgumentVar(result, true, &ctx));
      }
    }
    for (auto& argument : arguments) {
      if (isScalar(argument.getType())) {
        taco_iassert(!util::contains(scalars, argument));
        taco_iassert(util::contains(ctx.tensorVars, argument));
        scalars.insert({argument, ctx.tensorVars.at(argument)});
        headerStmts.push_back(declareScalarArgumentVar(argument, false, &ctx));
      }
    }
  }

  // Declare, allocate, and initialize temporaries.
  if (ctx.compute) {
    for (auto& temporary : temporaries) {
      if (isScalar(temporary.getType())) {
        taco_iassert(!util::contains(scalars, temporary)) << temporary;
        taco_iassert(util::contains(ctx.tensorVars, temporary));
        scalars.insert({temporary, ctx.tensorVars.at(temporary)});
        headerStmts.push_back(declareScalarArgumentVar(temporary, true, &ctx));
      }
    }
  }

  // Allocate memory for dense results up front.
  if (ctx.assemble) {
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
  Stmt body = lower(stmt, &ctx);

  // Store scalar stack variables back to results.
  if (ctx.compute) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(util::contains(scalars, result));
        taco_iassert(util::contains(ctx.tensorVars, result));
        Expr resultIR = scalars.at(result);
        Expr varValueIR = ctx.tensorVars.at(result);
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


bool isLowerable(IndexStmt stmt, std::string* reason) {
  INIT_REASON(reason);

  // Must be concrete index notation
  if (!isConcreteNotation(stmt)) {
    *reason = "The index statement is not in concrete index notation";
    return false;
  }

  // Check for transpositions
//  if (!error::containsTranspose(this->getFormat(), freeVars, indexExpr)) {
//    *reason = error::expr_transposition;
//  }

  return true;
}

/// Prints the hierarchy of merge cases that result from lowering `stmt`.
void printMergeCaseHierarchy(IndexStmt stmt, std::ostream& os) {

}

}
