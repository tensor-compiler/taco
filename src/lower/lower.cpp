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

#include "storage/iterator.h"
#include "merge_lattice.h"

#include "error/error_checks.h"
#include "taco/error/error_messages.h"
#include "taco/util/name_generator.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

struct Context {
  // Configuration options
  bool assemble;
  bool compute;

  // Map from tensor variables in index notation to variables in the IR
  map<TensorVar, Expr> vars;

  // Map from index variables to their ranges, currently [0, expr>.
  map<IndexVar, Expr> ranges;

  // Map from locations in access expressions to iterators.
  map<pair<Access,int>, Iterator> iterators;
};


/// Convert index notation tensor variables to IR pointer variables.
static vector<Expr> createIRVars(const vector<TensorVar>& tensorVars,
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
                     : Load::make(GetProperty::make(ctx->vars.at(var),
                                                    TensorProperty::Values));
  ctx->vars.find(var)->second = varValueIR;
  return VarAssign::make(varValueIR, init, true);
}


/// Create iterators
void createIterators(IndexStmt stmt, Context* ctx) {
  list<IndexVar> order;
  match(stmt,
    function<void(const ForallNode*, Matcher*)>([&](const ForallNode* n,
                                                    Matcher* m) {
      order.push_back(n->indexVar);
      m->match(n->stmt);
      order.pop_back();
    }),
    function<void(const AccessNode*)>([&](const AccessNode* n) {
      taco_iassert(util::contains(ctx->vars, n->tensorVar));
      Expr tensorVarIR = ctx->vars.at(n->tensorVar);
      Format format = n->tensorVar.getFormat();
      set<IndexVar> vars(n->indexVars.begin(), n->indexVars.end());

      Iterator root = Iterator::makeRoot(tensorVarIR);
      ctx->iterators.insert({{Access(n),0}, root});

      Iterator parent = root;
      int i = 0;
      for (auto& var : order) {
        if (util::contains(vars, var)) {
          string iteratorName = n->tensorVar.getName()+util::toString(i+1);
          Iterator iterator = Iterator::make(var.getName(), tensorVarIR,
                                             format.getModeTypes()[i], parent,
                                             iteratorName);
          ctx->iterators.insert({{Access(n),i+1}, iterator});
        }
        i++;
      }
    }),
    function<void(const AssignmentNode*,Matcher*)>([&](const AssignmentNode* n,
                                                       Matcher* m) {
      m->match(n->rhs);
      m->match(n->lhs);
    })
  );
}


/// Create an expression to index into a tensor value array.
static Expr locExpr(const AccessNode* node, Context* ctx) {
  if (node->indexVars.size() == 0) {
    return ir::Literal::make(0);
  }
  else {
    taco_iassert(util::contains(ctx->iterators, {node, node->indexVars.size()}))
        << Access(node);
    Iterator it = ctx->iterators.at({Access(node), node->indexVars.size()});
    return it.getPosVar();
  }
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
      taco_iassert(util::contains(ctx->vars, node->tensorVar));
      TensorVar var = node->tensorVar;
      Expr varIR = ctx->vars.at(node->tensorVar);
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
  struct Lower : IndexStmtVisitorStrict {
    using IndexStmtVisitorStrict::visit;
    Context* ctx;
    Stmt ir;
    Lower(Context* ctx) : ctx(ctx) {}
    Stmt rewrite(IndexStmt stmt) {
      visit(stmt);
      return ir;
    }

    void visit(const AssignmentNode* node) {
      TensorVar result = node->lhs.getTensorVar();
      if (ctx->compute) {
        taco_iassert(util::contains(ctx->vars, node->lhs.getTensorVar()))
            << node->lhs.getTensorVar();
        ir::Expr varIR = ctx->vars.at(result);
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
          // When we're assembling while computing we need to allocate more value
          // memory as we write to the values array.
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

    void visit(const ForallNode* node) {
      IndexVar indexVar = node->indexVar;
      Expr i = Var::make(indexVar.getName(), type<int32_t>());
      taco_iassert(util::contains(ctx->ranges, indexVar));
      ir::Stmt body = rewrite(node->stmt);
      ir = For::make(i, 0, ctx->ranges.at(indexVar), 1, body);
    }

    void visit(const WhereNode* node) {
      ir::Stmt producer = rewrite(node->producer);
      ir::Stmt consumer = rewrite(node->consumer);
      ir = Block::make({producer, consumer});
      // TODO: Re-initialize temporary memory
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
  return Lower(ctx).rewrite(stmt);
}


Stmt lower(IndexStmt stmt, std::string name, bool assemble, bool compute) {
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
  vector<Expr> resultsIR = createIRVars(results, &resultVars);
  ctx.vars.insert(resultVars.begin(), resultVars.end());
  vector<Expr> argumentsIR = createIRVars(arguments, &ctx.vars);
  vector<Expr> temporariesIR = createIRVars(temporaries, &ctx.vars);

  // Create iterators
  createIterators(stmt, &ctx);

  // Emit code
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
          dimension = GetProperty::make(ctx.vars.at(n->lhs.getTensorVar()),
                                        TensorProperty::Dimension, loc);
        }
      }),
      function<void(const AccessNode*)>([&](const AccessNode* n) {
        auto ivars = n->indexVars;
        int loc = std::distance(ivars.begin(),
                                std::find(ivars.begin(),ivars.end(), ivar));
        dimension = GetProperty::make(ctx.vars.at(n->tensorVar),
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
        taco_iassert(util::contains(ctx.vars, result));
        scalars.insert({result, ctx.vars.at(result)});
        headerStmts.push_back(declareScalarArgumentVar(result, true, &ctx));
      }
    }
    for (auto& argument : arguments) {
      if (isScalar(argument.getType())) {
        taco_iassert(!util::contains(scalars, argument));
        taco_iassert(util::contains(ctx.vars, argument));
        scalars.insert({argument, ctx.vars.at(argument)});
        headerStmts.push_back(declareScalarArgumentVar(argument, false, &ctx));
      }
    }
  }

  // Declare, allocate, and initialize temporaries.
  if (ctx.compute) {
    for (auto& temporary : temporaries) {
      if (isScalar(temporary.getType())) {
        taco_iassert(!util::contains(scalars, temporary)) << temporary;
        taco_iassert(util::contains(ctx.vars, temporary));
        scalars.insert({temporary, ctx.vars.at(temporary)});
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
  ir::Stmt header = Block::make(headerStmts);

  ir::Stmt body = lower(stmt, &ctx);

  if (ctx.compute) {
    // Store scalar stack variables back to results
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(util::contains(scalars, result));
        taco_iassert(util::contains(ctx.vars, result));
        Expr resultIR = scalars.at(result);
        Expr varValueIR = ctx.vars.at(result);
        Expr valuesArrIR = GetProperty::make(resultIR, TensorProperty::Values);
        footerStmts.push_back(Store::make(valuesArrIR, 0, varValueIR));
      }
    }
  }
  ir::Stmt footer = Block::make(footerStmts);

  return Function::make(name, resultsIR, argumentsIR,
                        Block::make({header,
                                     BlankLine::make(),
                                     body,
                                     BlankLine::make(),
                                     footer}));
}


bool isLowerable(IndexStmt stmt, std::string* reason) {
  INIT_REASON(reason);

  // Must be concrete index notation
  if (!isConcreteNotation(stmt)) {
    *reason = "The index statement is not in concrete index notation";
    return false;
  }

  // Check for transpositions
  // TODO
//  if (!error::containsTranspose(this->getFormat(), freeVars, indexExpr)) {
//    *reason = error::expr_transposition;
//  }

  return true;
}

/// Prints the hierarchy of merge cases that result from lowering `stmt`.
void printMergeCaseHierarchy(IndexStmt stmt, std::ostream& os) {

}

}
