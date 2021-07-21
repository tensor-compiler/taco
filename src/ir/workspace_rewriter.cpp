#include "taco/ir/workspace_rewriter.h"

#include <map>
#include <vector>

#include "taco/ir/ir.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/util/collections.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/lower/lowerer_impl_dataflow.h"
#include "taco/spatial.h"
#include <codegen/codegen_spatial.h>
#include <codegen/codegen.h>

using namespace std;
using namespace taco::ir;
namespace taco {

struct WorkspaceDimensionRewriter : ir::IRRewriter {
  WorkspaceDimensionRewriter(std::vector<TensorVar> whereTemps, std::map<TensorVar,
    std::vector<ir::Expr>> temporarySizeMap) : whereTemps(whereTemps),
  temporarySizeMap(temporarySizeMap) {}
  std::vector<TensorVar> whereTemps;
  std::map<TensorVar, std::vector<ir::Expr>> temporarySizeMap;

  using IRRewriter::visit;
  void visit(const ir::GetProperty* op) {
    Expr tensor = rewrite(op->tensor);

    if (op->property == TensorProperty::Dimension && !whereTemps.empty()) {
      for (auto& temp : whereTemps) {
        string gpName = temp.getName() + to_string(op->mode + 1) + "_dimension";

        if (temp.defined() && gpName == op->name) {
          taco_iassert(temporarySizeMap.find(temp) != temporarySizeMap.end()) << "Cannot rewrite workspace "
                                                                                 "Dimension GetProperty due "
                                                                                 "to tensorVar not in "
                                                                                 "expression map";
          auto tempExprList = temporarySizeMap.at(temp);

          taco_iassert((int)tempExprList.size() > op->mode) << "Cannot rewrite workspace (" 
                                                            << op->tensor 
                                                            << ") Dimension GetProperty due to mode (" 
                                                            << op->mode 
                                                            << ") not in expression map (size = "
                                                            << tempExprList.size() << ")";
          expr = tempExprList.at(op->mode);
          return;
        }
      }
    }
    expr = op;
  }

  void visit(const VarDecl* decl) {
    Expr rhs = rewrite(decl->rhs);
    stmt = (rhs == decl->rhs) ? decl : VarDecl::make(decl->var, rhs);
  }
};

struct HasWorkspaceGP : ir::IRVisitor {
  HasWorkspaceGP(std::vector<TensorVar> whereTemps, std::map<TensorVar, TemporaryArrays> temporaryArrays)
    : whereTemps(whereTemps), temporaryArrays(temporaryArrays) {}

  std::vector<TensorVar> whereTemps;
  std::map<TensorVar, TemporaryArrays> temporaryArrays;
  bool hasGP = false;
  using IRVisitor::visit;
  void visit(const ir::GetProperty* op) {
    op->tensor.accept(this);

    if (op->property == TensorProperty::Indices && !whereTemps.empty()) {
      for (auto& temp : whereTemps) {
        string gpNamePos = temp.getName() + to_string(op->mode + 1) + "_pos";
        string gpNameCrd = temp.getName() + to_string(op->mode + 1) + "_crd";

        if (temp.defined() && (gpNameCrd == op->name || gpNamePos == op->name)) {
          hasGP = true;
        }
      }
    }
  }
  bool hasWsGP(Stmt stmt) {
    stmt.accept(this);
    return this->hasGP;
  }
  bool hasWsGP(Expr expr) {
    expr.accept(this);
    return this->hasGP;
  }
};

struct WorkspaceIndRewriter : ir::IRRewriter {
  WorkspaceIndRewriter(std::vector<TensorVar> whereTemps, std::map<TensorVar, TemporaryArrays> temporaryArrays)
    : whereTemps(whereTemps), temporaryArrays(temporaryArrays) {}

  std::vector<TensorVar> whereTemps;
  std::map<TensorVar, TemporaryArrays> temporaryArrays;

  using IRRewriter::visit;
  void visit(const ir::GetProperty* op) {
    Expr tensor = rewrite(op->tensor);

    if (op->property == TensorProperty::Indices && !whereTemps.empty()) {
      for (auto& temp : whereTemps) {
        string gpNamePos = temp.getName() + to_string(op->mode + 1) + "_pos";
        string gpNameCrd = temp.getName() + to_string(op->mode + 1) + "_crd";

        if (temp.defined() && (gpNameCrd == op->name || gpNamePos == op->name)) {
          taco_iassert(temporaryArrays.find(temp) != temporaryArrays.end()) << "Cannot rewrite workspace "
                                                                                 "Indices GetProperty due "
                                                                                 "to tensorVar"
                                                                                 << temp
                                                                                 << "not in "
                                                                                 "expression map";
          auto tempArray = temporaryArrays.at(temp);

          taco_iassert((int)tempArray.indices.count(op->name) > 0) << "Cannot rewrite workspace ("
                                                            << op->tensor
                                                            << ") Indices GetProperty due to array ("
                                                            << op->index
                                                            << ") not in TemporaryArray struct";
          expr = tempArray.indices.at(op->name);
          return;
        }
      }
    }
    expr = op;
  }
};



ir::Stmt rewriteTemporaryGP(const ir::Stmt& stmt, std::vector<TensorVar> whereTemps,
                            std::map<TensorVar, std::vector<ir::Expr>> temporarySizeMap,
                            std::map<TensorVar, TemporaryArrays> temporaryArrays) {
  ir::Stmt rewrittenStmt = WorkspaceDimensionRewriter(whereTemps, temporarySizeMap).rewrite(stmt);


  rewrittenStmt = WorkspaceIndRewriter(whereTemps, temporaryArrays).rewrite(rewrittenStmt);
  auto hasGP = HasWorkspaceGP(whereTemps, temporaryArrays).hasWsGP(rewrittenStmt);
  taco_iassert(!hasGP) << "Rewriter should completely remove workspace GetProperty nodes";

  return rewrittenStmt;
}

ir::Stmt rewriteTemporaryGP(const ir::Stmt& stmt, std::vector<TensorVar> whereTemps,
                            std::map<TensorVar, std::vector<ir::Expr>> temporarySizeMap) {
  ir::Stmt rewrittenStmt = WorkspaceDimensionRewriter(whereTemps, temporarySizeMap).rewrite(stmt);

  return rewrittenStmt;
}

}
