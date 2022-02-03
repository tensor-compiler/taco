#include "taco/ir/workspace_rewriter.h"

#include <map>
#include <vector>

#include "taco/ir/ir.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/util/collections.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"

using namespace std;
using namespace taco::ir;
namespace taco {

struct WorkspaceRewriter : ir::IRRewriter {
  WorkspaceRewriter(std::vector<TensorVar> whereTemps, std::map<TensorVar, 
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

        if (temp.defined() && gpName == op->name && (temporarySizeMap.find(temp) != temporarySizeMap.end())) {
          //taco_iassert(temporarySizeMap.find(temp) != temporarySizeMap.end()) << "Cannot rewrite workspace "
          //                                                                       "Dimension GetProperty due "
          //                                                                       "to tensorVar not in "
          //                                                                       "expression map";
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

ir::Stmt rewriteTemporaryGP(const ir::Stmt& stmt, std::vector<TensorVar> whereTemps,
                            std::map<TensorVar, std::vector<ir::Expr>> temporarySizeMap) {
  return WorkspaceRewriter(whereTemps, temporarySizeMap).rewrite(stmt);
}

}
