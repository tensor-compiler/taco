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
          int opMode = std::max((int)0, op->mode);
          expr = tempExprList.at(opMode);
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

struct GetPropertyRewriter : ir::IRRewriter {
  GetPropertyRewriter(std::map<ir::Expr, ir::Expr> gpToVarMap)
    : gpToVarMap(gpToVarMap) {}

  std::map<ir::Expr, ir::Expr> gpToVarMap;

  using IRRewriter::visit;
  void visit(const ir::GetProperty* op) {
    Expr tensor = rewrite(op->tensor);

    if (gpToVarMap.count(op) > 0) {
      expr = GetProperty::make(op->tensor, op->property, op->mode, op->index, op->name, op->dim, true, op->useBP);
    } else {
      expr = op;
    }
  }
};

struct TagTensorPropertyLoads : IRRewriter {
  using IRRewriter::visit;
  TensorVar tv;
  map<TensorVar, Expr> tvs;

  TagTensorPropertyLoads(TensorVar tv, map<TensorVar, Expr> tvs) : tv(tv), tvs(tvs) {}

  void visit(const GetProperty* op) {
    op->tensor.accept(this);

    if (tvs.count(tv) > 0 && op->tensor == tvs.at(tv)) {
      if ((op->property == TensorProperty::Indices && op->index == 1) || op->property == TensorProperty::Values) {
        expr = GetProperty::make(op->tensor, op->property, op->mode, op->index, op->name, true);
        return;
      }
    }
    expr = op;
  }
};

struct TagTensorPropertyLoadsAll : IRRewriter {
  using IRRewriter::visit;
  TensorVar tv;
  map<TensorVar, Expr> tvs;

  TagTensorPropertyLoadsAll(TensorVar tv, map<TensorVar, Expr> tvs) : tv(tv), tvs(tvs) {}

  void visit(const GetProperty* op) {
    op->tensor.accept(this);

    if (tvs.count(tv) > 0 && op->tensor == tvs.at(tv)) {
      expr = GetProperty::make(op->tensor, op->property, op->mode, op->index, op->name, true);
      return;
    }
    expr = op;
  }
};

struct DetectBPTensorProperties : IRRewriter {
  using IRRewriter::visit;
  map<Expr, TensorVar> tensors;
  std::map<std::string, ir::Expr> envValMap;

  DetectBPTensorProperties(map<Expr, TensorVar> tensors, std::map<std::string, ir::Expr> envValMap) : tensors(tensors), envValMap(envValMap) {}

  void visit(const GetProperty* op) {
    op->tensor.accept(this);

    if (tensors.count(op->tensor) > 0 &&  envValMap["bp"].as<ir::Literal>()->getIntValue() > 1) {
      bool useBP = false;
      MemoryLocation memLoc = tensors.at(op->tensor).getMemoryLocation();
      switch (memLoc) {
        case MemoryLocation::SpatialSparseParSRAMSwizzle:
        case MemoryLocation::SpatialSparseParSRAM:
        case MemoryLocation::SpatialSparseSRAM:
          useBP = true;
          break;
        default:
          break;
      }

      if (useBP)
        expr = GetProperty::make(op->tensor, op->property, op->mode, op->index, op->name, op->load_local, true);
      else
        expr = op;
    } else {
      expr = op;
    }
  }
};

struct RewriteGPDim : IRRewriter {
  using IRRewriter::visit;
  const GetProperty* gp;
  Expr dim;

  RewriteGPDim(const GetProperty* gp, ir::Expr dim) :
    gp(gp), dim(dim) {}

  void visit(const GetProperty* op) {
    op->tensor.accept(this);

    if (op->name == gp->name) {
        expr = GetProperty::make(op->tensor, op->property, op->mode, op->index, op->name, dim, op->load_local, op->useBP);
    } else {
      expr = op;
    }
  }
};

ir::Stmt rewriteGPDim(ir::Stmt stmt, const GetProperty* gp, Expr dim) {
  Stmt rewrittenStmt = RewriteGPDim(gp, dim).rewrite(stmt);
  return rewrittenStmt;
}

ir::Stmt addUseBPFlag(const ir::Stmt& stmt, std::map<Expr, TensorVar> tensors, std::map<std::string, ir::Expr> envValMap) {
  ir::Stmt rewrittenStmt = DetectBPTensorProperties(tensors, envValMap).rewrite(stmt);
  return rewrittenStmt;
}

ir::Stmt addGPLoadFlag(const ir::Stmt& stmt, TensorVar tv, map<TensorVar, Expr> tvs) {
  ir::Stmt rewrittenStmt = TagTensorPropertyLoads(tv, tvs).rewrite(stmt);
  return rewrittenStmt;
}

ir::Stmt addGPLoadFlagAll(const ir::Stmt& stmt, TensorVar tv, map<TensorVar, Expr> tvs) {
  ir::Stmt rewrittenStmt = TagTensorPropertyLoadsAll(tv, tvs).rewrite(stmt);
  return rewrittenStmt;
}

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

ir::Stmt replaceGPs(const ir::Stmt& stmt, std::map<ir::Expr, ir::Expr> gpToVarMap) {
  return GetPropertyRewriter(gpToVarMap).rewrite(stmt);
}

}
