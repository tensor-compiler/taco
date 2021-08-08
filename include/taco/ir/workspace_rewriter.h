#ifndef TACO_WORKSPACE_REWRITER_H
#define TACO_WORKSPACE_REWRITER_H

#include <vector>
#include <map>


namespace taco {
class TensorVar;
struct TemporaryArrays;

namespace ir {
class Stmt;
class Expr;
struct GetProperty;
}

/// Rewrite a post-lowered IR statement to take into account multidimensional temporaries. 
/// Replaces Dimension and Indices GetProperty nodes that correspond to temporary workspaces with
/// their corresponding dimension found in the temporarySizeMap. 
ir::Stmt rewriteTemporaryGP(const ir::Stmt& stmt, std::vector<TensorVar> whereTemps,
                            std::map<TensorVar, std::vector<ir::Expr>> temporarySizeMap,
                            std::map<TensorVar, TemporaryArrays> temporaryArrays);

/// Rewrite a post-lowered IR statement to take into account multidimensional temporaries.
/// Replaces Dimension  GetProperty nodes that correspond to temporary workspaces with
/// their corresponding dimension found in the temporarySizeMap.
ir::Stmt rewriteTemporaryGP(const ir::Stmt& stmt, std::vector<TensorVar> whereTemps,
                            std::map<TensorVar, std::vector<ir::Expr>> temporarySizeMap);

/// Add in the flags for GetProperty nodes that shouldn't be emitted
ir::Stmt addGPLoadFlag(const ir::Stmt& stmt, TensorVar tv, std::map<TensorVar, ir::Expr> tvs);
ir::Stmt addGPLoadFlagAll(const ir::Stmt& stmt, TensorVar tv, std::map<TensorVar, ir::Expr> tvs);

/// Add in the flags for memories that need bp tags
ir::Stmt addUseBPFlag(const ir::Stmt& stmt, std::map<ir::Expr, TensorVar> tensors, std::map<std::string, ir::Expr> funcEnvMap);

ir::Stmt rewriteGPDim(ir::Stmt stmt, const ir::GetProperty* gp, ir::Expr dim);

ir::Stmt replaceGPs(const ir::Stmt& stmt, std::map<ir::Expr, ir::Expr> gpToVarMap);

}
#endif
