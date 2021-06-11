#ifndef TACO_WORKSPACE_REWRITER_H
#define TACO_WORKSPACE_REWRITER_H

#include <vector>
#include <map>


namespace taco {
class TensorVar;

namespace ir {
class Stmt;
class Expr;
}

/// Simplifies a statement (e.g. by applying constant copy propagation).
ir::Stmt rewriteTemporaryGP(const ir::Stmt& stmt, std::vector<TensorVar> whereTemps,
                            std::map<TensorVar, std::vector<ir::Expr>> temporarySizeMap);

}
#endif
