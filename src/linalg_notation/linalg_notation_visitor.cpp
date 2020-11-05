
#include "taco/linalg_notation/linalg_notation_visitor.h"
#include "taco/linalg_notation/linalg_notation_nodes.h"

using namespace std;

namespace taco {

void LinalgExprVisitorStrict::visit(const LinalgExpr &expr) {
  expr.accept(this);
}

void LinalgStmtVisitorStrict::visit(const LinalgStmt& stmt) {
  stmt.accept(this);
}

}