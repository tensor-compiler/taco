#include "lower_util.h"

#include "ir.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace lower {

ir::Stmt mergePathIndexVars(ir::Expr var, vector<ir::Expr> pathVars){
  return ir::VarAssign::make(var, ir::Min::make(pathVars));
}

vector<ir::Stmt> printCoordinate(const vector<ir::Expr>& indexVars) {
  vector<string> indexVarNames;
  indexVarNames.reserve((indexVars.size()));
  for (auto& indexVar : indexVars) {
    indexVarNames.push_back(util::toString(indexVar));
  }

  vector<string> fmtstrings(indexVars.size(), "%d");
  string format = util::join(fmtstrings, ",");
  vector<ir::Expr> printvars = indexVars;
  return {ir::Print::make("("+util::join(indexVarNames)+") = "
                          "("+format+")\\n", printvars)};
}

}}
