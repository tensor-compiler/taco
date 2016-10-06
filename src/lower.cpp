#include "lower.h"

#include "iteration_schedule.h"
#include "ir.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace internal {

Stmt lower(string name, vector<taco::Var> indexVars, taco::Expr expr,
           const IterationSchedule& schedule) {
  std::cout << name << "(" << util::join(indexVars) << ") = "
            << expr << std::endl;
  Stmt stmt = Block::make();

  // Generate one loop nest per index variable in the iteration schedule
  // TODO

  std::cout << stmt << std::endl;
  return stmt;
}

}}
