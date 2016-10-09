#include "iteration_schedule.h"

#include "var.h"
#include "expr.h"

using namespace std;

namespace taco {
namespace internal {

/// A tensor Read expression such as A(i,j,k) results in a path in an iteration
/// schedule through i,j,k. The exact path (i->j->k, j->k->i, etc.) is for now
/// dictated by the order of the levels in the tensor storage tree. The index
/// variable that indexes into the dimension at the first level is the first
/// index variable in the path, and so forth.
struct TensorPath {

};

struct IterationSchedule::Content {
  /// A two dimensional ordering of index variables. The first (x) dimension
  /// corresponds to nested loops and the second (y) dimension correspond to
  /// sequenced loops.
  vector<vector<Var>> indexVariables;
};

IterationSchedule::IterationSchedule() {
}

IterationSchedule IterationSchedule::make(const taco::Expr& expr) {


  return IterationSchedule(nullptr);
}

IterationSchedule::IterationSchedule(shared_ptr<Content> content)
    : content(content) {
}

}}
