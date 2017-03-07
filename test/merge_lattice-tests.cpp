#include "test.h"
#include "tensor.h"
#include "ir/ir.h"
#include "lower/lower_codegen.h"
#include "lower/iterators.h"
#include "lower/iteration_schedule.h"
#include "lower/merge_lattice.h"

using namespace std;
using namespace taco;
using namespace taco::lower;

static MergeLattice buildLattice(const TensorBase& tensor, taco::Var i) {
IterationSchedule schedule = IterationSchedule::make(tensor);
  map<TensorBase,ir::Expr> tensorVars;
  tie(std::ignore, std::ignore, tensorVars) = getTensorVars(tensor);
  Iterators iterators(schedule, tensorVars);
  return MergeLattice::make(tensor.getExpr(), i, schedule,iterators);
}

TEST(mergelattice, sparse_elmul) {
  Tensor<double> a({5}, SVEC);
  Tensor<double> b({5}, SVEC);
  Tensor<double> c({5}, SVEC);
  Var i;
  a(i) = b(i) * c(i);

  MergeLattice lattice = buildLattice(a.getTensorBase(), i);

  ASSERT_TRUE(isa<Mul>(lattice.getExpr()));

  ASSERT_EQ(1u, lattice.getPoints().size());
  ASSERT_EQ(2u, lattice.getPoints()[0].getIterators().size());
}

TEST(mergelattice, dense_dense_add) {
  Tensor<double> a({5}, DVEC);
  Tensor<double> b({5}, DVEC);
  Tensor<double> c({5}, DVEC);
  Var i;
  a(i) = b(i) + c(i);

  MergeLattice lattice = buildLattice(a.getTensorBase(), i);

  ASSERT_EQ(1u, lattice.getPoints().size());
  ASSERT_EQ(2u, lattice.getPoints()[0].getIterators().size());
}

TEST(mergelattice, dense_sparse_add) {
  Tensor<double> a({5}, DVEC);
  Tensor<double> b({5}, DVEC);
  Tensor<double> c({5}, SVEC);
  Var i;
  a(i) = b(i) + c(i);

  MergeLattice lattice = buildLattice(a.getTensorBase(), i);

  ASSERT_EQ(2u, lattice.getPoints().size());
  ASSERT_EQ(2u, lattice.getPoints()[0].getIterators().size());
}
