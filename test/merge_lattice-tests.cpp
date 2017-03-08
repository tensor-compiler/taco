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

TEST(mergelattice, iterator) {
  Tensor<double> a({5}, SVEC);
  Tensor<double> b({5}, SVEC);
  Tensor<double> c({5}, SVEC);
  Var i;
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a.getTensorBase(), i);

  auto it = lattice.begin();
  ASSERT_TRUE(isa<Add>(it++->getExpr()));
  ASSERT_TRUE(isa<Read>(it++->getExpr()));
  ASSERT_TRUE(isa<Read>(it++->getExpr()));
  ASSERT_TRUE(it == lattice.end());
}

TEST(mergelattice, sparse_elmul) {
  Tensor<double> a({5}, SVEC);
  Tensor<double> b({5}, SVEC);
  Tensor<double> c({5}, SVEC);
  Var i;
  a(i) = b(i) * c(i);
  MergeLattice lattice = buildLattice(a.getTensorBase(), i);

  ASSERT_EQ(1u, lattice.getSize());
  ASSERT_EQ(2u, lattice[0].getIterators().size());

  ASSERT_TRUE(isa<Mul>(lattice.getExpr()));
}

TEST(mergelattice, dense_dense_add) {
  Tensor<double> a({5}, DVEC);
  Tensor<double> b({5}, DVEC);
  Tensor<double> c({5}, DVEC);
  Var i;
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a.getTensorBase(), i);

  ASSERT_EQ(1u, lattice.getSize());
  ASSERT_EQ(2u, lattice[0].getIterators().size());
}

TEST(mergelattice, dense_sparse_add) {
  Tensor<double> a({5}, DVEC);
  Tensor<double> b({5}, DVEC);
  Tensor<double> c({5}, SVEC);
  Var i;
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a.getTensorBase(), i);

  ASSERT_EQ(2u, lattice.getSize());
  ASSERT_EQ(2u, lattice[0].getIterators().size());

  ASSERT_TRUE(isa<Add>(lattice.getExpr()));
  ASSERT_TRUE(isa<Add>(lattice[0].getExpr()));

  auto lp1Expr = lattice[1].getExpr();
  ASSERT_TRUE(isa<Read>(lp1Expr));
  ASSERT_TRUE(to<Read>(lp1Expr).getTensor().getName() == b.getName());
}
