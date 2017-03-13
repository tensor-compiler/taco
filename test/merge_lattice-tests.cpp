#include "test.h"
#include "taco/tensor.h"
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
  Tensor<double> a("a", {5}, SVEC);
  Tensor<double> b("b", {5}, SVEC);
  Tensor<double> c("c", {5}, SVEC);
  Var i("i");
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a, i);

  auto it = lattice.begin();
  ASSERT_TRUE(isa<Add>(it++->getExpr()));
  ASSERT_TRUE(isa<Read>(it++->getExpr()));
  ASSERT_TRUE(isa<Read>(it++->getExpr()));
  ASSERT_TRUE(it == lattice.end());
}

TEST(mergelattice, dense_dense_elmul) {
  Tensor<double> a("a", {5}, DVEC);
  Tensor<double> b("b", {5}, DVEC);
  Tensor<double> c("c", {5}, DVEC);
  Var i("i");
  a(i) = b(i) * c(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(1u, lattice.getSize());
  ASSERT_EQ(2u, lattice[0].getIterators().size());
  ASSERT_EQ(1u, simplify(lattice[0].getIterators()).size());

  ASSERT_TRUE(isa<Mul>(lattice.getExpr()));
}

TEST(mergelattice, sparse_sparse_elmul) {
  Tensor<double> a("a", {5}, SVEC);
  Tensor<double> b("b", {5}, SVEC);
  Tensor<double> c("c", {5}, SVEC);
  Var i("i");
  a(i) = b(i) * c(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(1u, lattice.getSize());
  ASSERT_EQ(2u, lattice[0].getIterators().size());

  ASSERT_TRUE(isa<Mul>(lattice.getExpr()));
}

TEST(mergelattice, dense_dense_add) {
  Tensor<double> a("a", {5}, DVEC);
  Tensor<double> b("b", {5}, DVEC);
  Tensor<double> c("c", {5}, DVEC);
  Var i("i");
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(1u, lattice.getSize());
  ASSERT_EQ(2u, lattice[0].getIterators().size());
  ASSERT_EQ(1u, simplify(lattice[0].getIterators()).size());
}

TEST(mergelattice, dense_sparse_add) {
  Tensor<double> a("a", {5}, DVEC);
  Tensor<double> b("b", {5}, DVEC);
  Tensor<double> c("c", {5}, SVEC);
  Var i("i");
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(2u, lattice.getSize());
  ASSERT_TRUE(isa<Add>(lattice.getExpr()));

  ASSERT_EQ(2u, lattice[0].getIterators().size());
  auto simplifiedIterators = simplify(lattice[0].getIterators());
  ASSERT_EQ(1u, simplifiedIterators.size());
  ASSERT_FALSE(simplifiedIterators[0].isDense());
  ASSERT_TRUE(isa<Add>(lattice[0].getExpr()));

  ASSERT_EQ(1u, lattice[1].getIterators().size());
  auto lp1Expr = lattice[1].getExpr();
  ASSERT_TRUE(isa<Read>(lp1Expr));
  ASSERT_TRUE(to<Read>(lp1Expr).getTensor().getName() == b.getName());
}

TEST(mergelattice, sparse_sparse_add) {
  Tensor<double> a("a", {5}, DVEC);
  Tensor<double> b("b", {5}, SVEC);
  Tensor<double> c("c", {5}, SVEC);
  Var i("i");
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(3u, lattice.getSize());
  ASSERT_TRUE(isa<Add>(lattice.getExpr()));

  ASSERT_EQ(2u, lattice[0].getIterators().size());
  ASSERT_TRUE(isa<Add>(lattice[0].getExpr()));

  ASSERT_EQ(1u, lattice[1].getIterators().size());
  auto lp1Expr = lattice[1].getExpr();
  ASSERT_TRUE(isa<Read>(lp1Expr));
  ASSERT_TRUE(to<Read>(lp1Expr).getTensor().getName() == b.getName());

  ASSERT_EQ(1u, lattice[2].getIterators().size());
  auto lp2Expr = lattice[2].getExpr();
  ASSERT_TRUE(isa<Read>(lp2Expr));
  ASSERT_TRUE(to<Read>(lp2Expr).getTensor().getName() == c.getName());
}

TEST(mergelattice, dense_dense_dense_add) {
  Tensor<double> a("a", {5}, DVEC);
  Tensor<double> b("b", {5}, DVEC);
  Tensor<double> c("c", {5}, DVEC);
  Tensor<double> d("d", {5}, DVEC);
  Var i("i");
  a(i) = b(i) + c(i) + d(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(1u, lattice.getSize());

  ASSERT_EQ(3u, lattice[0].getIterators().size());
  auto iters = simplify(lattice[0].getIterators());
  ASSERT_EQ(1u, iters.size());
  ASSERT_TRUE(iters[0].isDense());
}

TEST(mergelattice, dense_dense_sparse_add) {
  Tensor<double> a("a", {5}, DVEC);
  Tensor<double> b("b", {5}, DVEC);
  Tensor<double> c("c", {5}, DVEC);
  Tensor<double> d("d", {5}, SVEC);
  Var i("i");
  a(i) = b(i) + c(i) + d(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(2u, lattice.getSize());

  auto lp0 = lattice[0];
  ASSERT_EQ(3u, lp0.getIterators().size());
  auto lp0iters = simplify(lp0.getIterators());
  ASSERT_EQ(1u, lp0iters.size());
  ASSERT_FALSE(lp0iters[0].isDense());
  ASSERT_TRUE(isa<Add>(lp0.getExpr()));
  auto lp0add = to<Add>(lp0.getExpr());
  ASSERT_TRUE(isa<Add>(lp0add.getLhs()));
  ASSERT_TRUE(isa<Read>(lp0add.getRhs()));

  auto lp1 = lattice[1];
  ASSERT_EQ(2u, lp1.getIterators().size());
  auto lp1iters = simplify(lp1.getIterators());
  ASSERT_EQ(1u, lp1iters.size());
  ASSERT_TRUE(lp1iters[0].isDense());
  ASSERT_TRUE(isa<Add>(lp1.getExpr()));
  auto lp1add = to<Add>(lp1.getExpr());
  ASSERT_TRUE(isa<Read>(lp1add.getLhs()));
  ASSERT_TRUE(isa<Read>(lp1add.getRhs()));
  ASSERT_EQ(b.getName(), to<Read>(lp1add.getLhs()).getTensor().getName());
  ASSERT_EQ(c.getName(), to<Read>(lp1add.getRhs()).getTensor().getName());
}

TEST(mergelattice, dense_sparse_sparse_add) {
  Tensor<double> a("a", {5}, DVEC);
  Tensor<double> b("b", {5}, DVEC);
  Tensor<double> c("c", {5}, SVEC);
  Tensor<double> d("d", {5}, SVEC);
  Var i("i");
  a(i) = b(i) + c(i) + d(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(4u, lattice.getSize());

  auto lp0 = lattice[0];
  ASSERT_EQ(3u, lp0.getIterators().size());
  auto lp0iters = simplify(lp0.getIterators());
  ASSERT_EQ(2u, lp0iters.size());
  ASSERT_FALSE(lp0iters[0].isDense());
  ASSERT_FALSE(lp0iters[1].isDense());
  ASSERT_TRUE(isa<Add>(lp0.getExpr()));
  auto lp0add = to<Add>(lp0.getExpr());
  ASSERT_TRUE(isa<Add>(lp0add.getLhs()));
  ASSERT_TRUE(isa<Read>(lp0add.getRhs()));

  auto lp1 = lattice[1];
  ASSERT_EQ(2u, lp1.getIterators().size());
  auto lp1iters = simplify(lp1.getIterators());
  ASSERT_EQ(1u, lp1iters.size());
  ASSERT_FALSE(lp1iters[0].isDense());
  ASSERT_TRUE(isa<Add>(lp1.getExpr()));
  auto lp1add = to<Add>(lp1.getExpr());
  ASSERT_TRUE(isa<Read>(lp1add.getLhs()));
  ASSERT_TRUE(isa<Read>(lp1add.getRhs()));
  ASSERT_EQ(b.getName(), to<Read>(lp1add.getLhs()).getTensor().getName());
  ASSERT_EQ(d.getName(), to<Read>(lp1add.getRhs()).getTensor().getName());

  auto lp2 = lattice[2];
  ASSERT_EQ(2u, lp2.getIterators().size());
  auto lp2iters = simplify(lp2.getIterators());
  ASSERT_EQ(1u, lp2iters.size());
  ASSERT_FALSE(lp2iters[0].isDense());
  ASSERT_TRUE(isa<Add>(lp2.getExpr()));
  auto lp2add = to<Add>(lp2.getExpr());
  ASSERT_TRUE(isa<Read>(lp2add.getLhs()));
  ASSERT_TRUE(isa<Read>(lp2add.getRhs()));
  ASSERT_EQ(b.getName(), to<Read>(lp2add.getLhs()).getTensor().getName());
  ASSERT_EQ(c.getName(), to<Read>(lp2add.getRhs()).getTensor().getName());

  auto lp3 = lattice[3];
  ASSERT_EQ(1u, lp3.getIterators().size());
  auto lp3iters = simplify(lp3.getIterators());
  ASSERT_EQ(1u, lp3iters.size());
  ASSERT_TRUE(lp3iters[0].isDense());
  ASSERT_TRUE(isa<Read>(lp3.getExpr()));
  ASSERT_EQ(b.getName(), to<Read>(lp3.getExpr()).getTensor().getName());
}

TEST(mergelattice, sparse_sparse_sparse_add) {
  Tensor<double> a("a", {5}, DVEC);
  Tensor<double> b("b", {5}, SVEC);
  Tensor<double> c("c", {5}, SVEC);
  Tensor<double> d("d", {5}, SVEC);
  Var i("i");
  a(i) = b(i) + c(i) + d(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(7u, lattice.getSize());

  auto lp0 = lattice[0];
  ASSERT_EQ(3u, lp0.getIterators().size());
  auto lp0iters = simplify(lp0.getIterators());
  ASSERT_EQ(3u, lp0iters.size());
  ASSERT_FALSE(lp0iters[0].isDense());
  ASSERT_FALSE(lp0iters[1].isDense());
  ASSERT_FALSE(lp0iters[2].isDense());
  ASSERT_TRUE(isa<Add>(lp0.getExpr()));
  auto lp0add = to<Add>(lp0.getExpr());
  ASSERT_TRUE(isa<Add>(lp0add.getLhs()));
  ASSERT_TRUE(isa<Read>(lp0add.getRhs()));

  auto lp1 = lattice[1];
  ASSERT_EQ(2u, lp1.getIterators().size());
  auto lp1iters = simplify(lp1.getIterators());
  ASSERT_EQ(2u, lp1iters.size());
  ASSERT_FALSE(lp1iters[0].isDense());
  ASSERT_FALSE(lp1iters[1].isDense());
  ASSERT_TRUE(isa<Add>(lp1.getExpr()));
  auto lp1add = to<Add>(lp1.getExpr());
  ASSERT_TRUE(isa<Read>(lp1add.getLhs()));
  ASSERT_TRUE(isa<Read>(lp1add.getRhs()));
  ASSERT_EQ(b.getName(), to<Read>(lp1add.getLhs()).getTensor().getName());
  ASSERT_EQ(d.getName(), to<Read>(lp1add.getRhs()).getTensor().getName());

  auto lp2 = lattice[2];
  ASSERT_EQ(2u, lp2.getIterators().size());
  auto lp2iters = simplify(lp2.getIterators());
  ASSERT_EQ(2u, lp2iters.size());
  ASSERT_FALSE(lp2iters[0].isDense());
  ASSERT_FALSE(lp2iters[1].isDense());
  ASSERT_TRUE(isa<Add>(lp2.getExpr()));
  auto lp2add = to<Add>(lp2.getExpr());
  ASSERT_TRUE(isa<Read>(lp2add.getLhs()));
  ASSERT_TRUE(isa<Read>(lp2add.getRhs()));
  ASSERT_EQ(c.getName(), to<Read>(lp2add.getLhs()).getTensor().getName());
  ASSERT_EQ(d.getName(), to<Read>(lp2add.getRhs()).getTensor().getName());

  auto lp3 = lattice[3];
  ASSERT_EQ(2u, lp3.getIterators().size());
  auto lp3iters = simplify(lp3.getIterators());
  ASSERT_EQ(2u, lp3iters.size());
  ASSERT_FALSE(lp3iters[0].isDense());
  ASSERT_FALSE(lp3iters[1].isDense());
  ASSERT_TRUE(isa<Add>(lp3.getExpr()));
  auto lp3add = to<Add>(lp3.getExpr());
  ASSERT_TRUE(isa<Read>(lp3add.getLhs()));
  ASSERT_TRUE(isa<Read>(lp3add.getRhs()));
  ASSERT_EQ(b.getName(), to<Read>(lp3add.getLhs()).getTensor().getName());
  ASSERT_EQ(c.getName(), to<Read>(lp3add.getRhs()).getTensor().getName());

  auto lp4 = lattice[4];
  ASSERT_EQ(1u, lp4.getIterators().size());
  auto lp4iters = simplify(lp4.getIterators());
  ASSERT_EQ(1u, lp4iters.size());
  ASSERT_FALSE(lp4iters[0].isDense());
  ASSERT_TRUE(isa<Read>(lp4.getExpr()));
  ASSERT_EQ(b.getName(), to<Read>(lp4.getExpr()).getTensor().getName());

  auto lp5 = lattice[5];
  ASSERT_EQ(1u, lp5.getIterators().size());
  auto lp5iters = simplify(lp5.getIterators());
  ASSERT_EQ(1u, lp5iters.size());
  ASSERT_FALSE(lp5iters[0].isDense());
  ASSERT_TRUE(isa<Read>(lp5.getExpr()));
  ASSERT_EQ(c.getName(), to<Read>(lp5.getExpr()).getTensor().getName());

  auto lp6 = lattice[6];
  ASSERT_EQ(1u, lp6.getIterators().size());
  auto lp6iters = simplify(lp6.getIterators());
  ASSERT_EQ(1u, lp6iters.size());
  ASSERT_FALSE(lp6iters[0].isDense());
  ASSERT_TRUE(isa<Read>(lp6.getExpr()));
  ASSERT_EQ(d.getName(), to<Read>(lp6.getExpr()).getTensor().getName());
}

TEST(DISABLED_mergelattice, distribute_vector) {
  Tensor<double> A("A", {5,5}, DMAT);
  Tensor<double> b("b", {5}, SVEC);
  Var i("i"), j("j");
  A(i,j) = b(i);

  MergeLattice ilattice = buildLattice(A, i);
  ASSERT_EQ(1u, ilattice.getSize());
  ASSERT_TRUE(isa<Read>(ilattice.getExpr()));
  ASSERT_TRUE(to<Read>(ilattice.getExpr()).getTensor().getName()==b.getName());
  ASSERT_EQ(1u, ilattice[0].getIterators().size());
  ASSERT_TRUE(!ilattice[0].getIterators()[0].isDense());

  MergeLattice jlattice = buildLattice(A , j);
  std::cout << jlattice << std::endl;
}
