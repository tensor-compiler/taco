#include "test.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/ir/ir.h"
#include "lower/lower_codegen.h"
#include "lower/iterators.h"
#include "lower/iteration_graph.h"
#include "lower/merge_lattice.h"

using namespace std;
using namespace taco;

static MergeLattice buildLattice(const TensorBase& tensor, IndexVar i) {
  old::IterationGraph iterationGraph =
      old::IterationGraph::make(tensor.getAssignment());
  map<TensorVar,ir::Expr> tensorVars;
  tie(ignore,ignore,tensorVars) = old::getTensorVars(tensor.getAssignment());
  old::Iterators iterators(iterationGraph, tensorVars);
  return MergeLattice::make(tensor.getAssignment().getRhs(), i,
                            iterationGraph, iterators);
}

TEST(MergeLattice, iterator) {
  Tensor<double> a("a", {5}, Sparse);
  Tensor<double> b("b", {5}, Sparse);
  Tensor<double> c("c", {5}, Sparse);
  IndexVar i("i");
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a, i);

  auto it = lattice.begin();
  ASSERT_TRUE(isa<AddNode>(it++->getExpr().ptr));
  ASSERT_TRUE(isa<Access>(it++->getExpr()));
  ASSERT_TRUE(isa<Access>(it++->getExpr()));
  ASSERT_TRUE(it == lattice.end());
}

TEST(MergeLattice, dense_dense_elmul) {
  Tensor<double> a("a", {5}, Dense);
  Tensor<double> b("b", {5}, Dense);
  Tensor<double> c("c", {5}, Dense);
  IndexVar i("i");
  a(i) = b(i) * c(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(1u, lattice.getSize());
  ASSERT_EQ(2u, lattice[0].getIterators().size());
  ASSERT_EQ(1u, lattice[0].getMergeIterators().size());
  ASSERT_EQ(1u, lattice[0].getRangeIterators().size());
  ASSERT_TRUE(lattice.isFull());

  ASSERT_TRUE(isa<MulNode>(lattice.getExpr().ptr));
}

TEST(MergeLattice, sparse_sparse_elmul) {
  Tensor<double> a("a", {5}, Sparse);
  Tensor<double> b("b", {5}, Sparse);
  Tensor<double> c("c", {5}, Sparse);
  IndexVar i("i");
  a(i) = b(i) * c(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(1u, lattice.getSize());
  ASSERT_EQ(2u, lattice[0].getIterators().size());
  ASSERT_EQ(2u, lattice[0].getMergeIterators().size());
  ASSERT_EQ(2u, lattice[0].getRangeIterators().size());
  ASSERT_FALSE(lattice.isFull());

  ASSERT_TRUE(isa<MulNode>(lattice.getExpr().ptr));
}

TEST(MergeLattice, dense_dense_add) {
  Tensor<double> a("a", {5}, Dense);
  Tensor<double> b("b", {5}, Dense);
  Tensor<double> c("c", {5}, Dense);
  IndexVar i("i");
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(1u, lattice.getSize());
  ASSERT_EQ(2u, lattice[0].getIterators().size());
  ASSERT_EQ(2u, lattice[0].getMergeIterators().size());
  ASSERT_EQ(1u, lattice[0].getRangeIterators().size());
  ASSERT_TRUE(lattice.isFull());
}

TEST(MergeLattice, dense_sparse_add) {
  Tensor<double> a("a", {5}, Dense);
  Tensor<double> b("b", {5}, Dense);
  Tensor<double> c("c", {5}, Sparse);
  IndexVar i("i");
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(2u, lattice.getSize());
  ASSERT_TRUE(isa<AddNode>(lattice.getExpr().ptr));
  ASSERT_TRUE(lattice.isFull());

  ASSERT_EQ(2u, lattice[0].getIterators().size());
  ASSERT_EQ(2u, lattice[0].getMergeIterators().size());
  ASSERT_EQ(2u, lattice[0].getRangeIterators().size());
  ASSERT_TRUE(isa<AddNode>(lattice[0].getExpr().ptr));

  ASSERT_EQ(1u, lattice[1].getIterators().size());
  auto lp1Expr = lattice[1].getExpr();
  ASSERT_TRUE(isa<Access>(lp1Expr));
  ASSERT_TRUE(to<Access>(lp1Expr).getTensorVar().getName() == b.getName());
}

TEST(MergeLattice, sparse_sparse_add) {
  Tensor<double> a("a", {5}, Dense);
  Tensor<double> b("b", {5}, Sparse);
  Tensor<double> c("c", {5}, Sparse);
  IndexVar i("i");
  a(i) = b(i) + c(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(3u, lattice.getSize());
  ASSERT_TRUE(isa<AddNode>(lattice.getExpr().ptr));
  ASSERT_TRUE(lattice.isFull());

  ASSERT_EQ(2u, lattice[0].getIterators().size());
  ASSERT_TRUE(isa<AddNode>(lattice[0].getExpr().ptr));

  ASSERT_EQ(1u, lattice[1].getIterators().size());
  auto lp1Expr = lattice[1].getExpr();
  ASSERT_TRUE(isa<Access>(lp1Expr));
  ASSERT_TRUE(to<Access>(lp1Expr).getTensorVar().getName() == b.getName());

  ASSERT_EQ(1u, lattice[2].getIterators().size());
  auto lp2Expr = lattice[2].getExpr();
  ASSERT_TRUE(isa<Access>(lp2Expr));
  ASSERT_TRUE(to<Access>(lp2Expr).getTensorVar().getName() == c.getName());
}

TEST(MergeLattice, dense_dense_dense_add) {
  Tensor<double> a("a", {5}, Dense);
  Tensor<double> b("b", {5}, Dense);
  Tensor<double> c("c", {5}, Dense);
  Tensor<double> d("d", {5}, Dense);
  IndexVar i("i");
  a(i) = b(i) + c(i) + d(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(1u, lattice.getSize());
  ASSERT_TRUE(lattice.isFull());

  ASSERT_EQ(3u, lattice[0].getIterators().size());
  auto rangeIterators = lattice[0].getRangeIterators();
  ASSERT_EQ(1u, rangeIterators.size());
  ASSERT_TRUE(rangeIterators[0].isFull());
}

TEST(MergeLattice, dense_dense_sparse_add) {
  Tensor<double> a("a", {5}, Dense);
  Tensor<double> b("b", {5}, Dense);
  Tensor<double> c("c", {5}, Dense);
  Tensor<double> d("d", {5}, Sparse);
  IndexVar i("i");
  a(i) = b(i) + c(i) + d(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(2u, lattice.getSize());
  ASSERT_TRUE(lattice.isFull());

  auto lp0 = lattice[0];
  ASSERT_EQ(3u, lp0.getIterators().size());
  ASSERT_EQ(2u, lp0.getRangeIterators().size());
  ASSERT_TRUE(isa<AddNode>(lp0.getExpr().ptr));
  auto lp0add = to<AddNode>(lp0.getExpr().ptr);
  ASSERT_TRUE(isa<AddNode>(lp0add->a.ptr));
  ASSERT_TRUE(isa<Access>(lp0add->b));

  auto lp1 = lattice[1];
  ASSERT_EQ(2u, lp1.getIterators().size());
  auto lp1RangeIterators = lp1.getRangeIterators();
  ASSERT_EQ(1u, lp1RangeIterators.size());
  ASSERT_TRUE(lp1RangeIterators[0].isFull());
  ASSERT_TRUE(isa<AddNode>(lp1.getExpr().ptr));
  auto lp1add = to<AddNode>(lp1.getExpr().ptr);
  ASSERT_TRUE(isa<Access>(lp1add->a));
  ASSERT_TRUE(isa<Access>(lp1add->b));
  ASSERT_EQ(b.getName(), to<Access>(lp1add->a).getTensorVar().getName());
  ASSERT_EQ(c.getName(), to<Access>(lp1add->b).getTensorVar().getName());
}

TEST(MergeLattice, dense_sparse_sparse_add) {
  Tensor<double> a("a", {5}, Dense);
  Tensor<double> b("b", {5}, Dense);
  Tensor<double> c("c", {5}, Sparse);
  Tensor<double> d("d", {5}, Sparse);
  IndexVar i("i");
  a(i) = b(i) + c(i) + d(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(4u, lattice.getSize());
  ASSERT_TRUE(lattice.isFull());

  auto lp0 = lattice[0];
  ASSERT_EQ(3u, lp0.getIterators().size());
  auto lp0RangeIterators = lp0.getRangeIterators();
  ASSERT_EQ(3u, lp0RangeIterators.size());
  ASSERT_TRUE(isa<AddNode>(lp0.getExpr().ptr));
  auto lp0add = to<AddNode>(lp0.getExpr().ptr);
  ASSERT_TRUE(isa<AddNode>(lp0add->a.ptr));
  ASSERT_TRUE(isa<Access>(lp0add->b));

  auto lp1 = lattice[1];
  ASSERT_EQ(2u, lp1.getIterators().size());
  auto lp1RangeIterators = lp1.getRangeIterators();
  ASSERT_EQ(2u, lp1RangeIterators.size());
  ASSERT_TRUE(isa<AddNode>(lp1.getExpr().ptr));
  auto lp1add = to<AddNode>(lp1.getExpr().ptr);
  ASSERT_TRUE(isa<Access>(lp1add->a));
  ASSERT_TRUE(isa<Access>(lp1add->b));
  ASSERT_EQ(b.getName(), to<Access>(lp1add->a).getTensorVar().getName());
  ASSERT_EQ(d.getName(), to<Access>(lp1add->b).getTensorVar().getName());

  auto lp2 = lattice[2];
  ASSERT_EQ(2u, lp2.getIterators().size());
  auto lp2RangeIterators = lp2.getRangeIterators();
  ASSERT_EQ(2u, lp2RangeIterators.size());
  ASSERT_TRUE(isa<AddNode>(lp2.getExpr().ptr));
  auto lp2add = to<AddNode>(lp2.getExpr().ptr);
  ASSERT_TRUE(isa<Access>(lp2add->a));
  ASSERT_TRUE(isa<Access>(lp2add->b));
  ASSERT_EQ(b.getName(), to<Access>(lp2add->a).getTensorVar().getName());
  ASSERT_EQ(c.getName(), to<Access>(lp2add->b).getTensorVar().getName());

  auto lp3 = lattice[3];
  ASSERT_EQ(1u, lp3.getIterators().size());
  auto lp3RangeIterators = lp3.getRangeIterators();
  ASSERT_EQ(1u, lp3RangeIterators.size());
  ASSERT_TRUE(lp3RangeIterators[0].isFull());
  ASSERT_TRUE(isa<Access>(lp3.getExpr()));
  ASSERT_EQ(b.getName(), to<Access>(lp3.getExpr()).getTensorVar().getName());
}

TEST(MergeLattice, sparse_sparse_sparse_add) {
  Tensor<double> a("a", {5}, Dense);
  Tensor<double> b("b", {5}, Sparse);
  Tensor<double> c("c", {5}, Sparse);
  Tensor<double> d("d", {5}, Sparse);
  IndexVar i("i");
  a(i) = b(i) + c(i) + d(i);
  MergeLattice lattice = buildLattice(a, i);

  ASSERT_EQ(7u, lattice.getSize());
  ASSERT_TRUE(lattice.isFull());

  auto lp0 = lattice[0];
  ASSERT_EQ(3u, lp0.getIterators().size());
  auto lp0RangeIterators = lp0.getRangeIterators();
  ASSERT_EQ(3u, lp0RangeIterators.size());
  ASSERT_FALSE(lp0RangeIterators[0].isFull());
  ASSERT_FALSE(lp0RangeIterators[1].isFull());
  ASSERT_FALSE(lp0RangeIterators[2].isFull());
  ASSERT_TRUE(isa<AddNode>(lp0.getExpr().ptr));
  auto lp0add = to<AddNode>(lp0.getExpr().ptr);
  ASSERT_TRUE(isa<AddNode>(lp0add->a.ptr));
  ASSERT_TRUE(isa<Access>(lp0add->b));

  auto lp1 = lattice[1];
  ASSERT_EQ(2u, lp1.getIterators().size());
  auto lp1RangeIterators = lp1.getRangeIterators();
  ASSERT_EQ(2u, lp1RangeIterators.size());
  ASSERT_FALSE(lp1RangeIterators[0].isFull());
  ASSERT_FALSE(lp1RangeIterators[1].isFull());
  ASSERT_TRUE(isa<AddNode>(lp1.getExpr().ptr));
  auto lp1add = to<AddNode>(lp1.getExpr().ptr);
  ASSERT_TRUE(isa<Access>(lp1add->a));
  ASSERT_TRUE(isa<Access>(lp1add->b));
  ASSERT_EQ(b.getName(), to<Access>(lp1add->a).getTensorVar().getName());
  ASSERT_EQ(d.getName(), to<Access>(lp1add->b).getTensorVar().getName());

  auto lp2 = lattice[2];
  ASSERT_EQ(2u, lp2.getIterators().size());
  auto lp2RangeIterators = lp2.getRangeIterators();
  ASSERT_EQ(2u, lp2RangeIterators.size());
  ASSERT_FALSE(lp2RangeIterators[0].isFull());
  ASSERT_FALSE(lp2RangeIterators[1].isFull());
  ASSERT_TRUE(isa<AddNode>(lp2.getExpr().ptr));
  auto lp2add = to<AddNode>(lp2.getExpr().ptr);
  ASSERT_TRUE(isa<Access>(lp2add->a));
  ASSERT_TRUE(isa<Access>(lp2add->b));
  ASSERT_EQ(c.getName(), to<Access>(lp2add->a).getTensorVar().getName());
  ASSERT_EQ(d.getName(), to<Access>(lp2add->b).getTensorVar().getName());

  auto lp3 = lattice[3];
  ASSERT_EQ(2u, lp3.getIterators().size());
  auto lp3RangeIterators = lp3.getRangeIterators();
  ASSERT_EQ(2u, lp3RangeIterators.size());
  ASSERT_FALSE(lp3RangeIterators[0].isFull());
  ASSERT_FALSE(lp3RangeIterators[1].isFull());
  ASSERT_TRUE(isa<AddNode>(lp3.getExpr().ptr));
  auto lp3add = to<AddNode>(lp3.getExpr().ptr);
  ASSERT_TRUE(isa<Access>(lp3add->a));
  ASSERT_TRUE(isa<Access>(lp3add->b));
  ASSERT_EQ(b.getName(), to<Access>(lp3add->a).getTensorVar().getName());
  ASSERT_EQ(c.getName(), to<Access>(lp3add->b).getTensorVar().getName());

  auto lp4 = lattice[4];
  ASSERT_EQ(1u, lp4.getIterators().size());
  auto lp4Iterators = simplify(lp4.getIterators());
  ASSERT_EQ(1u, lp4Iterators.size());
  ASSERT_FALSE(lp4Iterators[0].isFull());
  ASSERT_TRUE(isa<Access>(lp4.getExpr()));
  ASSERT_EQ(b.getName(), to<Access>(lp4.getExpr()).getTensorVar().getName());

  auto lp5 = lattice[5];
  ASSERT_EQ(1u, lp5.getIterators().size());
  auto lp5RangeIterators = simplify(lp5.getIterators());
  ASSERT_EQ(1u, lp5RangeIterators.size());
  ASSERT_FALSE(lp5RangeIterators[0].isFull());
  ASSERT_TRUE(isa<Access>(lp5.getExpr()));
  ASSERT_EQ(c.getName(), to<Access>(lp5.getExpr()).getTensorVar().getName());

  auto lp6 = lattice[6];
  ASSERT_EQ(1u, lp6.getIterators().size());
  auto lp6RangeIterators = simplify(lp6.getIterators());
  ASSERT_EQ(1u, lp6RangeIterators.size());
  ASSERT_FALSE(lp6RangeIterators[0].isFull());
  ASSERT_TRUE(isa<Access>(lp6.getExpr()));
  ASSERT_EQ(d.getName(), to<Access>(lp6.getExpr()).getTensorVar().getName());
}

/*
TEST(DISABLED_MergeLattice, distribute_vector) {
  Tensor<double> A("A", {5,5}, DMAT);
  Tensor<double> b("b", {5}, Sparse);
  IndexVar i("i"), j("j");
  A(i,j) = b(i);

  MergeLattice ilattice = buildLattice(A, i);
  ASSERT_EQ(1u, ilattice.getSize());
  ASSERT_TRUE(isa<Access>(ilattice.getExpr()));
  ASSERT_EQ(b.getName(), to<AccessNode>(ilattice.getExpr())->tensor.getName());
  ASSERT_EQ(1u, ilattice[0].getIterators().size());
  ASSERT_TRUE(!ilattice[0].getIterators()[0].isFull());

  MergeLattice jlattice = buildLattice(A , j);
  std::cout << jlattice << std::endl;
}
*/
