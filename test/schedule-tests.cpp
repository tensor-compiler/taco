#include "test.h"
#include "test_tensors.h"

#include "taco/index_notation/schedule.h"
#include "taco/index_notation/index_notation.h"
#include "taco/util/name_generator.h"
#include "taco/tensor.h"

using namespace taco;

static const Dimension n;
static const Type vectype(Float64, {n});

// Sparse vectors
static TensorVar a("a", vectype, Sparse);
static TensorVar b("b", vectype, Sparse);
static TensorVar c("c", vectype, Sparse);
static TensorVar w("w", vectype, dense);

static const IndexVar i("i"), iw("iw");
static const IndexVar j("j"), jw("jw");
static const IndexVar k("k"), kw("kw");

/*
TEST(schedule, workspace_elmul) {
  Assignment assignment = (a(i) = b(i) * c(i));
  Workspace wopt(assignment.getRhs(), i, iw, w);
  std::cout << assignment << std::endl;

  IndexStmt elmul = makeConcreteNotation(assignment);
  ASSERT_NOTATION_EQ(forall(i, a(i) = b(i) * c(i)), elmul);

  IndexStmt elmul_ws = apply(wopt, elmul);
  ASSERT_NOTATION_EQ(where(forall(i, a(i) = w(i)),
                           forall(iw, w(iw) = b(iw) * c(iw))), elmul_ws);
}
*/

/*
TEST(schedule, workspace_elmul) {
  TensorBase a("a", Float64, {8}, Sparse);
  TensorBase b = d8a("b", Sparse);
  TensorBase c = d8b("c", Sparse);
  b.pack();
  c.pack();

  IndexVar i("i");
  IndexExpr mul = b(i) * c(i);
  a(i) = mul;

  IndexVar iw("iw");
  mul.workspace(i, iw);

  a.evaluate();

  Tensor<double> e("e", {8}, Sparse);
  e.insert({0}, 10.0);
  e.insert({2}, 60.0);
  e.pack();
  ASSERT_TENSOR_EQ(e,a);
}
*/

/*
TEST(schedule, workspace_spmspm) {
  TensorBase A("A", Float(64), {3,3}, Format({Dense,Sparse}));
  TensorBase B = d33a("B", Format({Dense,Sparse}));
  TensorBase C = d33b("C", Format({Dense,Sparse}));
  B.pack();
  C.pack();

  IndexVar i, j, k;
  IndexExpr matmul = B(i,k) * C(k,j);
  A(i,j) = matmul;

  A.evaluate();

  std::cout << A << std::endl;
  Tensor<double> E("e", {3,3}, Format({Dense,Sparse}));
  E.insert({2,0}, 30.0);
  E.insert({2,1}, 180.0);
  E.pack();
  ASSERT_TENSOR_EQ(E,A);
}
*/
