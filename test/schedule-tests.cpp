#include "test.h"
#include "test_tensors.h"

#include "taco/index_notation/schedule.h"
#include "taco/index_notation/index_notation.h"
#include "taco/util/name_generator.h"
#include "taco/tensor.h"

using namespace taco;

static const Dimension n, m;
static const Type vectype(Float64, {n});
static const Type mattype(Float64, {n,m});

// Sparse vectors
static TensorVar a("a", vectype, Sparse);
static TensorVar b("b", vectype, Sparse);
static TensorVar c("c", vectype, Sparse);
static TensorVar w("w", vectype, dense);

static TensorVar A("A", mattype, Sparse);
static TensorVar B("B", mattype, Sparse);
static TensorVar C("C", mattype, Sparse);

static const IndexVar i("i"), iw("iw");
static const IndexVar j("j"), jw("jw");
static const IndexVar k("k"), kw("kw");

TEST(schedule, reorder_preconditions) {
  // Must be concrete index notation
  ASSERT_FALSE(Reorder(i,j).isValid(a(i) = B(i,j)*c(j)));
  ASSERT_FALSE(Reorder(i,j).isValid(a(i) = sum(j,B(i,j)*c(j))));

  ASSERT_FALSE(Reorder(i,j).isValid(forall(i,
                                           multi(forall(j,
                                                        A(i,j) = B(i,j)
                                                        ),
                                                 c(i) = b(i)
                                                 )

                                           )
                                    )
               );
  ASSERT_FALSE(Reorder(i,j).isValid(forall(i,
                                           sequence(forall(j,
                                                           A(i,j) = B(i,j)
                                                           ),
                                                    forall(k,
                                                           A(i,k) += C(i,k)
                                                           )
                                                    )

                                           )
                                    )
               );
}

TEST(schedule, reorder_foralls_assignment) {
  string reason;
  auto forallij = forall(i,
                        forall(j,
                               A(i,j) = B(i,j)
                               )
                        );
  Reorder reorderij(i,j);
  Reorder reorderji(j,i);
  ASSERT_TRUE(reorderij.isValid(forallij,&reason))
      << reorderij << " in " << forallij << endl << reason;
  ASSERT_TRUE(reorderji.isValid(forallij,&reason))
      << reorderji << " in " << forallij << endl << reason;

  auto forallji = forall(j,
                         forall(i,
                                A(i,j) = B(i,j)
                                )
                         );
  ASSERT_NOTATION_EQ(reorderij.apply(forallij), forallji);
  ASSERT_NOTATION_EQ(reorderji.apply(forallij), forallji);
}

/*
TEST(schedule, reorder_foralls_add) {
  auto foralls = forall(i,
                        forall(j,
                               A(i,j) += B(i,j)
                               )
                        );
  Reorder reorder(i,j);
  ASSERT_TRUE(reorder.isValid(foralls)) << reorder << " in " << foralls;
}
*/

/*
TEST(schedule, workspace_elmul) {
  Assignment assignment = (a(i) = b(i) * c(i));
  std::cout << assignment << std::endl;

  IndexStmt elmul = makeConcreteNotation(assignment);
  ASSERT_NOTATION_EQ(forall(i, a(i) = b(i) * c(i)), elmul);

  IndexStmt elmul_ws = Workspace(assignment.getRhs(),i,iw,w).apply(elmul);
  ASSERT_NOTATION_EQ(where(forall(i, a(i) = w(i)),
                           forall(iw, w(iw) = b(iw) * c(iw))),
                     elmul_ws);
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
