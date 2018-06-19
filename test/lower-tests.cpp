#include "test.h"
#include "test_tensors.h"

#include "taco/lower/lower.h"
#include "taco/ir/ir.h"

using namespace taco;
using namespace taco::lower;

static const Dimension n, m, o;
static const Type vectype(Float64, {n});
static const Type mattype(Float64, {n,m});
static const Type tentype(Float64, {n,m,o});

// Sparse vectors
static TensorVar a("a", vectype, Format());
static TensorVar b("b", vectype, Format());
static TensorVar c("c", vectype, Format());
static TensorVar d("d", vectype, Format());

static TensorVar w("w", vectype, dense);

static TensorVar A("A", mattype, Format());
static TensorVar B("B", mattype, Format());
static TensorVar C("C", mattype, Format());
static TensorVar D("D", mattype, Format());

static TensorVar S("S", tentype, Format());
static TensorVar T("T", tentype, Format());
static TensorVar U("U", tentype, Format());
static TensorVar V("V", tentype, Format());

const IndexVar i("i"), iw("iw");
const IndexVar j("j"), jw("jw");
const IndexVar k("k"), kw("kw");

struct LowerTest {
  LowerTest(IndexStmt stmt) : stmt(stmt) {}
  IndexStmt stmt;
};
struct stmt : public TestWithParam<LowerTest> {};

TEST_P(stmt, lower) {
  IndexStmt stmt = GetParam().stmt;

  ir::Stmt  func = lower::lower(stmt, "func", true, false);
  ASSERT_TRUE(func.defined())
      << "The call to lower returned an undefined IR function.";
}

//INSTANTIATE_TEST_CASE_P(scalars, stmt,
//  Values(
//         LowerTest(IndexStmt())
//         )
//);

INSTANTIATE_TEST_CASE_P(DISABLED_copies, stmt,
  Values(
         LowerTest(forall(i, a(i) = b(i)))
         )
);

//INSTANTIATE_TEST_CASE_P(elwise, stmt,
//  Values(
//         LowerTest(IndexStmt())
//         )
//);

//INSTANTIATE_TEST_CASE_P(reductions, stmt,
//  Values(
//         LowerTest(IndexStmt())
//         )
//);


TEST(DISABLED_lower, transpose) {
  TensorVar A(mattype, Format({Sparse,Sparse}, {0,1}));
  TensorVar B(mattype, Format({Sparse,Sparse}, {0,1}));
  TensorVar C(mattype, Format({Sparse,Sparse}, {1,0}));
  string reason;
  ASSERT_FALSE(isLowerable(forall(i,
                                  forall(j,
                                         A(i,j) = B(i,j) + C(i,j)
                                         )),
                           &reason));
  ASSERT_EQ(error::expr_transposition, reason);
}

TEST(DISABLED_lower, transpose2) {
  TensorVar A(mattype, Format({Sparse,Sparse}, {0,1}));
  TensorVar B(mattype, Format({Sparse,Sparse}, {0,1}));
  TensorVar C(mattype, Format({Sparse,Sparse}, {0,1}));
  string reason;
  ASSERT_FALSE(isLowerable(forall(i,
                                  forall(j,
                                         A(i,j) = B(i,j) + C(j,i)
                                         )),
                           &reason));
  ASSERT_EQ(error::expr_transposition, reason);
}

TEST(DISABLED_lower, transpose3) {
  TensorVar A(tentype, Format({Sparse,Sparse,Sparse}, {0,1,2}));
  TensorVar B(tentype, Format({Sparse,Sparse,Sparse}, {0,1,2}));
  TensorVar C(tentype, Format({Sparse,Sparse,Sparse}, {0,1,2}));
  string reason;
  ASSERT_FALSE(isLowerable(forall(i,
                                  forall(j,
                                         forall(k,
                                                A(i,j,k) = B(i,j,k) + C(k,i,j)
                                                ))),
                           &reason));
  ASSERT_EQ(error::expr_transposition, reason);
}
