#include "test.h"

#include "taco/lower/lower.h"
#include "taco/error/error_messages.h"

using namespace taco;

static const Dimension n, m, o;
static const Type vectype(Float64, {n});
static const Type mattype(Float64, {n,m});
static const Type tentype(Float64, {n,m,o});

static TensorVar alpha("alpha", Float64);
static TensorVar beta("beta",   Float64);
static TensorVar delta("delta", Float64);
static TensorVar  zeta("zeta",  Float64);
static TensorVar   eta("eta",   Float64);

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

TEST(DISABLED_lower, transpose) {
  TensorVar A(mattype, Format({sparse,sparse}, {0,1}));
  TensorVar B(mattype, Format({sparse,sparse}, {0,1}));
  TensorVar C(mattype, Format({sparse,sparse}, {1,0}));
  string reason;
  ASSERT_FALSE(isLowerable(forall(i,
                                  forall(j,
                                         A(i,j) = B(i,j) + C(i,j)
                                         )),
                           &reason));
  ASSERT_EQ(error::expr_transposition, reason);
}

TEST(DISABLED_lower, transpose2) {
  TensorVar A(mattype, Format({sparse,sparse}, {0,1}));
  TensorVar B(mattype, Format({sparse,sparse}, {0,1}));
  TensorVar C(mattype, Format({sparse,sparse}, {0,1}));
  string reason;
  ASSERT_FALSE(isLowerable(forall(i,
                                  forall(j,
                                         A(i,j) = B(i,j) + C(j,i)
                                         )),
                           &reason));
  ASSERT_EQ(error::expr_transposition, reason);
}

TEST(DISABLED_lower, transpose3) {
  TensorVar A(tentype, Format({sparse,sparse,sparse}, {0,1,2}));
  TensorVar B(tentype, Format({sparse,sparse,sparse}, {0,1,2}));
  TensorVar C(tentype, Format({sparse,sparse,sparse}, {0,1,2}));
  string reason;
  ASSERT_FALSE(isLowerable(forall(i,
                                  forall(j,
                                         forall(k,
                                                A(i,j,k) = B(i,j,k) + C(k,i,j)
                                                ))),
                           &reason));
  ASSERT_EQ(error::expr_transposition, reason);
}

// denseIterationTranspose tests a dense iteration that contain a transposition
// of one of the tensors.
TEST(lower, denseIterationTranspose) {
  auto dim = 4;
  Tensor<int> A("A", {dim, dim, dim}, {Dense, Dense, Dense});
  Tensor<int> B("B", {dim, dim, dim}, {Dense, Dense, Dense});
  Tensor<int> C("C", {dim, dim, dim}, {Dense, Dense, Dense});
  Tensor<int> expected("expected", {dim, dim, dim}, {Dense, Dense, Dense});
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        A.insert({i, j, k}, i + j + k);
        B.insert({i, j, k}, i + j + k);
        expected.insert({i, j, k}, 2 * (i + j + k));
      }
    }
  }
  A.pack(); B.pack(); expected.pack();
  C(i, j, k) = A(i, j, k) + B(k, j, i);
  C.evaluate();
  ASSERT_TRUE(equals(C, expected));
}
