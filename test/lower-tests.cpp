#include "test.h"
#include "test_tensors.h"

#include "taco/lower/lower.h"
#include "taco/ir/ir.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"

using namespace taco;
using namespace taco::lower;

static const Dimension n, m, o;
static const Type vectype(Float64, {n});
static const Type mattype(Float64, {n,m});
static const Type tentype(Float64, {n,m,o});

static TensorVar alpha("alpha", Float64);
static TensorVar beta("beta",   Float64);
static TensorVar delta("delta", Float64);
static TensorVar zeta("zeta",   Float64);

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

struct TestCase {
  TestCase(const map<TensorVar,vector<pair<vector<int>,double>>>& input,
           const vector<pair<vector<int>,double>>& expected,
           const map<TensorVar, Shape>& dimensions = {})
      : input(input), expected(expected), dimensions(dimensions) {}
  map<TensorVar,vector<pair<vector<int>,double>>> input;
  vector<pair<vector<int>,double>> expected;
  map<TensorVar, Shape> dimensions;  // Shapes default to 5x5x...
};

struct Test {
  Test() {}
  IndexStmt stmt;
  vector<TestCase> testCases;
  Test(IndexStmt stmt, const vector<TestCase>& testCases) : stmt(stmt),
      testCases(testCases) {}
};

ostream& operator<<(ostream& os, const Test& stmt) {
  os << endl;
  return os << "  " << stmt.stmt;
}

struct Formats {
  Formats() {}
  Formats(map<TensorVar, Format> formats) : formats(formats) {}
  map<TensorVar, Format> formats;
};

ostream& operator<<(ostream& os, const Formats& formats) {
  for (auto& format : formats.formats) {
    os << endl << "  " << format.first.getName() << " : " << format.second;
  }
  return os << endl;
}

struct stmt : public TestWithParam<::testing::tuple<Test,Formats>> {};

static IndexStmt getFormattedStmt(const Test& s, const Formats& f) {
  struct Formater : IndexNotationRewriter {
    using IndexNotationRewriter::visit;
    Formater(const map<TensorVar, Format>& formats) : formats(formats) {}
    const map<TensorVar, Format>& formats;
    map<TensorVar, TensorVar> varMapping;

    TensorVar formatTensorVar(TensorVar var) {
      if (util::contains(formats, var)) {
        if (!util::contains(varMapping, var)) {
          varMapping.insert({var, TensorVar(var.getName(), var.getType(),
                                            formats.at(var))});
        }
        var = varMapping.at(var);
      }
      return var;
    }

    void visit(const AccessNode* node) {
      TensorVar var = formatTensorVar(node->tensorVar);
      if (var != node->tensorVar) {
        expr = Access(var, node->indexVars);
      }
      else {
        expr = node;
      }
    }

    void visit(const AssignmentNode* node) {
      TensorVar var = formatTensorVar(node->lhs.getTensorVar());
      IndexExpr rhs = rewrite(node->rhs);
      if (var == node->lhs.getTensorVar() && rhs == node->rhs) {
        stmt = node;
      }
      else {
        stmt = Assignment(Access(var, node->lhs.getIndexVars()), rhs, node->op);
      }
    }
  };

  return Formater(f.formats).rewrite(s.stmt);
}

TEST_P(stmt, lower) {
  IndexStmt stmt = getFormattedStmt(get<0>(GetParam()), get<1>(GetParam()));
  ASSERT_TRUE(isLowerable(stmt));

  ir::Stmt  func = lower::lower(stmt, "compute", false, true);
  ASSERT_TRUE(func.defined())
      << "The call to lower returned an undefined IR function.";
}

#define TEST_STMT(name, statement, formats, testcases) \
INSTANTIATE_TEST_CASE_P(name, stmt,                    \
Combine(Values(Test(statement, testcases)), formats));

TEST_STMT(DISABLED_scalar_neg,
  alpha = -beta,
  Values(Formats()),
  {
    TestCase({{b, {{{}, 42.0}}}}, {{{}, -42.0}})
  }
)

TEST_STMT(DISABLED_vector_neg,
  forall(i,
         a(i) = -b(i)
         ),
  Values(
         Formats({{a,dense},  {b,dense}}),
         Formats({{a,dense},  {b,sparse}}),
         Formats({{a,sparse}, {b,dense}}),
         Formats({{a,sparse}, {b,sparse}})
         ),
  {
    TestCase({{{b, {{{0},  42.0}, {{3},  4.0}}}},
                   {{{0}, -42.0}, {{3}, -4.0}}})
  }
)

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
