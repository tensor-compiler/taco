#include "test.h"
#include "test_tensors.h"

#include <cmath>

#include "taco/lower/lower.h"
#include "taco/ir/ir.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/kernel.h"
#include "taco/codegen/module.h"
#include "taco/storage/storage.h"
#include "taco/storage/pack.h"
#include "taco/lower/lower.h"
#include "taco/format.h"
#include "taco/util/strings.h"

namespace taco {
namespace test {

static const Dimension n;
static const Type vectype(Float64, {n});
static const Type mattype(Float64, {n,n});
static const Type tentype(Float64, {n,n,n});

static TensorVar alpha("alpha", Float64);
static TensorVar beta("beta",   Float64);
static TensorVar delta("delta", Float64);
static TensorVar  zeta("zeta",  Float64);
static TensorVar   eta("eta",   Float64);

static TensorVar t("t", Float64);
static TensorVar ti("ti", Float64);
static TensorVar tj("tj", Float64);

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
  TestCase(const map<TensorVar, vector<pair<vector<int>,double>>>& inputs,
           const map<TensorVar, vector<pair<vector<int>,double>>>& expected,
           const map<TensorVar, vector<int>>& dimensions = {})
      : inputs(inputs), expected(expected), dimensions(dimensions) {}

  map<TensorVar, vector<pair<vector<int>,double>>> inputs;
  map<TensorVar, vector<pair<vector<int>,double>>> expected;
  map<TensorVar, vector<int>> dimensions;  // Shapes default to 5x5x...

  vector<int> getDimensions(TensorVar var) const {
    vector<int> dims;
    if (contains(dimensions, var)) {
      dims = dimensions.at(var);
    }
    else {
      for (int i=0; i < var.getOrder(); ++i) {
        dims.push_back(5);
      }
    }
    return dims;
  }

  TensorStorage getResult(TensorVar var, Format format) const {
    auto dimensions = getDimensions(var);
    TensorStorage storage(type<double>(), dimensions, format);

    // TODO: Get rid of this and lower to use dimensions instead
    vector<taco::ModeIndex> modeIndices(format.getOrder());
    for (int i = 0; i < format.getOrder(); ++i) {
      if (format.getModeFormats()[i] == dense) {
        const size_t idx = format.getModeOrdering()[i];
        modeIndices[i] = taco::ModeIndex({taco::makeArray({dimensions[idx]})});
      }
    }
    storage.setIndex(taco::Index(format, modeIndices));

    return storage;
  }

  static
  TensorStorage pack(Format format, const vector<int>& dims,
                     const vector<pair<vector<int>,double>>& components){
    size_t order = dims.size();
    size_t num = components.size();
    if (order == 0) {
      TensorStorage storage = TensorStorage(type<double>(), {}, format);
      Array array = makeArray(type<double>(), 1);
      *((double*)array.getData()) = components[0].second;
      storage.setValues(array);
      return storage;
    }
    else {
      vector<TypedIndexVector> coords;
      for (size_t i=0; i < order; ++i) {
        coords.push_back(TypedIndexVector(type<int>(), num));
      }
      vector<double> values(num);
      for (size_t i=0; i < components.size(); ++i) {
        auto& coordinates = components[i].first;
        for (size_t j=0; j < coordinates.size(); ++j) {
          coords[j][i] = coordinates[j];
        }
        values[i] = components[i].second;
      }
      return taco::pack(type<double>(), dims, format, coords, values.data());
    }
  }

  TensorStorage getArgument(TensorVar var, Format format) const {
    taco_iassert(contains(inputs, var)) << var;
    return pack(format, getDimensions(var), inputs.at(var));
  }

  TensorStorage getExpected(TensorVar var, Format format) const {
    taco_iassert(contains(expected, var)) << var;
    return pack(format, getDimensions(var), expected.at(var));
  }
};

std::ostream& operator<<(std::ostream& os, const TestCase& testcase) {
  os << endl << " Inputs:";
  for (auto& input : testcase.inputs) {
    os << endl << "  " << input.first.getName() << ":";
    for (auto& component : input.second) {
      os << " (" << util::join(component.first) << "),"
         << component.second << " ";
    }
  }
  os << endl << " Expected:";
  for (auto& expected : testcase.expected) {
    os << endl << "  " << expected.first.getName() << ":";
    for (auto& component : expected.second) {
      os << " (" << util::join(component.first) << "),"
         << component.second << " ";
    }
  }
  return os;
}

struct Test {
  Test() = default;
  Test(IndexStmt stmt, const vector<TestCase>& testCases)
      : stmt(stmt), testCases(testCases) {}

  IndexStmt stmt;
  vector<TestCase> testCases;
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

struct lower : public TestWithParam<::testing::tuple<Test,Formats>> {};


static
map<TensorVar,TensorVar> formatVars(const std::vector<TensorVar>& vars,
                                    const map<TensorVar,Format>& formats) {
  map<TensorVar,TensorVar> formatted;
  for (auto& var : vars) {
    Format format;
    if (contains(formats, var)) {
      format = formats.at(var);
    }
    else {
      // Default format is dense in all dimensions
      format = Format(vector<ModeFormatPack>(var.getOrder(), dense));
    }
    formatted.insert({var, TensorVar(var.getName(), var.getType(), format)});
  }
  return formatted;
}

static void verifyResults(const vector<TensorVar>& results,
                          const vector<TensorStorage>& arguments,
                          const map<TensorVar,TensorVar>& varsFormatted,
                          const map<TensorVar, TensorStorage>& expected) {
  for (size_t i = 0; i < results.size(); i++) {
    TensorVar result = results[i];
    TensorStorage actualStorage = arguments[i];
    TensorStorage expectedStorage = expected.at(result);
    Format format = varsFormatted.at(result).getFormat();
    Tensor<double> actual(actualStorage.getDimensions(), format);
    Tensor<double> expected(expectedStorage.getDimensions(), format);
    actual.setStorage(actualStorage);
    expected.setStorage(expectedStorage);
    ASSERT_TENSOR_EQ(expected, actual);
  }
}

TEST_P(lower, compile) {
  map<TensorVar,TensorVar> varsFormatted =
      formatVars(getTensorVars(get<0>(GetParam()).stmt),
                 get<1>(GetParam()).formats);

  IndexStmt stmt = replace(get<0>(GetParam()).stmt, varsFormatted);
  ASSERT_TRUE(isLowerable(stmt)) << stmt;
  Kernel kernel = compile(stmt);

  for (auto& testCase : get<0>(GetParam()).testCases) {
    vector<TensorStorage> arguments;

    // Result tensors
    vector<TensorVar> results = getResults(get<0>(GetParam()).stmt);
    for (auto& result : results) {
      Format format = varsFormatted.at(result).getFormat();
      TensorStorage resultStorage = testCase.getResult(result, format);
      arguments.push_back(resultStorage);
    }

    // Input tensors
    for (auto& argument : getArguments(get<0>(GetParam()).stmt)) {
      Format format = varsFormatted.at(argument).getFormat();
      TensorStorage operandStorage = testCase.getArgument(argument, format);
      arguments.push_back(operandStorage);
    }

    // Expected tensors values
    map<TensorVar, TensorStorage> expected;
    for (auto& result : results) {
      Format format = varsFormatted.at(result).getFormat();
      expected.insert({result, testCase.getExpected(result, format)});
    }

    {
      SCOPED_TRACE("Separate Assembly and Compute\n");
      ASSERT_TRUE(kernel.assemble(arguments));
      ASSERT_TRUE(kernel.compute(arguments));
      verifyResults(results, arguments, varsFormatted, expected);
    }

    {
      SCOPED_TRACE("Fused Assembly and Compute\n");
      ASSERT_TRUE(kernel(arguments));
      verifyResults(results, arguments, varsFormatted, expected);
    }
  }
}

#define TEST_STMT(name, statement, formats, testcases) \
INSTANTIATE_TEST_CASE_P(name, lower,                   \
Combine(Values(Test(statement, testcases)), formats));


// Test scalar operations

TEST_STMT(scalar_copy,
  alpha = beta(),
  Values(Formats()),
  {
    TestCase({{beta,  {{{},  42.0}}}},
             {{alpha, {{{},  42.0}}}})
  }
)

TEST_STMT(scalar_neg,
  alpha = -beta,
  Values(Formats()),
  {
    TestCase({{beta,  {{{},  42.0}}}},
             {{alpha, {{{}, -42.0}}}})
  }
)

TEST_STMT(scalar_add,
  alpha = beta + delta,
  Values(Formats()),
  {
    TestCase({{beta,  {{{},  2.0}}},
              {delta, {{{},  30.0}}}},
             {{alpha, {{{},  32.0}}}})
  }
)

TEST_STMT(scalar_sub,
  alpha = beta - delta,
  Values(Formats()),
  {
    TestCase({{beta,  {{{},  2.0}}},
              {delta, {{{},  30.0}}}},
             {{alpha, {{{}, -28.0}}}})
  }
)

TEST_STMT(scalar_mul,
  alpha = beta * delta,
  Values(Formats()),
  {
    TestCase({{beta,  {{{},  2.0}}},
              {delta, {{{},  30.0}}}},
             {{alpha, {{{},  60.0}}}})
  }
)

TEST_STMT(scalar_div,
  alpha = beta / delta,
  Values(Formats()),
  {
    TestCase({{beta,  {{{},  2.0}}},
              {delta, {{{},  30.0}}}},
             {{alpha, {{{},  0.06666666667}}}})
  }
)

TEST_STMT(scalar_sqr,
  alpha = sqrt(beta),
  Values(Formats()),
  {
    TestCase({{beta,  {{{},  4.0}}}},
             {{alpha, {{{},  2.0}}}})
  }
)


// Test vector operations

TEST_STMT(vector_neg,
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
    TestCase({{b, {{{0},  42.0}, {{3},  4.0}}}},
             {{a, {{{0}, -42.0}, {{3}, -4.0}}}})
  }
)

TEST_STMT(vector_mul,
  forall(i,
         a(i) = b(i) * c(i)
         ),
  Values(
         Formats({{a, dense}, {b, dense}, {c, dense}}),
         Formats({{a, dense}, {b,sparse}, {c, dense}}),
         Formats({{a, dense}, {b, dense}, {c, sparse}}),
         Formats({{a, dense}, {b,sparse}, {c, sparse}}),
         Formats({{a,sparse}, {b,sparse}, {c, sparse}})
         ),
  {
    TestCase({{b, {{{0},  1.0}, {{1},   2.0}, {{3},  3.0}}},
              {c, {{{1}, 10.0}, {{2},  20.0}, {{4}, 30.0}}}},
             {{a, {{{1}, 20.0}}}})
  }
)

TEST_STMT(vector_div,
  forall(i,
         a(i) = b(i) / c(i)
         ),
  Values(
         Formats({{a, dense}, {b, dense}, {c, dense}}),
         Formats({{a, dense}, {b,sparse}, {c, dense}}),
         Formats({{a, dense}, {b, dense}, {c, sparse}}),
         Formats({{a, dense}, {b,sparse}, {c, sparse}}),
         Formats({{a,sparse}, {b,sparse}, {c, sparse}})
         ),
  {
    TestCase({{b, {{{0}, 2.0}, {{1}, 1.0}, {{3}, 6.0}}},
              {c, {{{0}, 1.0}, {{1}, 2.0}, {{2}, 3.0}, {{3}, 4.0}, 
                   {{4}, 5.0}}}},
             {{a, {{{0}, 2.0}, {{1}, 0.5}, {{3}, 1.5}}}})
  }
)

TEST_STMT(vector_intdiv,
  forall(i,
         a(i) = Cast(Cast(b(i), Int64) / Cast(c(i), Int64), Float64)
         ),
  Values(
         Formats({{a, dense}, {b, dense}, {c, dense}}),
         Formats({{a, dense}, {b,sparse}, {c, dense}}),
         Formats({{a, dense}, {b, dense}, {c, sparse}}),
         Formats({{a, dense}, {b,sparse}, {c, sparse}}),
         Formats({{a,sparse}, {b,sparse}, {c, sparse}})
         ),
  {
    TestCase({{b, {{{0}, 2.0}, {{1}, 1.0}, {{3}, 6.0}}},
              {c, {{{0}, 1.0}, {{1}, 2.0}, {{2}, 3.0}, {{3}, 4.0}, 
                   {{4}, 5.0}}}},
             {{a, {{{0}, 2.0}, {{3}, 1.0}}}})
  }
)

TEST_STMT(vector_add,
  forall(i,
         a(i) = b(i) + c(i)
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}}),
         Formats({{a,sparse}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0},  1.0}, {{3},  2.0}}},
              {c, {{{0}, 10.0}, {{2},  20.0}, {{4}, 30.0}}}},
             {{a, {{{0}, 11.0}, {{2},  20.0}, {{3},  2.0}, {{4}, 30.0}}}})
  }
)

TEST_STMT(vector_sub,
  forall(i,
         a(i) = b(i) - c(i)
         ),
  Values(
         Formats({{a,dense},  {b,dense}}),
         Formats({{a,dense},  {b,sparse}}),
         Formats({{a,sparse}, {b,dense}}),
         Formats({{a,sparse}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0},  1.0}, {{3},  2.0}}},
              {c, {{{0}, 10.0}, {{2},  20.0}, {{4}, 30.0}}}},
             {{a, {{{0}, -9.0}, {{2}, -20.0}, {{3},  2.0}, {{4}, -30.0}}}})
  }
)

TEST_STMT(vector_inner_product,
  forall(i,
         alpha += b(i) * c(i)
         ),
  Values(
         Formats({{b,dense},  {c,dense}}),
         Formats({{b,dense},  {c,sparse}}),
         Formats({{b,sparse}, {c,dense}}),
         Formats({{b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0},  1.0}, {{2},  2.0}, {{3},  3.0}}},
              {c, {{{0}, 10.0},              {{3}, 20.0}, {{4}, 30.0}}}},
             {{alpha, {{{}, 70.0}}}})
  }
)

TEST_STMT(vector_or,
  forall(i,
         a(i) = Cast(Cast(b(i), taco::Bool) + Cast(c(i), taco::Bool), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{3}, 1.0}}},
              {c, {{{0}, 0.0}, {{2}, 1.0}, {{4}, 1.0}}}},
             {{a, {{{0}, 1.0}, {{2}, 1.0}, {{3}, 1.0}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_and,
  forall(i,
         a(i) = Cast(Cast(b(i), taco::Bool) * Cast(c(i), taco::Bool), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{1}, 1.0}, {{2}, 0.0}, {{3}, 1.0}}},
              {c, {{{1}, 1.0}, {{2}, 1.0}, {{4}, 1.0}}}},
             {{a, {{{1}, 1.0}}}})
  }
)


// Test matrix operations
TEST_STMT(matrix_neg,
  forall(i,
         forall(j,
                A(i,j) = -B(i,j)
         )),
  Values(
         Formats({{A,Format({ dense, dense})}, {B,Format({ dense, dense})}}),
         Formats({{A,Format({ dense, dense})}, {B,Format({ dense,sparse})}}),
         Formats({{A,Format({ dense, dense})}, {B,Format({sparse, dense})}}),
         Formats({{A,Format({ dense, dense})}, {B,Format({sparse,sparse})}}),

         Formats({{A,Format({ dense,sparse})}, {B,Format({ dense, dense})}}),
         Formats({{A,Format({ dense,sparse})}, {B,Format({ dense,sparse})}}),
         Formats({{A,Format({ dense,sparse})}, {B,Format({sparse, dense})}}),
         Formats({{A,Format({ dense,sparse})}, {B,Format({sparse,sparse})}}),

         Formats({{A,Format({sparse, dense})}, {B,Format({ dense, dense})}}),
         Formats({{A,Format({sparse, dense})}, {B,Format({ dense,sparse})}}),
         Formats({{A,Format({sparse, dense})}, {B,Format({sparse, dense})}}),
         Formats({{A,Format({sparse, dense})}, {B,Format({sparse,sparse})}}),

         Formats({{A,Format({sparse,sparse})}, {B,Format({ dense, dense})}}),
         Formats({{A,Format({sparse,sparse})}, {B,Format({ dense,sparse})}}),
         Formats({{A,Format({sparse,sparse})}, {B,Format({sparse, dense})}}),
         Formats({{A,Format({sparse,sparse})}, {B,Format({sparse,sparse})}})
         ),
  {
    TestCase({{B, {{{0,0},  42.0}, {{0,2},  2.0}, {{1,1},  3.0}, {{3,3},  4.0}}}},
             {{A, {{{0,0}, -42.0}, {{0,2}, -2.0}, {{1,1}, -3.0}, {{3,3}, -4.0}}}})
  }
)

TEST_STMT(matrix_sum,
  forall(i,
         forall(j,
                alpha() += B(i,j)
         )),
  Values(
         Formats({{B,Format({ dense, dense})}}),
         Formats({{B,Format({ dense,sparse})}}),
         Formats({{B,Format({sparse, dense})}}),
         Formats({{B,Format({sparse,sparse})}})
         ),
  {
    TestCase({{B, {{{0,0},42.0}, {{0,2},2.0}, {{1,1},3.0}, {{3,3},4.0}}}},
             {{alpha, {{{}, 51.0}}}})
  }
)

TEST_STMT(matrix_rowsum,
  forall(i,
         forall(j,
                a(i) += B(i,j)
         )),
  Values(
         Formats({{a,dense}, {B,Format({ dense, dense})}}),
         Formats({{a,dense}, {B,Format({ dense,sparse})}}),
         Formats({{a,dense}, {B,Format({sparse, dense})}}),
         Formats({{a,dense}, {B,Format({sparse,sparse})}}),
         Formats({{a,sparse}, {B,Format({ dense, dense})}}),
         Formats({{a,sparse}, {B,Format({ dense,sparse})}}),
         Formats({{a,sparse}, {B,Format({sparse, dense})}}),
         Formats({{a,sparse}, {B,Format({sparse,sparse})}})
         ),
  {
    TestCase({{B, {{{0,0},42.0}, {{0,2},2.0}, {{1,1},3.0}, {{3,3},4.0}}}},
             {{a, {{{0}, 44.0}, {{1},  3.0}, {{3}, 4.0}}}})
  }
)

TEST_STMT(matrix_vector_mul,
  forall(i,
         forall(j,
                a(i) += B(i,j) * c(j)
         )),
  Values(
         Formats({{a,Format({ dense})},
                  {B,Format({ dense,  dense})}, {c,Format({ dense})}}),
         Formats({{a,Format({ dense})},
                  {B,Format({ dense, sparse})}, {c,Format({ dense})}}),
         Formats({{a,Format({ dense})},
                  {B,Format({sparse,  dense})}, {c,Format({ dense})}}),
         Formats({{a,Format({ dense})},
                  {B,Format({sparse, sparse})}, {c,Format({ dense})}})
         ),
  {
    TestCase({{B, {{{0,0}, 42.0}, {{0,2}, 2.0}, {{1,1}, 3.0}, {{3,3}, 4.0}}},
              {c, {{{2}, 1.0}, {{3}, 2.0}}}},
             {{a, {{{0}, 2.0}, {{3}, 8.0}}}})
  }
)


// Test tensor operations

TEST_STMT(tensor_slicesum,
  forall(i,
         forall(j,
                forall(k,
                       A(i,j) += T(i,j,k)
         ))),
  Values(
         Formats({{A,Format({ dense, dense})}, {T,Format({ dense, dense,dense})}}),
         Formats({{A,Format({ dense, dense})}, {T,Format({ dense,sparse,dense})}}),
         Formats({{A,Format({ dense, dense})}, {T,Format({sparse, dense,dense})}}),
         Formats({{A,Format({ dense, dense})}, {T,Format({sparse,sparse,dense})}}),

         Formats({{A,Format({ dense,sparse})}, {T,Format({ dense, dense,dense})}}),
         Formats({{A,Format({ dense,sparse})}, {T,Format({ dense,sparse,dense})}}),
         Formats({{A,Format({ dense,sparse})}, {T,Format({sparse, dense,dense})}}),
         Formats({{A,Format({ dense,sparse})}, {T,Format({sparse,sparse,dense})}}),

         Formats({{A,Format({sparse, dense})}, {T,Format({ dense, dense,dense})}}),
         Formats({{A,Format({sparse, dense})}, {T,Format({ dense,sparse,dense})}}),
         Formats({{A,Format({sparse, dense})}, {T,Format({sparse, dense,dense})}}),
         Formats({{A,Format({sparse, dense})}, {T,Format({sparse,sparse,dense})}}),

         Formats({{A,Format({sparse,sparse})}, {T,Format({ dense, dense,dense})}}),
         Formats({{A,Format({sparse,sparse})}, {T,Format({ dense,sparse,dense})}}),
         Formats({{A,Format({sparse,sparse})}, {T,Format({sparse, dense,dense})}}),
         Formats({{A,Format({sparse,sparse})}, {T,Format({sparse,sparse,dense})}})
         ),
  {
    TestCase({{T, {{{0,0,0},1.0}, {{0,0,1},7.0}, {{0,2,1},5.0}, {{2,0,1},2.0}, {{2,2,0},4.0}, {{2,2,1},8.0}, {{2,3,0},3.0}, {{2,3,1},9.0}}}},
             {{A, {{{0,0},8.0}, {{0,2},5.0}, {{2,0},2.0}, {{2,2},12.0}, {{2,3},12.0}}}})
  }
)


// Test where statements

TEST_STMT(where_scalar,
  where(alpha = beta * delta, delta = zeta + eta),
  Values(Formats()),
  {
    TestCase({{beta,  {{{},  2.0}}},
              {zeta,  {{{},  400.0}}},
              {eta, {{{},    5000.0}}}},
             {{alpha, {{{},  10800.0}}}})
  }
)

TEST_STMT(where_vector_sum,
  where(alpha = t(),
        forall(i,
               t += b(i))),
  Values(Formats({{b,dense}}),
         Formats({{b,sparse}})
         ),
  {
    TestCase({{b, {{{0},  1.0}, {{2},  2.0}, {{3},  3.0}}}},
             {{alpha, {{{}, 6.0}}}})
  }
)

TEST_STMT(where_matrix_sum,
  where(alpha = ti(),
        forall(i,
               where(ti += tj(),
                     forall(j, tj += B(i,j))))),
  Values(
         Formats({{B,Format({ dense, dense})}}),
         Formats({{B,Format({ dense,sparse})}}),
         Formats({{B,Format({sparse, dense})}}),
         Formats({{B,Format({sparse,sparse})}})
         ),
  {
    TestCase({{B, {{{0,0},42.0}, {{0,2},2.0}, {{1,1},3.0}, {{3,3},4.0}}}},
             {{alpha, {{{}, 51.0}}}})
  }
)

TEST_STMT(where_matrix_vector_mul,
  forall(i,
         where(a(i) = tj,
               forall(j,
                      tj += B(i,j) * c(j)))),
  Values(
         Formats({{a,Format({ dense})},
                  {B,Format({ dense,  dense})}, {c,Format({ dense})}}),
         Formats({{a,Format({ dense})},
                  {B,Format({ dense, sparse})}, {c,Format({ dense})}}),
         Formats({{a,Format({ dense})},
                  {B,Format({sparse,  dense})}, {c,Format({ dense})}}),
         Formats({{a,Format({ dense})},
                  {B,Format({sparse, sparse})}, {c,Format({ dense})}})
         ),
  {
    TestCase({{B, {{{0,0}, 42.0}, {{0,2}, 2.0}, {{1,1}, 3.0}, {{3,3}, 4.0}}},
              {c, {{{2}, 1.0}, {{3}, 2.0}}}},
             {{a, {{{0}, 2.0}, {{3}, 8.0}}}})
  }
)

TEST_STMT(DISABLED_where_spmm,
  forall(i,
         where(forall(j,
                      A(i,j) = w(j)),
               forall(k,
                      forall(j,
                             w(j) += B(i,k) * C(k,j))))),
  Values(
//         Formats({{A,Format({dense,dense})},
//                  {B,Format({dense,dense})}, {C,Format({dense,dense})}}),
         Formats({{A,Format({dense,sparse})},
                  {B,Format({dense,sparse})}, {C,Format({dense,sparse})}})
         ),
  {
    TestCase({{B, { {{0,1}, 2.0}, {{2,0},  3.0}, {{2,2}, 4.0}} },
              {C, { {{0,0},10.0}, {{0,1}, 20.0}, {{2,1},30.0}} }},
             {{A, { {{2,0},30.0}, {{2,1},180.0} }}})
  }
)


// Test sequence statements

TEST_STMT(sequence_scalar,
  sequence(alpha = beta * delta, alpha += zeta + eta),
  Values(Formats()),
  {
    TestCase({{beta,  {{{},  2.0}}},
              {delta,  {{{}, 30.0}}},
              {zeta,  {{{},  400.0}}},
              {eta, {{{},    5000.0}}}},
             {{alpha, {{{},  5460.0}}}})
  }
)


// Test multi statements

TEST_STMT(multi_scalar,
  multi(alpha = delta * zeta, beta = zeta + eta),
  Values(Formats()),
  {
    TestCase({{delta, {{{},     30.0}}},
              {zeta,  {{{},    400.0}}},
              {eta,   {{{},   5000.0}}}},
             {{alpha, {{{},  12000.0}}},
              {beta,  {{{},   5400.0}}}})
  }
)


// Test transpose operations

TEST_STMT(matrix_transposed_output,
  forall(i,
         forall(j,
                A(j,i) = -B(i,j)
         )),
  Values(
         Formats({{A,Format({ dense, dense})}, {B,Format({ dense, dense})}}),
         Formats({{A,Format({ dense, dense})}, {B,Format({ dense,sparse})}}),
         Formats({{A,Format({ dense, dense})}, {B,Format({sparse, dense})}}),
         Formats({{A,Format({ dense, dense})}, {B,Format({sparse,sparse})}})
         ),
  {
    TestCase({{B, {{{0,0},  42.0}, {{0,2},  2.0}, {{1,3},  3.0}, {{3,2},  4.0}}}},
             {{A, {{{0,0}, -42.0}, {{2,0}, -2.0}, {{2,3}, -4.0}, {{3,1}, -3.0}}}})
  }
)

TEST_STMT(matrix_transposed_input,
  forall(i,
         forall(j,
                A(i,j) = B(i,j) + C(j,i)
         )),
  Values(
         Formats({{A,Format({ dense, dense})}, {B,Format({ dense, dense})},
                 {C,Format({dense,dense})}}),
         Formats({{A,Format({ dense,sparse})}, {B,Format({ dense,sparse})},
                  {C,Format({dense,dense})}}),
         Formats({{A,Format({sparse, dense})}, {B,Format({sparse, dense})},
                  {C,Format({dense,dense})}}),
         Formats({{A,Format({sparse,sparse})}, {B,Format({sparse,sparse})},
                  {C,Format({dense,dense})}})
         ),
  {
    TestCase({{B, {{{0,0}, 42.0}, {{0,2}, 2.0}, {{1,3}, 3.0}, {{3,2}, 4.0}}},
              {C, {{{0,0}, 42.0}, {{0,2}, 2.0}, {{1,3}, 3.0}, {{3,2}, 4.0}}}},
             {{A, {{{0,0}, 84.0}, {{0,2}, 2.0}, {{1,3}, 3.0}, {{2,0}, 2.0}, {{2,3}, 4.0}, {{3,1}, 3.0}, {{3,2}, 4.0}}}})
  }
)


// Test broadcast operations

TEST_STMT(broadcast_vector_mul_scalar,
  forall(i,
         a(i) = beta * c(i)
         ),
  Values(
         Formats({{a,dense},  {c,dense}}),
         Formats({{a,dense},  {c,sparse}}),
         Formats({{a,sparse}, {c,dense}}),
         Formats({{a,sparse}, {c,sparse}})
         ),
  {
    TestCase({{beta,  {{{},  42.0}}},
              {c, {{{0}, 1.0}, {{2},  2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, 42.0}, {{2},  84.0}, {{4}, 126.0}}}})
  }
)

TEST_STMT(broadcast_vector_add_scalar,
  forall(i,
         a(i) = beta + c(i)
         ),
  Values(
         Formats({{a,dense},  {c,dense}}),
         Formats({{a,dense},  {c,sparse}}),
         Formats({{a,sparse}, {c,dense}}),
         Formats({{a,sparse}, {c,sparse}})
         ),
  {
    TestCase({{beta,  {{{},  42.0}}},
              {c, {{{0}, 1.0}, {{2},  2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, 43.0}, {{1},  42.0}, {{2},  44.0},
                   {{3},  42.0}, {{4}, 45.0}}}})
  }
)

TEST_STMT(broadcast_vector_mul_constant,
  forall(i,
         a(i) = 42.0 * c(i)
         ),
  Values(
         Formats({{a,dense},  {c,dense}}),
         Formats({{a,dense},  {c,sparse}}),
         Formats({{a,sparse}, {c,dense}}),
         Formats({{a,sparse}, {c,sparse}})
         ),
  {
    TestCase({{c, {{{0}, 1.0}, {{2},  2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, 42.0}, {{2},  84.0}, {{4}, 126.0}}}})
  }
)

TEST_STMT(broadcast_vector_add_constant,
  forall(i,
         a(i) = 42.0 + c(i)
         ),
  Values(
         Formats({{a,dense},  {c,dense}}),
         Formats({{a,dense},  {c,sparse}}),
         Formats({{a,sparse}, {c,dense}}),
         Formats({{a,sparse}, {c,sparse}})
         ),
  {
    TestCase({{c, {{{0}, 1.0}, {{2},  2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, 43.0}, {{1},  42.0}, {{2},  44.0},
                   {{3},  42.0}, {{4}, 45.0}}}})
  }
)

// Test intrinsics

TEST_STMT(vector_mod,
  forall(i,
         a(i) = mod(b(i), c(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 6.0}}},
              {c, {{{0}, 1.0}, {{1}, 2.0}, {{2}, 3.0}, {{3}, 4.0}, 
                   {{4}, 5.0}}}},
             {{a, {{{2}, 2.0}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_abs,
  forall(i,
         a(i) = abs(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, -2.0}, {{4}, -4.0}}}},
             {{a, {{{0}, 1.0}, {{2},  2.0}, {{4},  4.0}}}})
  }
)

TEST_STMT(vector_pow_constant,
  forall(i,
         a(i) = pow(b(i), 5.0)
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2},  -2.0}, {{4}, 1.0}}}},
             {{a, {{{0}, 1.0}, {{2}, -32.0}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_pow_vector,
  forall(i,
         a(i) = pow(b(i), c(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 4.0}}},
              {c, {{{0}, 1.0}, {{1}, 2.0}, {{2}, 3.0}, {{3}, 4.0}, 
                   {{4}, 5.0}}}},
             {{a, {{{0}, 1.0}, {{2}, 8.0}, {{4}, 1024.0}}}})
  }
)

TEST_STMT(vector_square,
  forall(i,
         a(i) = square(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, -2.0}, {{4}, -4.0}}}},
             {{a, {{{0}, 1.0}, {{2},  4.0}, {{4}, 16.0}}}})
  }
)

TEST_STMT(vector_cube,
  forall(i,
         a(i) = cube(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, -2.0}, {{4},  -4.0}}}},
             {{a, {{{0}, 1.0}, {{2}, -8.0}, {{4}, -64.0}}}})
  }
)

TEST_STMT(vector_sqrt,
  forall(i,
         a(i) = sqrt(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 4.0}, {{4}, 16.0}}}},
             {{a, {{{0}, 1.0}, {{2}, 2.0}, {{4},  4.0}}}})
  }
)

TEST_STMT(vector_product_sqrt,
  forall(i,
         a(i) = sqrt(b(i) * c(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}},
              {c, {{{1}, 1.0}, {{2}, 8.0}, {{4}, 3.0}}}},
             {{a, {{{2}, 4.0}, {{4}, 3.0}}}})
  }
)

TEST_STMT(vector_sum_sqrt,
  forall(i,
         a(i) = sqrt(b(i) + c(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 8.0}, {{4}, 3.0}}},
              {c, {{{1}, 1.0}, {{2}, 8.0}, {{4}, 6.0}}}},
             {{a, {{{0}, 1.0}, {{1}, 1.0}, {{2}, 4.0}, {{4}, 3.0}}}})
  }
)

TEST_STMT(vector_cbrt,
  forall(i,
         a(i) = cbrt(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 8.0}, {{4}, -64.0}}}},
             {{a, {{{0}, 1.0}, {{2}, 2.0}, {{4},  -4.0}}}})
  }
)

TEST_STMT(vector_exp,
  forall(i,
         a(i) = exp(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::exp(1.0)}, {{1}, 1.0}, {{2}, std::exp(2.0)}, 
                   {{3}, 1.0}, {{4}, std::exp(3.0)}}}})
  }
)

TEST_STMT(vector_product_exp,
  forall(i,
         a(i) = exp(b(i) * c(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 4.0}}},
              {c, {{{1}, 1.0}, {{2}, 3.0}, {{4}, 5.0}}}},
             {{a, {{{0}, 1.0}, {{1}, 1.0}, {{2}, std::exp(6.0)}, {{3}, 1.0}, 
                   {{4}, std::exp(20.0)}}}})
  }
)

TEST_STMT(vector_log,
  forall(i,
         a(i) = log(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{1}, 2.0}, {{2}, 3.0}, {{3}, 4.0}, 
                   {{4}, 5.0}}}},
             {{a, {{{0}, 0.0}, {{1}, std::log(2.0)}, {{2}, std::log(3.0)}, 
                   {{3}, std::log(4.0)}, {{4}, std::log(5.0)}}}})
  }
)

TEST_STMT(vector_log10,
  forall(i,
         a(i) = log10(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{1}, 2.0}, {{2}, 3.0}, {{3}, 4.0}, 
                   {{4}, 5.0}}}},
             {{a, {{{0}, 0.0}, {{1}, std::log10(2.0)}, {{2}, std::log10(3.0)}, 
                   {{3}, std::log10(4.0)}, {{4}, std::log10(5.0)}}}})
  }
)

TEST_STMT(vector_sin,
  forall(i,
         a(i) = sin(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::sin(1.0)}, {{2}, std::sin(2.0)}, 
                   {{4}, std::sin(3.0)}}}})
  }
)

TEST_STMT(vector_cos,
  forall(i,
         a(i) = cos(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::cos(1.0)}, {{1}, 1.0}, {{2}, std::cos(2.0)}, 
                   {{3}, 1.0}, {{4}, std::cos(3.0)}}}})
  }
)

TEST_STMT(vector_tan,
  forall(i,
         a(i) = tan(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::tan(1.0)}, {{2}, std::tan(2.0)}, 
                   {{4}, std::tan(3.0)}}}})
  }
)

TEST_STMT(vector_asin,
  forall(i,
         a(i) = asin(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::asin(1.0)}, {{2}, std::asin(2.0)}, 
                   {{4}, std::asin(3.0)}}}})
  }
)

TEST_STMT(vector_acos,
  forall(i,
         a(i) = acos(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::acos(1.0)}, {{1}, std::acos(0.0)}, 
                   {{2}, std::acos(2.0)}, {{3}, std::acos(0.0)}, 
                   {{4}, std::acos(3.0)}}}})
  }
)

TEST_STMT(vector_atan,
  forall(i,
         a(i) = atan(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::atan(1.0)}, {{2}, std::atan(2.0)}, 
                   {{4}, std::atan(3.0)}}}})
  }
)

TEST_STMT(vector_atan2_constant,
  forall(i,
         a(i) = atan2(b(i), 5.0)
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, -2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::atan2(1.0, 5.0)}, {{2}, std::atan2(-2.0, 5.0)}, 
                   {{4}, std::atan2(3.0, 5.0)}}}})
  }
)

TEST_STMT(vector_atan2_vector,
  forall(i,
         a(i) = atan2(b(i), c(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 4.0}}},
              {c, {{{1}, 2.0}, {{2}, 3.0}, {{3}, 4.0}, {{4}, 5.0}}}},
             {{a, {{{0}, std::atan2(1.0, 0.0)}, {{1}, std::atan2(0.0, 2.0)},
                   {{2}, std::atan2(2.0, 3.0)}, {{3}, std::atan2(0.0, 3.0)},
                   {{4}, std::atan2(4.0, 5.0)}}}})
  }
)

TEST_STMT(vector_sinh,
  forall(i,
         a(i) = sinh(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::sinh(1.0)}, {{2}, std::sinh(2.0)}, 
                   {{4}, std::sinh(3.0)}}}})
  }
)

TEST_STMT(vector_cosh,
  forall(i,
         a(i) = cosh(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::cosh(1.0)}, {{1}, std::cosh(0.0)}, 
                   {{2}, std::cosh(2.0)}, {{3}, std::cosh(0.0)}, 
                   {{4}, std::cosh(3.0)}}}})
  }
)

TEST_STMT(vector_tanh,
  forall(i,
         a(i) = tanh(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::tanh(1.0)}, {{2}, std::tanh(2.0)}, 
                   {{4}, std::tanh(3.0)}}}})
  }
)

TEST_STMT(vector_asinh,
  forall(i,
         a(i) = asinh(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::asinh(1.0)}, {{2}, std::asinh(2.0)}, 
                   {{4}, std::asinh(3.0)}}}})
  }
)

TEST_STMT(vector_acosh,
  forall(i,
         a(i) = acosh(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::acosh(1.0)}, {{1}, std::acosh(0.0)}, 
                   {{2}, std::acosh(2.0)}, {{3}, std::acosh(0.0)}, 
                   {{4}, std::acosh(3.0)}}}})
  }
)

TEST_STMT(vector_atanh,
  forall(i,
         a(i) = atanh(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, std::atanh(1.0)}, {{2}, std::atanh(2.0)}, 
                   {{4}, std::atanh(3.0)}}}})
  }
)

TEST_STMT(vector_gt_positive_constant,
  forall(i,
         a(i) = Cast(gt(b(i), 1.0), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 5.0}}}},
             {{a, {{{2}, 1.0}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_gt_negative_constant,
  forall(i,
         a(i) = Cast(gt(b(i), -1.0), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, -2.0}, {{4}, -5.0}}}},
             {{a, {{{0}, 1.0}, {{1}, 1.0}, {{3}, 1.0}}}})
  }
)

TEST_STMT(vector_gt_vector,
  forall(i,
         a(i) = Cast(gt(b(i), c(i)), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 5.0}}},
              {c, {{{1}, 1.0}, {{2}, 3.0}, {{4}, 4.0}}}},
             {{a, {{{0}, 1.0}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_lt_positive_constant,
  forall(i,
         a(i) = Cast(lt(b(i), 1.0), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, -2.0}, {{4}, 5.0}}}},
             {{a, {{{1}, 1.0}, {{2}, 1.0}, {{3}, 1.0}}}})
  }
)

TEST_STMT(vector_lt_negative_constant,
  forall(i,
         a(i) = Cast(lt(b(i), -1.0), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, -2.0}, {{4}, -5.0}}}},
             {{a, {{{2}, 1.0}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_lt_vector,
  forall(i,
         a(i) = Cast(lt(b(i), c(i)), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 5.0}}},
              {c, {{{1}, 1.0}, {{2}, 3.0}, {{4}, 4.0}}}},
             {{a, {{{1}, 1.0}, {{2}, 1.0}}}})
  }
)

TEST_STMT(vector_gte_zero,
  forall(i,
         a(i) = Cast(gte(b(i), 0.0), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, -1.0}, {{2}, -2.0}, {{4}, 5.0}}}},
             {{a, {{{1}, 1.0}, {{3}, 1.0}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_gte_vector,
  forall(i,
         a(i) = Cast(gte(b(i), c(i)), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 5.0}}},
              {c, {{{1}, 1.0}, {{2}, 3.0}, {{4}, 4.0}}}},
             {{a, {{{0}, 1.0}, {{3}, 1.0}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_lte_zero,
  forall(i,
         a(i) = Cast(lte(b(i), 0.0), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, -1.0}, {{2}, -2.0}, {{4}, 5.0}}}},
             {{a, {{{0}, 1.0}, {{1}, 1.0}, {{2}, 1.0}, {{3}, 1.0}}}})
  }
)

TEST_STMT(vector_lte_vector,
  forall(i,
         a(i) = Cast(lte(b(i), c(i)), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 5.0}}},
              {c, {{{1}, 1.0}, {{2}, 3.0}, {{4}, 4.0}}}},
             {{a, {{{1}, 1.0}, {{2}, 1.0}, {{3}, 1.0}}}})
  }
)

TEST_STMT(vector_eq,
  forall(i,
         a(i) = Cast(eq(b(i), c(i)), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 5.0}}},
              {c, {{{1}, 1.0}, {{2}, 2.0}, {{4}, 4.0}}}},
             {{a, {{{2}, 1.0}, {{3}, 1.0}}}})
  }
)

TEST_STMT(vector_neq,
  forall(i,
         a(i) = Cast(neq(b(i), c(i)), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 5.0}}},
              {c, {{{1}, 1.0}, {{2}, 2.0}, {{4}, 4.0}}}},
             {{a, {{{0}, 1.0}, {{1}, 1.0}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_max,
  forall(i,
         a(i) = taco::max(b(i), c(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 5.0}}},
              {c, {{{1}, 1.0}, {{2}, 3.0}, {{4}, 4.0}}}},
             {{a, {{{0}, 1.0}, {{1}, 1.0}, {{2}, 3.0}, {{4}, 5.0}}}})
  }
)

TEST_STMT(vector_min,
  forall(i,
         a(i) = taco::min(b(i), c(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}, {c,dense}}),
         Formats({{a,dense}, {b,dense}, {c,sparse}}),
         Formats({{a,dense}, {b,sparse}, {c,dense}}),
         Formats({{a,dense}, {b,sparse}, {c,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 2.0}, {{4}, 5.0}}},
              {c, {{{1}, -1.0}, {{2}, 3.0}, {{4}, 4.0}}}},
             {{a, {{{1}, -1.0}, {{2}, 2.0}, {{4}, 4.0}}}})
  }
)

TEST_STMT(vector_heaviside,
  forall(i,
         a(i) = heaviside(b(i))
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, -2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, 1.0}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_heaviside_half_maximum,
  forall(i,
         a(i) = heaviside(b(i), 0.5)
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, -2.0}, {{4}, 3.0}}}},
             {{a, {{{0}, 1.0}, {{1}, 0.5}, {{3}, 0.5}, {{4}, 1.0}}}})
  }
)

TEST_STMT(vector_not,
  forall(i,
         a(i) = Cast(Not(Cast(b(i), taco::Bool)), Float64)
         ),
  Values(
         Formats({{a,dense}, {b,dense}}),
         Formats({{a,dense}, {b,sparse}})
         ),
  {
    TestCase({{b, {{{0}, 1.0}, {{2}, 0.0}, {{4}, 1.0}}}},
             {{a, {{{1}, 1.0}, {{2}, 1.0}, {{3}, 1.0}}}})
  }
)

}}
