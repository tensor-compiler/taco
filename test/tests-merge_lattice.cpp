#include "test.h"

#include <vector>
#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/lower/merge_lattice.h"
#include "taco/lower/iterator.h"
#include "lower/mode_access.h"
#include "taco/ir/ir.h"
#include "taco/lower/mode_format_impl.h"

using namespace std;

using namespace taco;

// Temporary hack until dense in format.h is transition from the old system
#include "taco/lower/mode_format_dense.h"
taco::ModeFormat dense_new(std::make_shared<taco::DenseModeFormat>());
#define dense dense_new

namespace tests {

class HashedModeFormat : public ModeFormatImpl {
public:
  HashedModeFormat() : ModeFormatImpl("hashed", false, false, true, false,
                                      false, false, true, true, true, false) {}

  ModeFormat copy(std::vector<ModeFormat::Property> properties) const {
    return ModeFormat(std::make_shared<HashedModeFormat>());
  }

  ir::Expr getSize(ir::Expr parentSize, Mode mode) const {
    return parentSize;
  }

  vector<ir::Expr> getArrays(ir::Expr tensor, int mode, int level) const {
    return {};
  }
};
ModeFormat hashed(make_shared<HashedModeFormat>());

static const Dimension n;
static const Type vectype(Float64, {n});

static TensorVar r1t("r1", vectype, Format(dense));
static TensorVar r2t("r2", vectype, Format(sparse));

static TensorVar d1t("d1", vectype, Format(dense));
static TensorVar d2t("d2", vectype, Format(dense));
static TensorVar d3t("d3", vectype, Format(dense));
static TensorVar d4t("d4", vectype, Format(dense));
static TensorVar s1t("s1", vectype, Format(compressed));
static TensorVar s2t("s2", vectype, Format(compressed));
static TensorVar s3t("s3", vectype, Format(compressed));
static TensorVar s4t("s4", vectype, Format(compressed));
static TensorVar h1t("h1", vectype, Format(hashed));
static TensorVar h2t("h2", vectype, Format(hashed));
static TensorVar h3t("h3", vectype, Format(hashed));
static TensorVar h4t("h4", vectype, Format(hashed));

static map<TensorVar, taco::ir::Expr> tensorVars {
  {r1t, taco::ir::Var::make("r1", taco::Int())},
  {r2t, taco::ir::Var::make("r2", taco::Int())},
  {d1t, taco::ir::Var::make("d1", taco::Int())},
  {d2t, taco::ir::Var::make("d2", taco::Int())},
  {d3t, taco::ir::Var::make("d3", taco::Int())},
  {d4t, taco::ir::Var::make("d4", taco::Int())},
  {s1t, taco::ir::Var::make("s1", taco::Int())},
  {s2t, taco::ir::Var::make("s2", taco::Int())},
  {s3t, taco::ir::Var::make("s3", taco::Int())},
  {s4t, taco::ir::Var::make("s4", taco::Int())},
  {h1t, taco::ir::Var::make("h1", taco::Int())},
  {h2t, taco::ir::Var::make("h2", taco::Int())},
  {h3t, taco::ir::Var::make("h3", taco::Int())},
  {h4t, taco::ir::Var::make("h4", taco::Int())}
};

static IndexVar i("i");

static Access rd = r1t(i);
static Access rs = r2t(i);
static Access d1 = d1t(i);
static Access d2 = d2t(i);
static Access d3 = d3t(i);
static Access d4 = d4t(i);
static Access s1 = s1t(i);
static Access s2 = s2t(i);
static Access s3 = s3t(i);
static Access s4 = s4t(i);
static Access h1 = h1t(i);
static Access h2 = h2t(i);
static Access h3 = h3t(i);
static Access h4 = h4t(i);

static map<IndexVar, taco::ir::Expr> coordVars;
static Forall dummy = forall(i, rd = rd + rs +
                                     d1 + d2 + d3 + d4 +
                                     s1 + s2 + s3 + s4 +
                                     h1 + h2 + h3 + h4);
static Iterators iterators = Iterators(dummy, tensorVars);

static Iterator it(Access access)
{
  return iterators.levelIterator(ModeAccess(access,1));
}

struct Test {
  Test(Forall forall, taco::MergeLattice expected)
      : forall(forall), expected(expected) {}
  Forall forall;
  taco::MergeLattice expected;
};
std::ostream& operator<<(std::ostream& os, const Test& test) {
  return os << test.forall;
}

struct merge_lattice : public TestWithParam<Test> {};

TEST_P(merge_lattice, test) {
  Forall forall = GetParam().forall;
  taco::MergeLattice lattice = taco::MergeLattice::make(forall, iterators);
  ASSERT_EQ(GetParam().expected, lattice);
}

INSTANTIATE_TEST_CASE_P(copy, merge_lattice,
  Values(Test(forall(i, rd = d1),
              MergeLattice({MergePoint({i},
                                       {it(d1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = s1),
              MergeLattice({MergePoint({it(s1)},
                                       {},
                                       {it(rd)})
                           })
              )
         )
);

INSTANTIATE_TEST_CASE_P(neg, merge_lattice,
  Values(Test(forall(i, rd = -d1),
              MergeLattice({MergePoint({i},
                                       {it(d1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = -s1),
              MergeLattice({MergePoint({it(s1)},
                                       {},
                                       {it(rd)})
                           })
              )
         )
);


INSTANTIATE_TEST_CASE_P(mul, merge_lattice,
  Values(Test(forall(i, rd = d1 * d2),
              MergeLattice({MergePoint({i},
                                       {it(d1), it(d2)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = d1 * d2 * d3),
              MergeLattice({MergePoint({i},
                                       {it(d1), it(d2), it(d3)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = s1 * s2),
              MergeLattice({MergePoint({it(s1), it(s2)},
                                       {},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = s1 * s2 * s3),
              MergeLattice({MergePoint({it(s1), it(s2), it(s3)},
                                       {},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = s1 * d1),
              MergeLattice({MergePoint({it(s1)},
                                       {it(d1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = d1 * s1),
              MergeLattice({MergePoint({it(s1)},
                                       {it(d1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (d1 * d2) * s1),
              MergeLattice({MergePoint({it(s1)},
                                       {it(d1), it(d2)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 * s2) * d1),
              MergeLattice({MergePoint({it(s1), it(s2)},
                                       {it(d1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 * d1) * d2),
              MergeLattice({MergePoint({it(s1)},
                                       {it(d1), it(d2)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = d2 * (s1 * d1)),
              MergeLattice({MergePoint({it(s1)},
                                       {it(d2), it(d1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 * d1) * s2),
              MergeLattice({MergePoint({it(s1), it(s2)},
                                       {it(d1)},
                                       {it(rd)})
                           })
              )
        )
);

INSTANTIATE_TEST_CASE_P(add, merge_lattice,
  Values(Test(forall(i, rd = d1 + d2),
              MergeLattice({MergePoint({i},
                                       {it(d1), it(d2)},
                                       {it(rd)}),
                           })
              ),
         Test(forall(i, rd = d1 + d2 + d3),
              MergeLattice({MergePoint({i},
                                       {it(d1), it(d2), it(d3)},
                                       {it(rd)}),
                           })
              ),
         Test(forall(i, rd = s1 + s2),
              MergeLattice({MergePoint({it(s1), it(s2)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s2)},
                                       {},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = s1 + s2 + s3),
              MergeLattice({MergePoint({it(s1), it(s2), it(s3)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s3)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s2), it(s3)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s2)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s2)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s3)},
                                       {},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = d1 + s2),
              MergeLattice({MergePoint({i, it(s2)},
                                       {it(d1)},
                                       {it(rd)}),
                            MergePoint({i},
                                       {it(d1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = s1 + d2),
              MergeLattice({MergePoint({it(s1), i},
                                       {it(d2)},
                                       {it(rd)}),
                            MergePoint({i},
                                       {it(d2)},
                                       {it(rd)})
                           })
              )
        )
);

INSTANTIATE_TEST_CASE_P(add_multiply, merge_lattice,
  Values(Test(forall(i, rd = (d1 + d2) * d3),
              MergeLattice({MergePoint({i},
                                       {it(d1), it(d2), it(d3)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 + s2) * d3),
              MergeLattice({MergePoint({it(s1), it(s2)},
                                       {it(d3)},
                                       {it(rd)}),
                            MergePoint({it(s1)},
                                       {it(d3)},
                                       {it(rd)}),
                            MergePoint({it(s2)},
                                       {it(d3)},
                                       {it(rd)}),
                           })
              ),
         Test(forall(i, rd = (d1 + d2) * s3),
              MergeLattice({MergePoint({it(s3)},
                                       {it(d1), it(d2)},
                                       {it(rd)}),
                           })
              ),
         Test(forall(i, rd = (s1 + d2) * d3),
              MergeLattice({MergePoint({it(s1), i},
                                       {it(d2), it(d3)},
                                       {it(rd)}),
                            MergePoint({i},
                                       {it(d2), it(d3)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 + s2) * s3),
              MergeLattice({MergePoint({it(s1), it(s2), it(s3)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s3)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s2), it(s3)},
                                       {},
                                       {it(rd)})
                           })
              )
        )
);

INSTANTIATE_TEST_CASE_P(multiply_add, merge_lattice,
  Values(Test(forall(i, rd = (d1 * d2) + d3),
              MergeLattice({MergePoint({i},
                                       {it(d1), it(d2), it(d3)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 * s2) + d3),
              MergeLattice({MergePoint({it(s1), it(s2), i},
                                       {it(d3)},
                                       {it(rd)}),
                            MergePoint({i},
                                       {it(d3)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (d1 * d2) + s3),
              MergeLattice({MergePoint({i, it(s3)},
                                       {it(d1), it(d2)},
                                       {it(rd)}),
                            MergePoint({i},
                                       {it(d1), it(d2)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 * d2) + d3),
              MergeLattice({MergePoint({it(s1), i},
                                       {it(d2), it(d3)},
                                       {it(rd)}),
                            MergePoint({i},
                                       {it(d3)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 * s2) + s3),
              MergeLattice({MergePoint({it(s1), it(s2), it(s3)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s2)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s3)},
                                       {},
                                       {it(rd)})
                           })
              )
        )
);

INSTANTIATE_TEST_CASE_P(add_multiply_multiply, merge_lattice,
  Values(Test(forall(i, rd = (d1 + d2) * (d3 * d4)),
              MergeLattice({MergePoint({i},
                                       {it(d1), it(d2), it(d3), it(d4)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (d1 + d2) * (s3 * s4)),
              MergeLattice({MergePoint({it(s3), it(s4)},
                                       {it(d1), it(d2)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 + s2) * (d3 * d4)),
              MergeLattice({MergePoint({it(s1), it(s2)},
                                       {it(d3), it(d4)},
                                       {it(rd)}),
                            MergePoint({it(s1)},
                                       {it(d3), it(d4)},
                                       {it(rd)}),
                            MergePoint({it(s2)},
                                       {it(d3), it(d4)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (d1 + s2) * (s3 * s4)),
              MergeLattice({MergePoint({it(s2), it(s3), it(s4)},
                                       {it(d1)},
                                       {it(rd)}),
                            MergePoint({it(s3), it(s4)},
                                       {it(d1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 + s2) * (s3 * s4)),
              MergeLattice({MergePoint({it(s1), it(s2), it(s3), it(s4)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s3), it(s4)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s2), it(s3), it(s4)},
                                       {},
                                       {it(rd)})
                           })
              )
        )
);

INSTANTIATE_TEST_CASE_P(add_multiply_add, merge_lattice,
  Values(Test(forall(i, rd = (d1 + d2) * (d3 + d4)),
              MergeLattice({MergePoint({i},
                                       {it(d1), it(d2), it(d3), it(d4)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (d1 + d2) * (d3 + s4)),
              MergeLattice({MergePoint({i, it(s4)},
                                       {it(d1), it(d2), it(d3)},
                                       {it(rd)}),
                            MergePoint({i},
                                       {it(d1), it(d2), it(d3)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 + s2) * (s3 + d4)),
              MergeLattice({MergePoint({it(s1), it(s2), it(s3)},
                                       {it(d4)},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s2)},
                                       {it(d4)},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s3)},
                                       {it(d4)},
                                       {it(rd)}),
                            MergePoint({it(s1)},
                                       {it(d4)},
                                       {it(rd)}),
                            MergePoint({it(s2), it(s3)},
                                       {it(d4)},
                                       {it(rd)}),
                            MergePoint({it(s2)},
                                       {it(d4)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = (s1 + s2) * (s3 + s4)),
              MergeLattice({MergePoint({it(s1), it(s2), it(s3), it(s4)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s2), it(s3)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s2), it(s4)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s3), it(s4)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s3)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s1), it(s4)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s2), it(s3), it(s4)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s2), it(s3)},
                                       {},
                                       {it(rd)}),
                            MergePoint({it(s2), it(s4)},
                                       {},
                                       {it(rd)})
                           })
              )
        )
);

INSTANTIATE_TEST_CASE_P(hashmap, merge_lattice,
  Values(Test(forall(i, rd = h1),
              MergeLattice({MergePoint({it(h1)},
                                       {},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = -h1),
              MergeLattice({MergePoint({it(h1)},
                                       {},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = d1 * h1),
              MergeLattice({MergePoint({it(h1)},
                                       {it(d1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = h1 * d1),
              MergeLattice({MergePoint({it(h1)},
                                       {it(d1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = s1 * h1),
              MergeLattice({MergePoint({it(s1)},
                                       {it(h1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = h1 * s1),
              MergeLattice({MergePoint({it(s1)},
                                       {it(h1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = h1 * h2),
              MergeLattice({MergePoint({it(h1)},
                                       {it(h2)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = h1 * h2 * h3),
              MergeLattice({MergePoint({it(h1)},
                                       {it(h2), it(h3)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = d1 + h1),
              MergeLattice({MergePoint({i},
                                       {it(d1), it(h1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = h1 + d1),
              MergeLattice({MergePoint({i},
                                       {it(d1), it(h1)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = h1 + h2),
              MergeLattice({MergePoint({i},
                                       {it(h1), it(h2)},
                                       {it(rd)})
                           })
              ),
         Test(forall(i, rd = h1 + h2 + h3),
              MergeLattice({MergePoint({i},
                                       {it(h1), it(h2), it(h3)},
                                       {it(rd)})
                           })
              )
        )
);

}
