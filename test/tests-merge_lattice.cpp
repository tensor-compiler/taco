#include "test.h"

#include <vector>
#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/lower/merge_lattice.h"
#include "taco/lower/iterator.h"
#include "lower/mode_access.h"
#include "taco/ir/ir.h"

using namespace std;

using namespace taco;

// Temporary hack until dense in format.h is transition from the old system
#include "taco/lower/mode_format_dense.h"
taco::ModeFormat dense_new(std::make_shared<taco::DenseModeFormat>());
#define dense dense_new

static const Dimension n;
static const Type vectype(Float64, {n});

static TensorVar r1t("r1", vectype, Format(dense));
static TensorVar r2t("r2", vectype, Format(sparse));

static TensorVar d1t("d1", vectype, Format(dense));
static TensorVar d2t("d2", vectype, Format(dense));
static TensorVar d3t("d3", vectype, Format(dense));
static TensorVar d4t("d4", vectype, Format(dense));
static TensorVar s1t("s1", vectype, Format(sparse));
static TensorVar s2t("s2", vectype, Format(sparse));
static TensorVar s3t("s3", vectype, Format(sparse));
static TensorVar s4t("s4", vectype, Format(sparse));

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

static map<Iterator, IndexVar> indexVars;
static map<IndexVar, taco::ir::Expr> coordVars;
static Forall dummy = forall(i, rd = rs + d1 + d2 + d3 + d4 + s1 + s2 + s3 + s4);
static map<ModeAccess, Iterator> iterators = createIterators(dummy, tensorVars,
                                                             &indexVars,
                                                             &coordVars);

static vector<Iterator> iter() {
  return {};
}

static vector<Iterator> iter(vector<Iterator> iterators) {
  return iterators;
}

static vector<Iterator> iter(vector<Access> accesses) {
  vector<Iterator> result;
  for (auto& access : accesses) {
    taco_iassert(util::contains(iterators, ModeAccess(access, 1)))
    << "Could not find " << ModeAccess(access, 1);
    result.push_back(iterators.at(ModeAccess(access, 1)));
  }
  return result;
}

namespace tests {

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

INSTANTIATE_TEST_CASE_P(vector_neg, merge_lattice,
  Values(Test(forall(i, rd = -d2),
              MergeLattice({MergePoint(iter({i}),
                                       iter({d2}),
                                       iter({rd}))
                           })
              ),
         Test(forall(i, rs = -s1),
              MergeLattice({MergePoint(iter({s1}),
                                       iter(),
                                       iter({rs}))
                           })
              )
         )
);

INSTANTIATE_TEST_CASE_P(vector_mul, merge_lattice,
  Values(Test(forall(i, rd = d1 * d2),
              MergeLattice({MergePoint(iter({i}),
                                       iter({d1,d2}),
                                       iter({rd}))
                           })
              ),
         Test(forall(i, rd = s1 * s2),
              MergeLattice({MergePoint(iter({s1, s2}),
                                       iter(),
                                       iter({rd}))
                           })
              ),
         Test(forall(i, rd = s1 * d1),
              MergeLattice({MergePoint(iter({s1}),
                                       iter({d1}),
                                       iter({rd}))
                           })
              )
        )
);

//INSTANTIATE_TEST_CASE_P(vector_add, merge_lattice,
//  Values(Test(forall(i, rd = d1 + d2),
//              MergeLattice({MergePoint(iter({i}),
//                                       iter({d1, d2}),
//                                       iter({rd})),
//                           })
//              ),
//         Test(forall(i, rd = s1 + s2),
//              MergeLattice({MergePoint(iter({s1, s2}),
//                                       iter({s1, s2}),
//                                       iter({rd}))
//                           })
//              ),
//         Test(forall(i, rs = s1 + d1),
//              MergeLattice({MergePoint(iter({s1, d1}),
//                                       iter({s1}),
//                                       iter({rs}))
//                           })
//              )
//        )
//);

}
