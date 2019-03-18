#include <taco/index_notation/transformations.h>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;
const IndexVar i("i"), j("j"), k("k");

TEST(scheduling, splitEquality) {
  IndexVar i1, i2;
  IndexVar j1, j2;
  SplitRel rel1 = SplitRel(i, i1, i2, 2);
  SplitRel rel2 = SplitRel(i, i1, i2, 2);
  SplitRel rel3 = SplitRel(j, i1, i1, 2);
  SplitRel rel4 = SplitRel(i, i1, i2, 4);
  SplitRel rel5 = SplitRel(i, j1, j2, 2);

  ASSERT_EQ(rel1, rel2);
  ASSERT_NE(rel1, rel3);
  ASSERT_NE(rel1, rel4);
  ASSERT_NE(rel1, rel5);
}

TEST(scheduling, splitIndexVar) {
  IndexVar i1, i2;
  i.split(i1, i2, 2);

  ASSERT_TRUE(i1.isIrregular());
  ASSERT_FALSE(i2.isIrregular());

  ASSERT_TRUE(i1.getDerivation().getRelType() == IndexVarRel::SPLIT);
  ASSERT_TRUE(i2.getDerivation().getRelType() == IndexVarRel::SPLIT);

  SplitRel rel1 = i1.getDerivation<SplitRel>();
  SplitRel rel2 = i2.getDerivation<SplitRel>();
  ASSERT_EQ(rel1, rel2);

  ASSERT_EQ(rel1.getParentVars(), std::vector<IndexVar>({i}));
  ASSERT_EQ(rel1.outerVar, i1);
  ASSERT_EQ(rel1.innerVar, i2);
  ASSERT_EQ(rel1.splitFactor, (size_t) 2);
}

TEST(scheduling, forallReplace) {
  IndexVar i1, j1, j2;
  Type t(type<double>(), {3});
  TensorVar a("a", t), b("b", t);
  IndexStmt stmt = forall(i, forall(i1, a(i) = b(i)));
  IndexStmt replaced = Transformation(ForAllReplace({i, i1}, {j, j1, j2})).apply(stmt);
  ASSERT_NE(stmt, replaced);

  ASSERT_TRUE(isa<Forall>(replaced));
  Forall jForall = to<Forall>(replaced);
  ASSERT_EQ(j, jForall.getIndexVar());

  ASSERT_TRUE(isa<Forall>(jForall.getStmt()));
  Forall j1Forall = to<Forall>(jForall.getStmt());
  ASSERT_EQ(j1, j1Forall.getIndexVar());

  ASSERT_TRUE(isa<Forall>(j1Forall.getStmt()));
  Forall j2Forall = to<Forall>(j1Forall.getStmt());
  ASSERT_EQ(j2, j2Forall.getIndexVar());

  ASSERT_TRUE(equals(a(i) = b(i), j2Forall.getStmt()));
  ASSERT_TRUE(equals(forall(j, forall(j1, forall(j2, a(i) = b(i)))), replaced));
}

TEST(scheduling, splitIndexStmt) {
  Type t(type<double>(), {3});
  TensorVar a("a", t), b("b", t);
  IndexVar i1, i2;
  IndexStmt stmt = forall(i, a(i) = b(i));
  IndexStmt splitStmt = stmt.split(i, i1, i2, 2);

  ASSERT_TRUE(isa<Forall>(splitStmt));
  Forall i1Forall = to<Forall>(splitStmt);
  ASSERT_EQ(i1, i1Forall.getIndexVar());

  ASSERT_TRUE(isa<Forall>(i1Forall.getStmt()));
  Forall i2Forall = to<Forall>(i1Forall.getStmt());
  ASSERT_EQ(i2, i2Forall.getIndexVar());

  ASSERT_TRUE(equals(a(i) = b(i), i2Forall.getStmt()));
  ASSERT_TRUE(equals(forall(i1, forall(i2, a(i) = b(i))), splitStmt));

  ASSERT_TRUE(i1.isIrregular());
  ASSERT_FALSE(i2.isIrregular());
}


