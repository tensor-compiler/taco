#include "test.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/iteration_algebra.h"

using namespace taco;

const IndexVar i("i"), j("j"), k("k");

Type vec(type<double>(), {3});
TensorVar v1("v1", vec), v2("v2", vec), v3("v3", vec);

TEST(iteration_algebra, region) {
  Access access = v1(i);
  IterationAlgebra alg = access;
  ASSERT_TRUE(isa<Region>(alg));
  ASSERT_TRUE(isa<RegionNode>(alg.ptr));
  Region region = to<Region>(alg);
  const RegionNode* node = to<RegionNode>(alg.ptr);
  ASSERT_FALSE(isa<Complement>(alg));
  ASSERT_EQ(access, node->expr());
}

TEST(iteration_algebra, Complement) {
  IndexExpr expr = v1(i) + v2(i);
  IterationAlgebra alg = Complement(expr);
  ASSERT_TRUE(isa<Complement>(alg));
  ASSERT_TRUE(isa<ComplementNode>(alg.ptr));
  Complement complement = to<Complement>(alg);
  const ComplementNode* n = to<ComplementNode>(alg.ptr);
  ASSERT_FALSE(isa<RegionNode>(n));
  ASSERT_TRUE(algEqual(expr, n->a));

  ASSERT_TRUE(isa<RegionNode>(n->a.ptr));
  const RegionNode* r = to<RegionNode>(n->a.ptr);
  ASSERT_EQ(expr, r->expr());
}

TEST(iteration_algebra, Union) {
  IndexExpr exprA = v1(i);
  IndexExpr exprB = v2(i);
  IterationAlgebra alg = Union(exprA, exprB);
  ASSERT_TRUE(isa<Union>(alg));
  ASSERT_TRUE(isa<UnionNode>(alg.ptr));
  Union u = to<Union>(alg);
  const UnionNode* n = to<UnionNode>(alg.ptr);
  ASSERT_FALSE(isa<IntersectNode>(n));
  ASSERT_TRUE(algEqual(exprA, n->a));
  ASSERT_TRUE(algEqual(exprB, n->b));

  ASSERT_TRUE(isa<RegionNode>(n->a.ptr));
  const RegionNode* r1 = to<RegionNode>(n->a.ptr);
  ASSERT_TRUE(isa<RegionNode>(n->b.ptr));
  const RegionNode* r2 = to<RegionNode>(n->b.ptr);

  ASSERT_EQ(exprA, r1->expr());
  ASSERT_EQ(exprB, r2->expr());
}

TEST(iteration_algebra, Intersect) {
  IndexExpr exprA = v1(i);
  IndexExpr exprB = v2(i);
  IterationAlgebra alg = Intersect(exprA, exprB);
  ASSERT_TRUE(isa<Intersect>(alg));
  ASSERT_TRUE(isa<IntersectNode>(alg.ptr));
  Intersect i = to<Intersect>(alg);
  const IntersectNode* n = to<IntersectNode>(alg.ptr);
  ASSERT_FALSE(isa<UnionNode>(n));
  ASSERT_TRUE(algEqual(exprA, n->a));
  ASSERT_TRUE(algEqual(exprB, n->b));

  ASSERT_TRUE(isa<RegionNode>(n->a.ptr));
  const RegionNode* r1 = to<RegionNode>(n->a.ptr);
  ASSERT_TRUE(isa<RegionNode>(n->b.ptr));
  const RegionNode* r2 = to<RegionNode>(n->b.ptr);

  ASSERT_EQ(exprA, r1->expr());
  ASSERT_EQ(exprB, r2->expr());
}

TEST(iteration_algebra, comparatorRegion) {
  IterationAlgebra alg1(v1(i));
  IterationAlgebra alg2(v2(j));
  ASSERT_TRUE(algStructureEqual(alg1, alg2));
  ASSERT_FALSE(algEqual(alg1, alg2));

  ASSERT_TRUE(algEqual(alg1, alg1));
  ASSERT_TRUE(algStructureEqual(alg1, alg1));
}

TEST(iteration_algebra, comparatorComplement) {
  IterationAlgebra alg1 = Complement(v2(i));
  IterationAlgebra alg2 = Complement(v3(j));
//  ASSERT_TRUE(algStructureEqual(alg1, alg2));
  ASSERT_FALSE(algEqual(alg1, alg2));

  ASSERT_TRUE(algStructureEqual(alg1, alg1));
  ASSERT_TRUE(algEqual(alg1, alg1));
}

TEST(iteration_algebra, comparatorIntersect) {
  IterationAlgebra alg1 = Intersect(v1(i), v2(i));
  IterationAlgebra alg2 = Intersect(v1(j), v3(j));
  ASSERT_TRUE(algStructureEqual(alg1, alg2));
  ASSERT_FALSE(algEqual(alg1, alg2));

  ASSERT_TRUE(algStructureEqual(alg1, alg1));
  ASSERT_TRUE(algEqual(alg1, alg1));
}

TEST(iteration_algebra, comparatorUnion) {
  IterationAlgebra alg1 = Union(v1(i), v2(i));
  IterationAlgebra alg2 = Union(v1(j), v3(j));
  ASSERT_TRUE(algStructureEqual(alg1, alg2));
  ASSERT_FALSE(algEqual(alg1, alg2));

  ASSERT_TRUE(algStructureEqual(alg1, alg1));
  ASSERT_TRUE(algEqual(alg1, alg1));
}

TEST(iteration_algebra, comparatorMix) {
  IterationAlgebra alg1 = Union(Intersect(v1(i), v2(i)), Complement(v3(i)));
  IterationAlgebra alg2 = Union(Intersect(v1(j), v2(j)), Complement(v3(j)));
  ASSERT_TRUE(algStructureEqual(alg1, alg2));
  ASSERT_FALSE(algEqual(alg1, alg2));

  ASSERT_TRUE(algStructureEqual(alg1, alg1));
  ASSERT_TRUE(algEqual(alg1, alg1));
}

TEST(iteration_algebra, deMorganRegion) {
  IterationAlgebra alg(v1(i));
  IterationAlgebra simplified = applyDemorgan(alg);

  ASSERT_TRUE(algEqual(alg, simplified));
  ASSERT_TRUE(algStructureEqual(alg, simplified));
}

TEST(iteration_algebra, deMorganComplement) {
  IterationAlgebra alg = Complement(v1(i));
  IterationAlgebra simplified = applyDemorgan(alg);

  ASSERT_TRUE(algEqual(alg, simplified));
  ASSERT_TRUE(algStructureEqual(alg, simplified));
}

TEST(iteration_algebra, DeMorganNestedComplements) {
  IterationAlgebra alg = v1(i);
  for(int cnt = 0; cnt < 10; ++cnt) {
    if(cnt % 2 == 0) {
      IterationAlgebra simplified = applyDemorgan(alg);
      IterationAlgebra expectedEven = v1(i);
      ASSERT_TRUE(algEqual(expectedEven, simplified));
    }
    else {
      IterationAlgebra simplified = applyDemorgan(alg);
      IterationAlgebra expectedOdd = Complement(v1(i));
      ASSERT_TRUE(algEqual(expectedOdd, simplified));
    }
    alg = Complement(alg);
  }
}

TEST(iteration_algebra, deMorganIntersect) {
  IterationAlgebra alg = Intersect(v1(i), v2(i));
  IterationAlgebra simplified = applyDemorgan(alg);

  ASSERT_TRUE(algEqual(alg, simplified));
  ASSERT_TRUE(algStructureEqual(alg, simplified));
}

TEST(iteration_algebra, deMorganUnion) {
  IterationAlgebra alg = Union(v1(i), v2(i));
  IterationAlgebra simplified = applyDemorgan(alg);

  ASSERT_TRUE(algEqual(alg, simplified));
  ASSERT_TRUE(algStructureEqual(alg, simplified));
}

TEST(iteration_algebra, UnionComplement) {
  IterationAlgebra alg = Union(v1(i), Complement(v2(i)));
  IterationAlgebra simplified = applyDemorgan(alg);

  ASSERT_TRUE(algEqual(alg, simplified));
  ASSERT_TRUE(algStructureEqual(alg, simplified));
}

TEST(iteration_algebra, flipUnionToIntersect) {
  IterationAlgebra alg = Complement(Union(v1(i), v2(i)));
  IterationAlgebra simplified = applyDemorgan(alg);

  ASSERT_FALSE(algEqual(alg, simplified));
  ASSERT_FALSE(algStructureEqual(alg, simplified));

  IterationAlgebra expected = Intersect(Complement(v1(i)), Complement(v2(i)));
  ASSERT_TRUE(algEqual(simplified, expected));
  ASSERT_TRUE(algStructureEqual(simplified, expected));
}

TEST(iteration_algebra, flipIntersectToUnion) {
  IterationAlgebra alg = Complement(Intersect(v1(i), v2(i)));
  IterationAlgebra simplified = applyDemorgan(alg);

  ASSERT_FALSE(algEqual(alg, simplified));
  ASSERT_FALSE(algStructureEqual(alg, simplified));

  IterationAlgebra expected = Union(Complement(v1(i)), Complement(v2(i)));
  ASSERT_TRUE(algEqual(simplified, expected));
  ASSERT_TRUE(algStructureEqual(simplified, expected));
}

TEST(iteration_algebra, hiddenIntersect) {
  IterationAlgebra alg = Complement(Union(Complement(v1(i)), Complement(v2(i))));
  IterationAlgebra simplified = applyDemorgan(alg);

  IterationAlgebra expected = Intersect(v1(i), v2(i));
  ASSERT_TRUE(algEqual(expected, simplified));
}

TEST(iteration_algebra, hiddenUnion) {
  IterationAlgebra alg = Complement(Intersect(Complement(v1(i)), Complement(v2(i))));
  IterationAlgebra simplified = applyDemorgan(alg);

  IterationAlgebra expected = Union(v1(i), v2(i));
  ASSERT_TRUE(algEqual(expected, simplified));
}