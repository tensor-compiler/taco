#include "test.h"

#include "taco/type.h"
#include "taco/index_notation/iteration_algebra.h"

using namespace taco;

const TensorVar A("A", Type()), B("B", Type()), C("C", Type());

TEST(iteration_algebra, iter_alg_print) {
  std::ostringstream ss;
  ss << Intersect(Union(Complement(A), B), C);
  std::string expected("(~A U B) * C");
  ASSERT_EQ(expected, ss.str());
}