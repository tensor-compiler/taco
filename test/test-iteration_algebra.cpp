#include "test.h"

#include "taco/type.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/iteration_algebra.h"

using namespace taco;

const TensorVar A_var("A", Type()), B_var("B", Type()), C_var("C", Type());
const Access A(A_var), B(B_var), C(C_var);

TEST(iteration_algebra, iter_alg_print) {
  std::ostringstream ss;
  ss << Intersect(Union(Complement(A), B), C);
  std::string expected("(~A U B) * C");
  ASSERT_EQ(expected, ss.str());
}