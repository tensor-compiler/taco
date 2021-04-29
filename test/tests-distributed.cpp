#include "test.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/distribution.h"
#include "taco/lower/lower.h"

using namespace taco;

TEST(distributed, test) {
  int dim = 10;
  Tensor<int> a("a", {dim}, Format{Dense});
  Tensor<int> b("b", {dim}, Format{Dense});

  IndexVar i("i"), in("in"), il("il"), il1 ("il1"), il2("il2");
  a(i) = b(i);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.distribute({i}, {in}, {il}, Grid(2));
  stmt = stmt.split(il, il1, il2, 256);


  // Communication modification must go at the end.
  stmt = stmt.pushCommUnder(a(i), in).pushCommUnder(b(i), il1);
//  stmt = stmt.pushCommUnder(a(i), il1).pushCommUnder(b(i), il1);
//  stmt = stmt.pushCommUnder(a(i), in).pushCommUnder(b(i), in);

  auto lowered = lower(stmt, "compute", false, true);
  std::cout << lowered << std::endl;
}