#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;
const IndexVar i("i"), j("j"), k("k");

TEST(concrete, forall) {
  Type t(type<double>(), {3});
  TensorVar a("a", t), b("b", t), c("c", t);

  Forall foralli = forall(i, a(i) = b(i) + c(i));
//  std::cout << foralli << std::endl;
}

TEST(concrete, where) {
  Type t(type<double>(), {3});
  TensorVar a("a", t, Sparse), b("b", t, Sparse), c("c", t, Sparse);
  TensorVar w("w", t, Dense);

  Where vecmul = where(forall(i, a(i)=w(i)*c(i)), forall(i, w(i)=b(i)));
//  std::cout << vecmul << std::endl;
}

TEST(concrete, spmm) {
  Type t(type<double>(), {3,3});
  TensorVar A("A", t, Sparse), B("B", t, Sparse), C("C", t, Sparse);
  TensorVar w("w", Type(type<double>(),{3}), Dense);  auto spmm = forall(i,
                     forall(k,
                            where(forall(j, A(i,j) = w(j)),
                                  forall(j,   w(j) += B(i,k)*C(k,j))
                                  )
                            )
                     );
//  std::cout << spmm << std::endl;
}

