#include "test.h"

#include "taco/tensor.h"

using namespace taco;

TEST(regression, issue46) {
  Format DD({Dense,Dense});
  Format DSDD({Dense,Sparse,Dense,Dense});

  Tensor<double> A({14,14,3,3},DSDD);
  Tensor<double> x({14,3},DD);
  Tensor<double> y_produced({14,3},DD);
  Tensor<double> y_expected({14,3},DD);

  A.read(testDirectory() + "/data/fidapm05.mtx");
  x.read(testDirectory() + "/data/x_issue46.mtx");
  y_expected.read(testDirectory() + "/data/y_expected46.mtx");
  A.pack();
  x.pack();
  y_expected.pack();

  // Blocked-SpMV
  Var i, j(Var::Sum), ib, jb(Var::Sum);
  y_produced(i,ib) = A(i,j,ib,jb) * x(j,jb);

  // Compile the expression
  y_produced.compile();

  // Assemble A's indices and numerically compute the result
  y_produced.assemble();
  y_produced.compute();
  y_produced.zero();
  y_produced.compute();

  ASSERT_TENSOR_EQ(y_produced,y_expected);
}
