#include "test.h"

#include "taco/tensor.h"

using namespace taco;

template <typename T>
  int compare(const Tensor<T>&tensorA, const Tensor<T>&tensorB,
              bool withPrint, T precision) {
    if (tensorA.getDimensions() != tensorB.getDimensions()) {
      return 1;
    }
      std::set<typename Tensor<T>::Coordinate> coordsA;
      for (const auto& val : tensorA) {
        if (!coordsA.insert(val.first).second) {
          return 2;
        }
      }
      std::set<typename Tensor<T>::Coordinate> coordsB;
      for (const auto& val : tensorB) {
        if (!coordsB.insert(val.first).second) {
          return 3;
        }
      }
    if (coordsA!=coordsB)
        return 4;

    typedef std::set<typename Tensor<T>::Value> Values;
    Values valsA;
    for (const auto& val : tensorA) {
      if (val.second != 0) {
        valsA.insert(val);
      }
    }
    Values valsB;
    for (const auto& val : tensorB) {
      if (val.second != 0) {
        valsB.insert(val);
      }
    }

    typedef typename Values::iterator itValues;
    if (valsA == valsB)
      return 0;
    else
      for (std::pair<itValues,itValues> val(valsA.begin(), valsB.begin());
          val.first != valsA.end();++val.first, ++val.second) {
        if (abs((*(val.first)).second - (*(val.second)).second) > precision) {
          if (withPrint) {
            std::cout << (*(val.first)).second << " " << (*(val.second)).second << std::endl;
          }
          return 5;
        }
      }
    return 0;
  }

TEST(regression, issue46) {
  Format DD({Dense,Dense});
  Format DSDD({Dense,Sparse,Dense,Dense});

  Tensor<double> A({14,14,3,3},DSDD);
  Tensor<double> x({14,3},DD);
  Tensor<double> y_produced({14,3},DD);
  Tensor<double> y_expected({14,3},DD);

  A.read("../test/data/fidapm05.mtx");
  x.read("../test/data/x_issue46.mtx");
  y_expected.read("../test/data/y_expected46.mtx");

  // Blocked-SpMV
  Var i, j(Var::Sum), ib, jb(Var::Sum);
  y_produced(i,ib) = A(i,j,ib,jb) * x(j,jb);

  // Compile the expression
  y_produced.compile();

  // Assemble A's indices and numerically compute the result
  y_produced.assemble();
  y_produced.compute();

  ASSERT_FALSE(compare(y_produced,y_expected,true,10e-6));
}
