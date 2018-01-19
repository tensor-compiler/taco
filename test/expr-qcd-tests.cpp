/// This file tests expressions from quantum chromodynamics (QCD).
#include "test.h"
#include "taco/tensor.h"

using namespace std;
using namespace taco;

static string qcdTestData(string name) {
  return testDataDirectory() + "qcd/" + name;
}

double getScalarValue(Tensor<double> tensor) {
  return ((double*)tensor.getStorage().getValues().getData())[0];
}

TEST(expr, qcd0) {
  Tensor<double> tau("tau");
  Tensor<double> z = read(qcdTestData("z0.ttx"), Dense);
  Tensor<double> theta = read(qcdTestData("theta0.ttx"), Dense);

  IndexVar i0, i1;
  tau = z(i0) * z(i1) * theta(i0,i1);

  tau.evaluate();
  ASSERT_DOUBLE_EQ(0.5274167972947047, getScalarValue(tau));
}

TEST(expr, qcd1) {
  Tensor<double> tau("tau");
  Tensor<double> z = read(qcdTestData("z0.ttx"), Dense);
  Tensor<double> theta = read(qcdTestData("theta0.ttx"), Dense);

  IndexVar i0, i1;
  tau = z(i0) * z(i1) * theta(i0,i1) * theta(i0,i1);

  tau.evaluate();
  ASSERT_DOUBLE_EQ(0.41212798763234737, getScalarValue(tau));
}

TEST(expr, qcd2) {
  Tensor<double> tau("tau");
  Tensor<double> z = read(qcdTestData("z0.ttx"), Dense);
  Tensor<double> theta = read(qcdTestData("theta0.ttx"), Dense);

  IndexVar i0, i1;
  tau = z(i0) * z(i1) * theta(i0,i1) * theta(i0,i1) * theta(i0,i1);

  tau.evaluate();
  ASSERT_DOUBLE_EQ(0.4120590379120669, getScalarValue(tau));
}

//TEST(expr, qcd3) {
//  Tensor<double> tau("tau");
//  Tensor<double> z = read(qcdTestData("z0.ttx"), Dense);
//  Tensor<double> theta = read(qcdTestData("theta0.ttx"), Dense);
//
//  IndexVar i0, i1, i2;
//  tau = z(i0) * z(i1) * z(i2) * theta(i0,i1) * theta(i0,i2);
//
//  tau.evaluate();
//  ASSERT_DOUBLE_EQ(0.3223971010027145, getScalarValue(tau));
//}

