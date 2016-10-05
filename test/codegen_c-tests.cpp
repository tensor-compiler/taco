#include "ir.h"
#include "test.h"
#include "backend_c.h"

using namespace ::testing;
using namespace tacit::internal;

struct BackendCTests : public Test {

};


TEST_F(BackendCTests, GenEmptyFunction) {
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())}, {}, Block::make({}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  
  string expected = "int foobar(int* x) {\n"
                    "  return 0;\n"
                    "}\n";
  
  EXPECT_EQ(expected, foo.str());
}

TEST_F(BackendCTests, GenEmptyFunctionWithOutput) {
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())},
    {Var::make("y", typeOf<double>())},
    Block::make({}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  
  string expected = "int foobar(int* x, double* y) {\n"
                    "  return 0;\n"
                    "}\n";
  
  EXPECT_EQ(expected, foo.str());
}

TEST_F(BackendCTests, GenVarAssign) {
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())},
    {Var::make("y", typeOf<double>())},
    Block::make({VarAssign::make(Var::make("z", typeOf<int>(), false), Literal::make(12))}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  
  string expected = "int foobar(int* x, double* y) {\n"
                    "  int _z_0;\n"
                    "  _z_0 = 12;\n"
                    "  return 0;\n"
                    "}\n";
  
  EXPECT_EQ(expected, foo.str());
}
