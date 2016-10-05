#include "ir.h"
#include "test.h"
#include "backend_c.h"
#include <regex>

string normalize(string str) {
  std::regex postfix("(_\\d+)");
  return std::regex_replace(str, postfix, "$$");
}

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
  
  EXPECT_EQ(expected, normalize(foo.str()));
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
  
  EXPECT_EQ(expected, normalize(foo.str()));
}

TEST_F(BackendCTests, GenStore) {
  auto var = Var::make("x", typeOf<int>());
  auto fn = Function::make("foobar", {Var::make("y", typeOf<double>())},
    {var},
    Block::make({Store::make(var, Literal::make(0), Literal::make(101))}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(fn.as<Function>());
  
  string expected = "int foobar(double* y, int* x) {\n"
                    "  x[0] = 101;\n"
                    "  return 0;\n"
                    "}\n";
  
  EXPECT_EQ(expected, normalize(foo.str()));
}

TEST_F(BackendCTests, GenVarAssign) {
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())},
    {Var::make("y", typeOf<double>())},
    Block::make({VarAssign::make(Var::make("z", typeOf<int>(), false), Literal::make(12))}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  
  string expected = "int foobar(int* x, double* y) {\n"
                    "  int _z$;\n"
                    "  _z$ = 12;\n"
                    "  return 0;\n"
                    "}\n";
  
  EXPECT_EQ(expected, normalize(foo.str()));
}

TEST_F(BackendCTests, GenFor) {
  auto var = Var::make("i", typeOf<int>(), false);
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())},
    {Var::make("y", typeOf<double>())},
    Block::make({For::make(var, Literal::make(0), Literal::make(10), Literal::make(1),
      Block::make({VarAssign::make(Var::make("z", typeOf<int>(), false), var)}))}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  string expected = "int foobar(int* x, double* y) {\n"
                    "  int _i$;\n"
                    "  int _z$;\n"
                    "  for (_i$=0; _i$<10; _i$+=1)\n"
                    "  {\n"
                    "    _z$ = _i$;\n"
                    "  }\n"
                    "  return 0;\n"
                    "}\n";
  
  EXPECT_EQ(expected, normalize(foo.str()));
}

TEST_F(BackendCTests, BuildModule) {
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())}, {}, Block::make({}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());

  Module mod(foo.str());
  mod.compile();
  
  typedef int (*fnptr_t)(int);
  
  fnptr_t func = (fnptr_t)mod.get_func("foobar");
  EXPECT_EQ(0, func(4));
}

TEST_F(BackendCTests, BuildModuleWithStore) {
  auto var = Var::make("x", typeOf<int>());
  auto fn = Function::make("foobar", {Var::make("y", typeOf<double>())},
    {var},
    Block::make({Store::make(var, Literal::make(0), Literal::make(101))}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(fn.as<Function>());

  Module mod(foo.str());
  mod.compile();
  
  typedef int (*fnptr_t)(double*,int*);
  
  fnptr_t func = (fnptr_t)mod.get_func("foobar");
  int x = 22;
  double y = 1.8;
  func(&y, &x);
  EXPECT_EQ(101, x);
}




