#include <regex>

#include "ir.h"
#include "test.h"
#include "format.h"
#include "tensor.h"
#include "backends/backend_c.h"

string normalize(string str) {
  std::regex postfix("(_\\d+)");
  std::string foo("$$");
  return std::regex_replace(str, postfix, foo);
}

using namespace ::testing;
using namespace taco::ir;

struct BackendCTests : public Test {

};


TEST_F(BackendCTests, GenEmptyFunction) {
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())}, {}, Block::make({}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  
  string expected = "int foobar(void** inputPack) {\n"
                    "  int* x = (int*)inputPack[0];\n\n"
                    "  return 0;\n"
                    "}\n";
  
  EXPECT_EQ(expected, normalize(foo.str()));
}

TEST_F(BackendCTests, GenMin) {
  auto body = VarAssign::make(Var::make("foo", typeOf<int>()),
    Min::make({Literal::make(3), Literal::make(4), Literal::make(10)}));
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())}, {},
    Block::make({body}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  
  string expected = "int foobar(void** inputPack) {\n"
                    "  int* x = (int*)inputPack[0];\n"
                    "  int* _foo$;\n  _foo$ = MIN(3,MIN(4,10));\n\n"
                    "  return 0;\n"
                    "}\n";
  
  EXPECT_EQ(expected, normalize(foo.str()));
}


TEST_F(BackendCTests, GenPrint) {
  auto x = Var::make("x", typeOf<int>());
  auto y = Var::make("y", typeOf<int>(), false);
  auto print = Print::make("blah: %d %l", {x, y});
  auto add = Function::make("foobar", {x, y}, {}, Block::make({print}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  
  string expected = "int foobar(void** inputPack) {\n"
                    "  int* x = (int*)inputPack[0];\n"
                    "  int y = *(int*)inputPack[1];\n"
                    "  printf(\"blah: %d %l\", x, y);\n\n"
                    "  return 0;\n"
                    "}\n";
  
  EXPECT_EQ(expected, normalize(foo.str()));
}


TEST_F(BackendCTests, GenCommentAndBlankLine) {
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())}, {},
    Block::make({BlankLine::make(), Comment::make("comment")}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  
  string expected = "int foobar(void** inputPack) {\n"
                    "  int* x = (int*)inputPack[0];\n"
                    "\n"
                    "  // comment\n\n"
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
  
  string expected = "int foobar(void** inputPack) {\n"
                    "  double* y = (double*)inputPack[0];\n"
                    "  int* x = (int*)inputPack[1];\n\n"
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
  
  string expected = "int foobar(void** inputPack) {\n"
                    "  int* x = (int*)inputPack[0];\n"
                    "  double* y = (double*)inputPack[1];\n"
                    "  x[0] = 101;\n\n"
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
  
  string expected = "int foobar(void** inputPack) {\n"
                    "  double* y = (double*)inputPack[0];\n"
                    "  int* x = (int*)inputPack[1];\n"
                    "  int _z$;\n"
                    "  _z$ = 12;\n\n"
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
  // Depending on ordering of decls, can be either of these
  string expected1 = "int foobar(void** inputPack) {\n"
                    "  double* y = (double*)inputPack[0];\n"
                    "  int* x = (int*)inputPack[1];\n"
                    "  int _z$;\n"
                    "  int _i$;\n"
                    "  for (_i$=0; _i$<10; _i$+=1)\n"
                    "  {\n"
                    "    _z$ = _i$;\n"
                    "  }\n\n"
                    "  return 0;\n"
                    "}\n";
  
  string expected2 = "int foobar(void** inputPack) {\n"
                    "  double* y = (double*)inputPack[0];\n"
                    "  int* x = (int*)inputPack[1];\n"
                    "  int _i$;\n"
                    "  int _z$;\n"
                    "  for (_i$=0; _i$<10; _i$+=1)\n"
                    "  {\n"
                    "    _z$ = _i$;\n"
                    "  }\n\n"
                    "  return 0;\n"
                    "}\n";

  auto normalized = normalize(foo.str());
  EXPECT_TRUE(normalized == expected1 || normalized == expected2)
    << "Result: " << normalized << "is not one of: \n"
    << expected1 << "or\n"
    << expected2;
}

TEST_F(BackendCTests, GenCase) {
  auto cond = Eq::make(Literal::make(3), Literal::make(4));
  auto cond2 = Eq::make(Literal::make(4), Literal::make(5));
  auto cs = Case::make({{cond,Block::make({})}, {cond2,Block::make({})}});
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())}, {},
    Block::make({cs}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  
  string expected =
                    "int foobar(void** inputPack) {\n"
                    "  int* x = (int*)inputPack[0];\n"
                    "  if (3 == 4)\n"
                    "  {\n"
                    "  }\n"
                    "  else if (4 == 5)\n"
                    "  {\n"
                    "  }\n"
                    "\n\n"
                    "  return 0;\n"
                    "}\n";
  
  EXPECT_EQ(expected, normalize(foo.str()));
}


TEST_F(BackendCTests, GenWhile) {
  auto var = Var::make("i", typeOf<int>(), false);
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())},
    {Var::make("y", typeOf<double>())},
    Block::make({While::make(Lt::make(var, Literal::make(10)),
      Block::make({VarAssign::make(var, Literal::make(11))}))}));
  
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());
  string expected = "int foobar(void** inputPack) {\n"
                    "  double* y = (double*)inputPack[0];\n"
                    "  int* x = (int*)inputPack[1];\n"
                    "  int _i$;\n"
                    "  while ((_i$ < 10))\n"
                    "  {\n"
                    "    _i$ = 11;\n"
                    "  }\n\n"
                    "  return 0;\n"
                    "}\n";
  EXPECT_EQ(expected, normalize(foo.str()));
}

//TEST_F(BackendCTests, GenTensorUnpack) {
//  taco::Format csr({taco::LevelType::Dense, taco::LevelType::Sparse});
//  auto tensor = Var::make("A", typeOf<float>(), csr);
//  auto unpack = GetProperty::make(tensor, TensorProperty::Index, 1);
//  auto ptr_to_idx = Var::make("p", typeOf<int>());
//  auto unpack2 = GetProperty::make(tensor, TensorProperty::Index, 1);
//  auto ptr_to_idx2 = Var::make("p2", typeOf<int>());
//
//  auto add = Function::make("foobar", {tensor}, {},
//    Block::make({VarAssign::make(ptr_to_idx, unpack),
//                 VarAssign::make(ptr_to_idx2, unpack2)}));
//  stringstream foo;
//  CodeGen_C cg(foo);
//  cg.compile(add.as<Function>());
////  cout << add << "\n";
////  cout << foo.str();
////  
//  string expected1 = normalize(
//                  "int foobar(void** inputPack) {\n"
//                  "  void** A = &(inputPack[0]);\n"
//                  "  int* ___A__L1_idx_5 = (int*)A[2];\n"
//                  "  int* _p_4;\n"
//                  "  int* _p2_6;\n"
//                  "  _p_4 = ___A__L1_idx_5;\n"
//                  "  _p2_6 = ___A__L1_idx_5;\n\n"
//                  "  return 0;\n"
//                  "}\n");
//  string expected2 = normalize(
//                  "int foobar(void** inputPack) {\n"
//                  "  void** A = &(inputPack[0]);\n"
//                  "  int* ___A__L1_idx_5 = (int*)A[2];\n"
//                  "  int* _p2_6;\n"
//                  "  int* _p_4;\n"
//                  "  _p_4 = ___A__L1_idx_5;\n"
//                  "  _p2_6 = ___A__L1_idx_5;\n\n"
//                  "  return 0;\n"
//                  "}\n");
//
//  
//  auto normalized = normalize(foo.str());
//  EXPECT_TRUE(normalized == expected1 || normalized == expected2)
//    << "Result: " << normalized << "is not one of: \n"
//    << expected1 << "or\n"
//    << expected2;
//
//}
//
//TEST_F(BackendCTests, GenTensorRepack) {
//  taco::Format csr({taco::LevelType::Dense, taco::LevelType::Sparse});
//  auto tensor = Var::make("A", typeOf<float>(), csr);
//  auto output_tensor = Var::make("Out", typeOf<float>(), csr);
//  auto unpack = GetProperty::make(tensor, TensorProperty::Index, 1);
//  auto ptr_to_idx = Var::make("p", typeOf<int>());
//  auto unpack2 = GetProperty::make(output_tensor, TensorProperty::Index, 1);
//  auto ptr_to_idx2 = Var::make("p2", typeOf<int>());
//  auto unpack3 = GetProperty::make(output_tensor, TensorProperty::Pointer, 0);
//
//  auto add = Function::make("foobar", {tensor}, {output_tensor},
//    Block::make({VarAssign::make(ptr_to_idx, unpack),
//                 VarAssign::make(ptr_to_idx2, unpack2),
//                 VarAssign::make(unpack3, Literal::make(4))}));
//  stringstream foo;
//  CodeGen_C cg(foo);
//  cg.compile(add.as<Function>());
//  
//  string expected = normalize(
//                  "int foobar(void** inputPack) {\n"
//                  "  void** Out = &(inputPack[0]);\n"
//                  "  void** A = &(inputPack[4]);\n"
//                  "  int* restrict ___A__L1_idx_9 = (int*)A[2];\n"
//                  "  int* _p_8;\n"
//                  "  int* restrict ___Out__L1_idx_11 = *(int**)Out[2];\n"
//                  "  int* _p2_10;\n"
//                  "  int ___Out__L0_ptr_12 = *(int*)Out[0];\n"
//                  "  _p_8 = ___A__L1_idx_9;\n"
//                  "  _p2_10 = ___Out__L1_idx_11;\n"
//                  "  ___Out__L0_ptr_12 = 4;\n"
//                  "\n"
//                  "  *(int**)Out[2] = (int*)___Out__L1_idx_11;\n"
//                  "  *(int*)Out[0] = ___Out__L0_ptr_12;\n"
//                  "  return 0;\n"
//                  "}\n");
//  
//  cout << expected << endl;
//  cout << normalize(foo.str()) << endl;
//  EXPECT_EQ(normalize(expected), normalize(foo.str()));
//
//}


TEST_F(BackendCTests, BuildModule) {
  auto add = Function::make("foobar", {Var::make("x", typeOf<int>())}, {}, Block::make({}));
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(add.as<Function>());

  Module mod(foo.str());
  mod.compile();
  
  typedef int (*fnptr_t)(void**);
  int ten = 10;
  void* pack[] = {(void*)&ten};
  //fnptr_t func = (fnptr_t)mod.get_func("foobar");
  static_assert(sizeof(void*) == sizeof(fnptr_t),
  "Unable to cast dlsym() returned void pointer to function pointer");
  void* v_func_ptr = mod.get_func("foobar");
  fnptr_t func;
  *reinterpret_cast<void**>(&func) = v_func_ptr;

  
  EXPECT_EQ(0, func(pack));
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
  
  typedef int (*fnptr_t)(void**);
  
//  fnptr_t func = (fnptr_t)mod.get_func("foobar");
  static_assert(sizeof(void*) == sizeof(fnptr_t),
  "Unable to cast dlsym() returned void pointer to function pointer");
  void* v_func_ptr = mod.get_func("foobar");
  fnptr_t func;
  *reinterpret_cast<void**>(&func) = v_func_ptr;
  
  int x = 22;
  double y = 1.8;
  // calling convention is output, inputs
  void* pack[] = {(void*)&x, (void*)&y};
  
  EXPECT_EQ(0, func(pack));
  EXPECT_EQ(101, x);
}

TEST_F(BackendCTests, CallModuleWithStore) {
  auto var = Var::make("x", typeOf<int>());
  auto fn = Function::make("foobar", {Var::make("y", typeOf<double>())},
    {var},
    Block::make({Store::make(var, Literal::make(0), Literal::make(99))}));
  
  auto var2 = Var::make("y", typeOf<double>());
  auto fn2 = Function::make("booper", {Var::make("x", typeOf<int>())},
    {var2},
    Block::make({Store::make(var2, Literal::make(0), Literal::make(-20.0))}));
  
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(fn.as<Function>());
  cg.compile(fn2.as<Function>());
  Module mod(foo.str());
  mod.compile();

  int x = 11;
  double y = 1.8;
  EXPECT_EQ(0, mod.call_func_packed("foobar", {(void*)(&x), (void*)(&y)}));
  EXPECT_EQ(99, x);
  
  EXPECT_EQ(0, mod.call_func_packed("booper", {(void*)&y, (void*)&x}));
  EXPECT_EQ(-20.0, y);
}

TEST_F(BackendCTests, FullVecAdd) {
  // implements:
  // for i = 0 to len
  //  a[i] = b[i] + c[i]
  auto a = Var::make("a", typeOf<float>());
  auto b = Var::make("b", typeOf<float>());
  auto c = Var::make("c", typeOf<float>());
  auto veclen = Var::make("len", typeOf<int>(), false);
  auto veclen_val = Load::make(veclen);
  auto i = Var::make("i", typeOf<int>(), false);

  auto fn = Function::make("vecadd",
    {veclen, b, c}, // inputs
    {a},    // outputs
    // body
    Block::make({
      For::make(i, Literal::make(0), veclen, Literal::make(1),
        Block::make({Store::make(a, i, Add::make(Load::make(b, i), Load::make(c, i)))
                    }))
      }));
  
  stringstream foo;
  CodeGen_C cg(foo);
  cg.compile(fn.as<Function>());
  Module mod(foo.str());
  mod.compile();
  
  float vec_a[10] = {0};
  float vec_b[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  float vec_c[10] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
  int ten = 10;
  
  // calling convention is output, inputs
  void* pack[] = {(void*)(vec_a), (void*)(&ten), (void*)(vec_b), (void*)(vec_c)};
  
  mod.call_func_packed("vecadd", pack);
  
  for (int j=0; j<10; j++)
    EXPECT_EQ(vec_b[j] + vec_c[j], vec_a[j]);
}





