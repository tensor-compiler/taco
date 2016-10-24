#include "ir.h"
#include "test.h"
#include "ir_printer.h"

using namespace ::testing;
using namespace taco::ir;

struct IRPrinterTests : public Test {

};

TEST_F(IRPrinterTests, PrintBinOps) {
  auto add = Add::make(Var::make("foo", typeOf<int>()),
    Literal::make(100));
  
  std::stringstream out;
  IRPrinter irp(out);
  add.accept(&irp);
  EXPECT_EQ("(foo + 100)", out.str());
  
  out.str("");
  
  auto neq = Neq::make(Var::make("bar", typeOf<float>()),
    Literal::make(2.2, typeOf<float>()));
  neq.accept(&irp);
  EXPECT_EQ("(bar != 2.2)", out.str());
  
}

TEST_F(IRPrinterTests, PrintIfThenElse) {
  auto ifte = IfThenElse::make(Neq::make(Var::make("bar", typeOf<float>()),
    Literal::make(2.2, typeOf<float>())), Block::make({VarAssign::make(Var::make("x", typeOf<int>()), Literal::make(10))}), Block::make());
  
  std::stringstream out;
  IRPrinter irp(out);
  ifte.accept(&irp);
  std::string output = "if ((bar != 2.2))\n"
                       "{\n  x = 10;\n}\n"
                       "else\n{\n}";
  EXPECT_EQ(output, out.str());
}

TEST_F(IRPrinterTests, PrintFor) {
  auto loop = For::make(Var::make("x", typeOf<int>()), Literal::make(0), Literal::make(10), Literal::make(2), Block::make({VarAssign::make(Var::make("z", typeOf<int>()), Literal::make(2)),
        Print::make("z") }));
  
  std::stringstream out;
  IRPrinter irp(out);
  loop.accept(&irp);
  auto output =
        "for (int x = 0; x < 10; x += 2)\n"
        "{\n"
        "  z = 2;\n"
        "  printf(\"z\");\n"
        "}";
  EXPECT_EQ(output, out.str());

}

TEST_F(IRPrinterTests, PrintWhile) {
  auto loop = While::make(Eq::make(Var::make("x", typeOf<int>()), Literal::make(0)),
    Block::make({VarAssign::make(Var::make("z", typeOf<int>()), Literal::make(2)),
        Print::make("z") }));
  
  std::stringstream out;
  IRPrinter irp(out);
  loop.accept(&irp);
  auto output =
        "while (x == 0)\n"
        "{\n"
        "  z = 2;\n"
        "  printf(\"z\");\n"
        "}";
  EXPECT_EQ(output, out.str());

}

TEST_F(IRPrinterTests, PrintWithProperty) {
  auto var = Var::make("A", typeOf<double>());
  auto prop = GetProperty::make(var, TensorProperty::Pointer, 2);
  auto loop = Function::make("foo", {}, {}, Block::make({While::make(Eq::make(Var::make("x", typeOf<int>()), Literal::make(0)),
    Block::make({VarAssign::make(Var::make("z", typeOf<int>()), prop),
        Print::make("z") }))}));
  
  std::stringstream out;
  IRPrinter irp(out);
  loop.accept(&irp);
  auto output =
        "function foo() -> ()\n"
        "{\n"
        "  while (x == 0)\n"
        "  {\n"
        "    z = A.L2.nnz;\n"
        "    printf(\"z\");\n"
        "  }\n"
        "}\n";
  
  EXPECT_EQ(output, out.str());

}

TEST_F(IRPrinterTests, PrintFunction) {
  auto loop = Function::make("foo", {}, {}, Block::make({While::make(Eq::make(Var::make("x", typeOf<int>()), Literal::make(0)),
    Block::make({VarAssign::make(Var::make("z", typeOf<int>()), Literal::make(2)),
        Print::make("z") }))}));
  
  std::stringstream out;
  IRPrinter irp(out);
  loop.accept(&irp);
  auto output =
        "function foo() -> ()\n"
        "{\n"
        "  while (x == 0)\n"
        "  {\n"
        "    z = 2;\n"
        "    printf(\"z\");\n"
        "  }\n"
        "}\n";
  
  EXPECT_EQ(output, out.str());

}
