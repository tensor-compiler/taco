#include "ir.h"
#include "test.h"
#include "ir_printer.h"

using namespace ::testing;
using namespace taco::internal;

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
                       "{\n  x = 10\n}\n"
                       "else\n{\n}";
  EXPECT_EQ(output, out.str());
}
