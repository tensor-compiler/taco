#include <sstream>
#include <iostream>
#include "ir_printer.h"
#include "ir.h"

using namespace std;

namespace taco {
namespace internal {

IRPrinter::IRPrinter(ostream &s) : stream(s), indent(0) {
}

void IRPrinter::do_indent() {
  for (int i=0; i<indent; i++)
    stream << "  ";
}


void IRPrinter::visit(const Literal* op) {
  if (op->type == typeOf<float>())
    stream << (float)(op->dbl_value);
  else if (op->type == typeOf<double>())
    stream << (double)(op->dbl_value);
  else
    stream << op->value;
}

void IRPrinter::visit(const Var* op) {
  stream << op->name;
}

void IRPrinter::print_binop(Expr a, Expr b, string op) {
  stream << "(";
  a.accept(this);
  stream << " " << op << " ";
  b.accept(this);
  stream << ")";
}

void IRPrinter::visit(const Add* op) {
  print_binop(op->a, op->b, "+");
}

void IRPrinter::visit(const Sub* op) {
  print_binop(op->a, op->b, "i");
}

void IRPrinter::visit(const Mul* op) {
  print_binop(op->a, op->b, "*");
}

void IRPrinter::visit(const Div* op) {
  print_binop(op->a, op->b, "/");
}

void IRPrinter::visit(const Rem* op) {
  print_binop(op->a, op->b, "%");
}

void IRPrinter::visit(const Min* op) {
  stream << "min(";
  op->a.accept(this);
  stream << ", ";
  op->b.accept(this);
  stream << ")";
}

void IRPrinter::visit(const Max* op){
  stream << "max(";
  op->a.accept(this);
  stream << ", ";
  op->b.accept(this);
  stream << ")";
}

void IRPrinter::visit(const Eq* op){
  print_binop(op->a, op->b, "==");
}

void IRPrinter::visit(const Neq* op) {
  print_binop(op->a, op->b, "!=");
}

void IRPrinter::visit(const Gt* op) {
  print_binop(op->a, op->b, ">");
}

void IRPrinter::visit(const Lt* op) {
  print_binop(op->a, op->b, "<");
}

void IRPrinter::visit(const Gte* op) {
  print_binop(op->a, op->b, ">=");
}

void IRPrinter::visit(const Lte* op) {
  print_binop(op->a, op->b, "<=");
}

void IRPrinter::visit(const And* op) {
  print_binop(op->a, op->b, "&&");
}

void IRPrinter::visit(const Or* op)
{
  print_binop(op->a, op->b, "||");
}

void IRPrinter::visit(const IfThenElse* op) {
  stream << "if (";
  op->cond.accept(this);
  stream << ")\n";
  if (!(op->then.as<Block>())) {
    indent++;
  }
  op->then.accept(this);
  if (!(op->then.as<Block>())) {
    indent--;
  }
  do_indent();
  stream << "\n";
  stream << "else\n";
  if (!(op->otherwise.as<Block>())) {
    indent++;
  }
  op->otherwise.accept(this);
    if (!(op->otherwise.as<Block>())) {
    indent--;
  }
}

void IRPrinter::visit(const Load* op) {
  op->arr.accept(this);
  stream << "[";
  op->loc.accept(this);
  stream << "]";
}

void IRPrinter::visit(const Store* op) {
  do_indent();
  op->arr.accept(this);
  stream << "[";
  op->loc.accept(this);
  stream << "] = ";
  op->data.accept(this);
}

void IRPrinter::visit(const For* op) {
  do_indent();
  stream << "for (int ";
  op->var.accept(this);
  stream << " = ";
  op->start.accept(this);
  stream << "; ";
  op->var.accept(this);
  stream << " < ";
  op->end.accept(this);
  stream << "; ";
  op->var.accept(this);
  stream << " += ";
  op->increment.accept(this);
  stream << ")\n";
  
  if (!(op->contents.as<Block>())) {
    indent++;
    do_indent();
  }
  op->contents.accept(this);
  
  if (!(op->contents.as<Block>())) {
    indent--;
  }
}

void IRPrinter::visit(const While* op) {
  do_indent();
  stream << "while (";
  op->cond.accept(this);
  stream << ")\n";
   if (!(op->contents.as<Block>())) {
    indent++;
    do_indent();
  }
  op->contents.accept(this);
  
  if (!(op->contents.as<Block>())) {
    indent--;
  }

}

void IRPrinter::visit(const Block* op) {
  do_indent();
  stream << "{\n";
  indent++;
  for (auto s: op->contents) {
    s.accept(this);
    stream << "\n";
  }
  indent--;
  do_indent();
  stream << "}";
}

void IRPrinter::visit(const Function* op) {
  stream << "function " << op->name;
  stream << "(";
  for (auto input : op->inputs) {
    input.accept(this);
    stream << " ";
  }
  stream << ") -> (";
  for (auto output : op->outputs) {
    output.accept(this);
    stream << " ";
  }
  stream << ")\n";
  
  op->body.accept(this);
}

void IRPrinter::visit(const VarAssign* op) {
  do_indent();
  op->lhs.accept(this);
  stream << " = ";
  op->rhs.accept(this);
}

void IRPrinter::visit(const Allocate* op) {
  do_indent();
  stream << "allocate ";
  op->var.accept(this);
  stream << "[ ";
  op->num_elements.accept(this);
  stream << "]";
}

}
}
