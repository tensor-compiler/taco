#include <sstream>
#include <iostream>
#include "ir_printer.h"
#include "ir.h"

using namespace std;

namespace taco {
namespace ir {

IRPrinterBase::IRPrinterBase(ostream &s) : stream(s), indent(0) {
}

IRPrinterBase::~IRPrinterBase() {
}

void IRPrinterBase::do_indent() {
  for (int i=0; i<indent; i++)
    stream << "  ";
}


void IRPrinterBase::visit(const Literal* op) {
  if (op->type == typeOf<float>())
    stream << (float)(op->dbl_value);
  else if (op->type == typeOf<double>())
    stream << (double)(op->dbl_value);
  else
    stream << op->value;
}

void IRPrinterBase::visit(const Var* op) {
  stream << op->name;
}

void IRPrinterBase::print_binop(Expr a, Expr b, string op) {
  stream << "(";
  a.accept(this);
  stream << " " << op << " ";
  b.accept(this);
  stream << ")";
}

void IRPrinterBase::visit(const Add* op) {
  print_binop(op->a, op->b, "+");
}

void IRPrinterBase::visit(const Sub* op) {
  print_binop(op->a, op->b, "i");
}

void IRPrinterBase::visit(const Mul* op) {
  print_binop(op->a, op->b, "*");
}

void IRPrinterBase::visit(const Div* op) {
  print_binop(op->a, op->b, "/");
}

void IRPrinterBase::visit(const Rem* op) {
  print_binop(op->a, op->b, "%");
}

void IRPrinterBase::visit(const Min* op) {
  stream << "min(";
  op->a.accept(this);
  stream << ", ";
  op->b.accept(this);
  stream << ")";
}

void IRPrinterBase::visit(const Max* op){
  stream << "max(";
  op->a.accept(this);
  stream << ", ";
  op->b.accept(this);
  stream << ")";
}

void IRPrinterBase::visit(const Eq* op){
  print_binop(op->a, op->b, "==");
}

void IRPrinterBase::visit(const Neq* op) {
  print_binop(op->a, op->b, "!=");
}

void IRPrinterBase::visit(const Gt* op) {
  print_binop(op->a, op->b, ">");
}

void IRPrinterBase::visit(const Lt* op) {
  print_binop(op->a, op->b, "<");
}

void IRPrinterBase::visit(const Gte* op) {
  print_binop(op->a, op->b, ">=");
}

void IRPrinterBase::visit(const Lte* op) {
  print_binop(op->a, op->b, "<=");
}

void IRPrinterBase::visit(const And* op) {
  print_binop(op->a, op->b, "&&");
}

void IRPrinterBase::visit(const Or* op)
{
  print_binop(op->a, op->b, "||");
}

void IRPrinterBase::visit(const IfThenElse* op) {
  stream << "if (";
  op->cond.accept(this);
  stream << ")\n";
  do_indent();
  stream << "{\n";
  if (!(op->then.as<Block>())) {
    indent++;
  }
  op->then.accept(this);
  if (!(op->then.as<Block>())) {
    indent--;
  }
  do_indent();
  stream << "}\n";
  do_indent();
  stream << "else\n";
  do_indent();
  stream << "{\n";
  if (!(op->otherwise.as<Block>())) {
    indent++;
  }
  op->otherwise.accept(this);
    if (!(op->otherwise.as<Block>())) {
    indent--;
  }
  do_indent();
  stream << "}";
}

void IRPrinterBase::visit(const Load* op) {
  op->arr.accept(this);
  stream << "[";
  op->loc.accept(this);
  stream << "]";
}

void IRPrinterBase::visit(const Store* op) {
  do_indent();
  op->arr.accept(this);
  stream << "[";
  op->loc.accept(this);
  stream << "] = ";
  op->data.accept(this);
  stream << ";";
}

void IRPrinterBase::visit(const For* op) {
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
  
  do_indent();
  stream << "{\n";
  
  if (!(op->contents.as<Block>())) {
    indent++;
    do_indent();
  }
  op->contents.accept(this);
  
  if (!(op->contents.as<Block>())) {
    indent--;
  }
  do_indent();
  stream << "}";
}

void IRPrinterBase::visit(const While* op) {
  do_indent();
  stream << "while ";
  op->cond.accept(this);
  stream << "\n";
  do_indent();
  stream << "{\n";
   if (!(op->contents.as<Block>())) {
    indent++;
    do_indent();
  }
  op->contents.accept(this);
  
  if (!(op->contents.as<Block>())) {
    indent--;
  }
  do_indent();
  stream << "}";

}

void IRPrinterBase::visit(const Block* op) {
  indent++;

  for (auto s: op->contents) {
    s.accept(this);
    stream << "\n";
  }
  indent--;
}

void IRPrinterBase::visit(const Function* op) {
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
  do_indent();
  stream << "{\n";
  op->body.accept(this);
  do_indent();
  stream << "}\n";
}

void IRPrinterBase::visit(const VarAssign* op) {
  do_indent();
  op->lhs.accept(this);
  stream << " = ";
  op->rhs.accept(this);
  stream << ";";
}

void IRPrinterBase::visit(const Allocate* op) {
  do_indent();
  if (op->is_realloc)
    stream << "reallocate ";
  else
    stream << "allocate ";
  op->var.accept(this);
  stream << "[ ";
  op->num_elements.accept(this);
  stream << "]";
}

void IRPrinterBase::visit(const Comment* op) {
  do_indent();
  stream << "// " << op->text;
}

void IRPrinterBase::visit(const BlankLine*) {
}

void IRPrinterBase::visit(const Print* op) {
  do_indent();
  stream << "printf(";
  stream << "\"" << op->fmt << "\"";
  for (auto e: op->params) {
    stream << ", ";
    e.accept(this);
  }
  stream << ");";
}

void IRPrinterBase::visit(const GetProperty* op) {
  op->tensor.accept(this);
  if (op->property == TensorProperty::Values) {
    stream << ".vals";
  } else {
    stream << ".d" << op->dim;
    if (op->property == TensorProperty::Index)
      stream << ".idx";
    if (op->property == TensorProperty::NNZ)
      stream << ".nnz";
    if (op->property == TensorProperty::Pointer)
      stream << ".ptr";
  }
  
}


// class IRPrinter
IRPrinter::~IRPrinter() {
}

template <class T>
static inline void acceptJoin(IRPrinter* printer, ostream& stream,
                              vector<T> nodes, string sep) {
  if (nodes.size() > 0) {
    nodes[0].accept(printer);
  }
  for (size_t i=1; i < nodes.size(); ++i) {
    stream << sep;
    nodes[i].accept(printer);
  }
}

void IRPrinter::visit(const Function* op) {
  stream << "function " << op->name;
  stream << "(";
  acceptJoin(this, stream, op->inputs, ", ");
  stream << ") -> (";
  acceptJoin(this, stream, op->outputs, ", ");
  stream << ")\n";
  do_indent();
  op->body.accept(this);
  do_indent();
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

void IRPrinter::visit(const Block* op) {
  indent++;
  acceptJoin(this, stream, op->contents, "\n");
  indent--;
}

}}
