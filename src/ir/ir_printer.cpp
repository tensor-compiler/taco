#include <sstream>
#include <iostream>

#include "ir_printer.h"
#include "ir.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace ir {

const std::string magenta="\033[38;5;204m";
const std::string blue="\033[38;5;67m";
const std::string green="\033[38;5;70m";
const std::string orange="\033[38;5;214m";
const std::string nc="\033[0m";

std::string IRPrinterBase::keywordString(std::string keyword) {
  if (color) {
    return magenta + keyword + nc;
  }
  else {
    return keyword;
  }
}

std::string IRPrinterBase::commentString(std::string comment) {
  if (color) {
    return green + "/* " + comment + " */" + nc;
  }
  else {
    return "/* " + comment + " */";
  }
}

IRPrinterBase::IRPrinterBase(ostream &s) : IRPrinterBase(s, false) {
}

IRPrinterBase::IRPrinterBase(ostream &s, bool color) : stream(s),
    indent(0), color(color), omitNextParen(false) {
}

IRPrinterBase::~IRPrinterBase() {
}

void IRPrinterBase::do_indent() {
  for (int i=0; i<indent; i++)
    stream << "  ";
}


void IRPrinterBase::visit(const Literal* op) {
  if (color)
    stream << blue ;
  if (op->type == typeOf<float>())
    stream << (float)(op->dbl_value);
  else if (op->type == typeOf<double>())
    stream << (double)(op->dbl_value);
  else
    stream << op->value;
  if (color)
    stream << nc;
}

void IRPrinterBase::visit(const Var* op) {
  stream << op->name;
}

void IRPrinterBase::visit(const Neg* op) {
  omitNextParen = false;
  stream << "-";
  op->a.accept(this);
}

void IRPrinterBase::visit(const Sqrt* op) {
  omitNextParen = false;
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}

void IRPrinterBase::print_binop(Expr a, Expr b, string op) {
  bool omitParen = omitNextParen;
  omitNextParen = false;

  if (!omitParen)
    stream << "(";
  a.accept(this);
  stream << " " << op << " ";
  b.accept(this);
  if (!omitParen)
    stream << ")";
}

void IRPrinterBase::visit(const Add* op) {
  print_binop(op->a, op->b, "+");
}

void IRPrinterBase::visit(const Sub* op) {
  print_binop(op->a, op->b, "-");
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
  omitNextParen = false;
  stream << "min(";
  for (size_t i=0; i<op->operands.size(); i++) {
    op->operands[i].accept(this);
    if (i < op->operands.size()-1)
      stream << ", ";
  }
  stream << ")";
}

void IRPrinterBase::visit(const Max* op){
  omitNextParen = false;
  stream << "max(";
  op->a.accept(this);
  stream << ", ";
  op->b.accept(this);
  stream << ")";
}

void IRPrinterBase::visit(const BitAnd* op){
  print_binop(op->a, op->b, "&");
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

void IRPrinterBase::visit(const Or* op) {
  print_binop(op->a, op->b, "||");
}

void IRPrinterBase::visit(const IfThenElse* op) {
  auto oparen = op->cond.as<Var>() ? "(" : "";
  auto cparen = op->cond.as<Var>() ? ")" : "";
  stream << "if " << oparen;
  op->cond.accept(this);
  stream << cparen << "\n";
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

  if (op->otherwise.defined()) {
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
}

void IRPrinterBase::visit(const Case* op) {
  for (size_t i=0; i < op->clauses.size(); ++i) {
    auto clause = op->clauses[i];
    if (i != 0) stream << "\n";
    do_indent();
    if (i == 0) {
      stream << keywordString("if ");
      clause.first.accept(this);
    }
    else if (i < op->clauses.size()-1 || !op->alwaysMatch) {
      stream << keywordString("else if ");
      clause.first.accept(this);
    }
    else {
      stream << keywordString("else");
    }
    stream << " {\n";
    indent++;
    clause.second.accept(this);
    indent--;
    stream << "\n";
    do_indent();
    stream << "}";
  }
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
  omitNextParen = true;
  op->data.accept(this);
  omitNextParen = false;
  stream << ";";
}

void IRPrinterBase::visit(const For* op) {
  do_indent();
  stream << keywordString("for") << " (int ";
  op->var.accept(this);
  stream << " = ";
  op->start.accept(this);
  stream << keywordString("; ");
  op->var.accept(this);
  stream << " < ";
  op->end.accept(this);
  stream << keywordString("; ");
  op->var.accept(this);
  stream << " += ";
  op->increment.accept(this);
  stream << ") {\n";

  indent++;
  if (!(op->contents.as<Block>())) {
    do_indent();
  }
  op->contents.accept(this);
  stream << "\n";
  indent--;
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
  if (op->is_decl) {
    stream << keywordString(util::toString(op->lhs.type())) << " ";
  }
  op->lhs.accept(this);
  stream << " = ";
  omitNextParen = true;
  op->rhs.accept(this);
  omitNextParen = false;
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
  stream << commentString(op->text);
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
    stream << ".d" << op->dim+1;
    if (op->property == TensorProperty::Index)
      stream << ".idx";
    if (op->property == TensorProperty::Pointer)
      stream << ".pos";
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

void IRPrinter::visit(const And* op) {
  print_binop(op->a, op->b, keywordString("&&"));
}

void IRPrinter::visit(const Or* op) {
  print_binop(op->a, op->b, keywordString("||"));
}

void IRPrinter::visit(const IfThenElse* op) {
  taco_iassert(op->cond.defined());
  taco_iassert(op->then.defined());

  do_indent();
  stream << keywordString("if ");
  op->cond.accept(this);

  if (op->then.as<Block>()) {
    stream << " {";
  }
  stream << "\n";

  indent++;
  op->then.accept(this);
  indent--;
  if (op->then.as<Block>()) {
    do_indent();
    stream << "\n";
    do_indent();
    stream << "}";
  }

  if (op->otherwise.defined()) {
    stream << "\n";
    do_indent();
    stream << "else";
    if (op->then.as<Block>()) {
      stream << " {";
    }
    stream << "\n";

    do_indent();
    stream << "\n";
    indent++;
    op->otherwise.accept(this);
    indent--;
    if (op->then.as<Block>()) {
      do_indent();
      stream << "\n";
      do_indent();
      stream << "}";
    }
  }
}

void IRPrinter::visit(const Function* op) {
  stream << keywordString("void ") << op->name;
  stream << "(";
  if (op->outputs.size() > 0) stream << "Tensor ";
  acceptJoin(this, stream, op->outputs, ", Tensor ");
  if (op->outputs.size() > 0 && op->inputs.size()) stream << ", ";
  if (op->inputs.size() > 0) stream << "Tensor ";
  acceptJoin(this, stream, op->inputs, ", Tensor ");
  stream << ") {\n";
  indent++;
  op->body.accept(this);
  indent--;
  stream << "\n";
  do_indent();
  stream << "}";
}

void IRPrinter::visit(const While* op) {
  do_indent();
  stream << keywordString("while ");
  op->cond.accept(this);
  stream << " {\n";

  indent++;
  op->contents.accept(this);
  indent--;
  stream << "\n";
  do_indent();
  stream << "}";
}

void IRPrinter::visit(const Block* op) {
  acceptJoin(this, stream, op->contents, "\n");
}

}}
