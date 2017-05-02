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

std::string IRPrinter::keywordString(std::string keyword) {
  if (color) {
    return magenta + keyword + nc;
  }
  else {
    return keyword;
  }
}

std::string IRPrinter::commentString(std::string comment) {
  if (color) {
    return green + "/* " + comment + " */" + nc;
  }
  else {
    return "/* " + comment + " */";
  }
}

IRPrinter::IRPrinter(ostream &s) : IRPrinter(s, false, false) {
}

IRPrinter::IRPrinter(ostream &s, bool color, bool simplify)
    : stream(s), indent(0), color(color), simplify(simplify),
      omitNextParen(false) {
}

IRPrinter::~IRPrinter() {
}

void IRPrinter::do_indent() {
  for (int i=0; i<indent; i++)
    stream << "  ";
}


void IRPrinter::visit(const Literal* op) {
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

void IRPrinter::visit(const Var* op) {
  stream << op->name;
}

void IRPrinter::visit(const Neg* op) {
  omitNextParen = false;
  stream << "-";
  op->a.accept(this);
}

void IRPrinter::visit(const Sqrt* op) {
  omitNextParen = false;
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}

void IRPrinter::print_binop(Expr a, Expr b, string op) {
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

void IRPrinter::visit(const Add* op) {
  print_binop(op->a, op->b, "+");
}

void IRPrinter::visit(const Sub* op) {
  print_binop(op->a, op->b, "-");
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
  omitNextParen = false;
  stream << "min(";
  for (size_t i=0; i<op->operands.size(); i++) {
    op->operands[i].accept(this);
    if (i < op->operands.size()-1)
      stream << ", ";
  }
  stream << ")";
}

void IRPrinter::visit(const Max* op){
  omitNextParen = false;
  stream << "max(";
  op->a.accept(this);
  stream << ", ";
  op->b.accept(this);
  stream << ")";
}

void IRPrinter::visit(const BitAnd* op){
  print_binop(op->a, op->b, "&");
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

void IRPrinter::visit(const Case* op) {
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
  omitNextParen = true;
  op->data.accept(this);
  omitNextParen = false;
  stream << ";";
}

void IRPrinter::visit(const For* op) {
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

void IRPrinter::visit(const VarAssign* op) {
  do_indent();
  if (op->is_decl) {
    stream << keywordString(util::toString(op->lhs.type())) << " ";
  }
  op->lhs.accept(this);
  omitNextParen = true;
  bool printed = false;
  if (simplify) {
    const Add* add = op->rhs.as<Add>();
    if (add != nullptr && add->a == op->lhs) {
      stream << " += ";
      add->b.accept(this);
      printed = true;
    }
  }
  if (!printed) {
    stream << " = ";
    op->rhs.accept(this);
  }

  omitNextParen = false;
  stream << ";";
}

void IRPrinter::visit(const Allocate* op) {
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

void IRPrinter::visit(const Comment* op) {
  do_indent();
  stream << commentString(op->text);
}

void IRPrinter::visit(const BlankLine*) {
}

void IRPrinter::visit(const Print* op) {
  do_indent();
  stream << "printf(";
  stream << "\"" << op->fmt << "\"";
  for (auto e: op->params) {
    stream << ", ";
    e.accept(this);
  }
  stream << ");";
}

void IRPrinter::visit(const GetProperty* op) {
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

}}
