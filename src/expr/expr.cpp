#include "taco/expr/expr.h"

#include "taco/type.h"
#include "taco/format.h"
#include "taco/expr/expr_nodes.h"
#include "taco/util/name_generator.h"

using namespace std;

namespace taco {

// class IndexVar
struct IndexVar::Content {
  string name;
};

IndexVar::IndexVar() : IndexVar(util::uniqueName('i')) {}

IndexVar::IndexVar(const std::string& name) : content(new Content) {
  content->name = name;
}

std::string IndexVar::getName() const {
  return content->name;
}

bool operator==(const IndexVar& a, const IndexVar& b) {
  return a.content == b.content;
}

bool operator<(const IndexVar& a, const IndexVar& b) {
  return a.content < b.content;
}

std::ostream& operator<<(std::ostream& os, const IndexVar& var) {
  return os << var.getName();
}


// class TensorVar
struct TensorVar::Content {
  string name;
  Type type;
  Format format;
};

TensorVar::TensorVar() : TensorVar(Type(), Dense) {
}

TensorVar::TensorVar(const Type& type, const Format& format)
    : TensorVar(util::uniqueName('A'), type, format) {
}

TensorVar::TensorVar(const string& name, const Type& type, const Format& format)
    : content(new Content) {
  content->name = name;
}

std::string TensorVar::getName() const {
  return content->name;
}

const Type& TensorVar::getType() const {
  return content->type;
}

const Format& TensorVar::getFormat() const {
  return content->format;
}

bool operator==(const TensorVar& a, const TensorVar& b) {
  return a.content == b.content;
}

bool operator<(const TensorVar& a, const TensorVar& b) {
  return a.content < b.content;
}

std::ostream& operator<<(std::ostream& os, const TensorVar& var) {
  return os << var.getName() << " : " << var.getType();
}


// class IndexExpr
IndexExpr::IndexExpr(int val) : IndexExpr(new IntImmNode(val)) {
}

IndexExpr::IndexExpr(float val) : IndexExpr(new FloatImmNode(val)) {
}

IndexExpr::IndexExpr(double val) : IndexExpr(new DoubleImmNode(val)) {
}

IndexExpr IndexExpr::operator-() {
  return new NegNode(this->ptr);
}

void IndexExpr::accept(ExprVisitorStrict *v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const IndexExpr& expr) {
  if (!expr.defined()) return os << "Expr()";
  expr.ptr->print(os);
  return os;
}


// class Read
Access::Access(const Node* n) : IndexExpr(n) {
}

Access::Access(const TensorBase& tensor, const std::vector<IndexVar>& indices)
    : Access(new Node(tensor, indices)) {
}

Access::Access(const TensorBase& tensor, const TensorVar& tensorVar,
               const std::vector<IndexVar>& indices)
    : Access(new Node(tensor, tensorVar, indices)) {
}

const Access::Node* Access::getPtr() const {
  return static_cast<const Node*>(ptr);
}

const TensorBase& Access::getTensor() const {
  return getPtr()->tensor;
}

const TensorVar& Access::getTensorVar() const {
  return getPtr()->tensorVar;
}

const std::vector<IndexVar>& Access::getIndexVars() const {
  return getPtr()->indexVars;
}

void Access::operator=(const IndexExpr& expr) {
  TensorBase result = getPtr()->tensor;
  taco_uassert(!result.getExpr().defined()) << "Cannot reassign " << result;
  result.setExpr(getIndexVars(), expr);
}

void Access::operator=(const Access& expr) {
  operator=(static_cast<IndexExpr>(expr));
}

void Access::operator+=(const IndexExpr& expr) {
  TensorBase result = getPtr()->tensor;
  taco_uassert(!result.getExpr().defined()) << "Cannot reassign " << result;
  // TODO: check that result format is dense (for now only support accumulation into dense)

  result.setExpr(getIndexVars(), expr, true);
}

void Access::operator+=(const Access& expr) {
  operator+=(static_cast<IndexExpr>(expr));
}

// Operators
IndexExpr operator+(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new AddNode(lhs, rhs);
}

IndexExpr operator-(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new SubNode(lhs, rhs);
}

IndexExpr operator*(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new MulNode(lhs, rhs);
}

IndexExpr operator/(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new DivNode(lhs, rhs);
}

}
