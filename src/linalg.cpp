#include "taco/linalg.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/linalg_notation/linalg_notation_nodes.h"

using namespace std;

namespace taco {

// Just trying this out. Need to accept dimensions and format too.
/* LinalgBase::LinalgBase(Datatype ctype) */
/*   : LinalgBase(/1* get a unique name *1/, ctype) { */
/* } */

LinalgBase::LinalgBase(string name, Type tensorType, bool isColVec) : name(name), tensorType(tensorType), idxcount(0),
  LinalgExpr(TensorVar(name, tensorType), isColVec) {
}
LinalgBase::LinalgBase(string name, Type tensorType, Format format, bool isColVec) : name(name), tensorType(tensorType),
  idxcount(0), LinalgExpr(TensorVar(name, tensorType, format), isColVec) {
}

LinalgAssignment LinalgBase::operator=(const LinalgExpr& expr) {
  taco_iassert(isa<LinalgVarNode>(this->ptr));
  TensorVar var = to<LinalgVarNode>(this->get())->tensorVar;

  taco_uassert(var.getOrder() == expr.getOrder()) << "RHS and LHS of linalg assignment must match order";
  if (var.getOrder() == 1)
    taco_uassert(this->isColVector() == expr.isColVector()) << "RHS and LHS of linalg assignment must match vector type";

  LinalgAssignment assignment = LinalgAssignment(var, expr);
  this->assignment = assignment;
  return assignment;
}

const LinalgAssignment LinalgBase::getAssignment() const{
  return this->assignment;
}
const IndexStmt LinalgBase::getIndexAssignment() const {
  if (this->indexAssignment.defined()) {
    return this->indexAssignment;
  }
  return IndexStmt();
}

vector<IndexVar> LinalgBase::getUniqueIndices(size_t order) {
  vector<IndexVar> result;
  for (int i = idxcount; i < (idxcount + (int)order); i++) {
    cout << i << ": ";
    string name = "i" + to_string(i);
    IndexVar indexVar(name);
    result.push_back(indexVar);
  }
  idxcount += order;
  return result;
}

IndexVar LinalgBase::getUniqueIndex() {
  string name = "i" + to_string(idxcount);
  idxcount += 1;
  IndexVar result(name);
  return result;
}

IndexExpr LinalgBase::rewrite(LinalgExpr linalg, vector<IndexVar> indices) {
  if (isa<LinalgSubNode>(linalg.get())) {
    auto sub = to<LinalgSubNode>(linalg.get());
    IndexExpr indexA = rewrite(sub->a, indices);
    IndexExpr indexB = rewrite(sub->b, indices);
    return new SubNode(indexA, indexB);
  } else if (isa<LinalgAddNode>(linalg.get())) {
    auto add = to<LinalgAddNode>(linalg.get());
    IndexExpr indexA = rewrite(add->a, indices);
    IndexExpr indexB = rewrite(add->b, indices);
    return new AddNode(indexA, indexB);
  } else if (isa<LinalgElemMulNode>(linalg.get())) {
    auto mul = to<LinalgElemMulNode>(linalg.get());
    IndexExpr indexA = rewrite(mul->a, indices);
    IndexExpr indexB = rewrite(mul->b, indices);
    return new MulNode(indexA, indexB);
  } else if (isa<LinalgMatMulNode>(linalg.get())) {
    auto mul = to<LinalgMatMulNode>(linalg.get());
    IndexVar index = getUniqueIndex();
    vector<IndexVar> indicesA;
    vector<IndexVar> indicesB;
    if (mul->a.getOrder() == 2 && mul->b.getOrder() == 2) {
      indicesA = {indices[0], index};
      indicesB = {index, indices[1]};
    }
    else if (mul->a.getOrder() == 1 && mul->b.getOrder() == 2) {
      indicesA = {index};
      indicesB = {index, indices[0]};
    }
    else if (mul->a.getOrder() == 2 && mul->b.getOrder() == 1) {
      indicesA = {indices[0], index};
      indicesB = {index};
    }
    else if (mul->a.getOrder() == 1 && mul->a.isColVector() && mul->b.getOrder() == 1) {
      indicesA = {indices[0], index};
      indicesB = {index};
    } else {
      indicesA = {index};
      indicesB = {index};
    }
    IndexExpr indexA = rewrite(mul->a, indicesA);
    IndexExpr indexB = rewrite(mul->b, indicesB);
    return new MulNode(indexA, indexB);
  } else if (isa<LinalgDivNode>(linalg.get())) {
    auto div = to<LinalgDivNode>(linalg.get());
    IndexExpr indexA = rewrite(div->a, indices);
    IndexExpr indexB = rewrite(div->b, indices);
    return new DivNode(indexA, indexB);
  } else if (isa<LinalgNegNode>(linalg.get())) {
    auto neg = to<LinalgNegNode>(linalg.get());
    IndexExpr index = rewrite(neg->a, indices);
    return new NegNode(index);
  } else if (isa<LinalgTransposeNode>(linalg.get())) {
    auto transpose = to<LinalgTransposeNode>(linalg.get());
    if (transpose->a.getOrder() == 2) {
      return rewrite(transpose->a, {indices[1], indices[0]});
    }
    else if (transpose->a.getOrder() == 1) {
      return rewrite(transpose->a, {indices[0]});
    }
    return rewrite(transpose->a, {});
  } else if (isa<LinalgLiteralNode>(linalg.get())) {
    auto lit = to<LinalgLiteralNode>(linalg.get());

    LiteralNode* value;
    switch (lit->getDataType().getKind()) {
      case Datatype::Bool:
        value = new LiteralNode(lit->getVal<bool>());
        break;
      case Datatype::UInt8:
        value = new LiteralNode(lit->getVal<uint8_t>());
        break;
      case Datatype::UInt16:
        value = new LiteralNode(lit->getVal<uint16_t>());
        break;
      case Datatype::UInt32:
        value = new LiteralNode(lit->getVal<uint32_t>());
        break;
      case Datatype::UInt64:
        value = new LiteralNode(lit->getVal<uint64_t>());
        break;
      case Datatype::UInt128:
        taco_not_supported_yet;
        break;
      case Datatype::Int8:
        value = new LiteralNode(lit->getVal<int8_t>());
        break;
      case Datatype::Int16:
        value = new LiteralNode(lit->getVal<int16_t>());
        break;
      case Datatype::Int32:
        value = new LiteralNode(lit->getVal<int32_t>());
        break;
      case Datatype::Int64:
        value = new LiteralNode(lit->getVal<int64_t>());
        break;
      case Datatype::Int128:
        taco_not_supported_yet;
        break;
      case Datatype::Float32:
        value = new LiteralNode(lit->getVal<float>());
        break;
      case Datatype::Float64:
        value = new LiteralNode(lit->getVal<double>());
        break;
      case Datatype::Complex64:
        value = new LiteralNode(lit->getVal<std::complex<float>>());
        break;
      case Datatype::Complex128:
        value = new LiteralNode(lit->getVal<std::complex<double>>());
        break;
      case Datatype::Undefined:
        taco_uerror << "unsupported Datatype";
        break;
    }
    return value;
  } else if (isa<LinalgVarNode>(linalg.get())) {
    auto var = to<LinalgVarNode>(linalg.get());
    return new AccessNode(var->tensorVar, indices);
  }
  return IndexExpr();
}

IndexStmt rewrite(LinalgStmt linalg) {
  return IndexStmt();
}

IndexStmt LinalgBase::rewrite() {
  if (this->assignment.defined()) {

    TensorVar tensor = this->assignment.getLhs();

    vector<IndexVar> indices;
    if (tensor.getOrder() == 1) {
      indices.push_back(IndexVar("i"));
    } else if (tensor.getOrder() == 2) {
      indices.push_back(IndexVar("i"));
      indices.push_back(IndexVar("j"));
    }
    Access lhs = Access(tensor, indices);
    IndexExpr rhs = rewrite(this->assignment.getRhs(), indices);

    Assignment indexAssign = Assignment(lhs, rhs);
    this->indexAssignment = indexAssign;
    return indexAssign;
  }
  return IndexStmt();
}

std::ostream& operator<<(std::ostream& os, const LinalgBase& linalg) {
  LinalgAssignment assignment = linalg.getAssignment();
  if (!assignment.defined()) return os << getNode(linalg)->tensorVar.getName();
  LinalgNotationPrinter printer(os);
  printer.print(assignment);
  return os;
}

}
