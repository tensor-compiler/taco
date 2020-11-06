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

LinalgBase::LinalgBase(string name, Type tensorType) : name(name), tensorType(tensorType), idxcount(0),
  LinalgExpr(TensorVar(name, tensorType)) {
}
LinalgBase::LinalgBase(string name, Type tensorType, Format format) : name(name), tensorType(tensorType), idxcount(0),
  LinalgExpr(TensorVar(name, tensorType, format)) {
}

LinalgAssignment LinalgBase::operator=(const LinalgExpr& expr) {
  taco_iassert(isa<VarNode>(this->ptr));
  LinalgAssignment assignment = LinalgAssignment(to<LinalgVarNode>(this->get())->tensorVar, expr);
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
    const LinalgSubNode* sub = to<LinalgSubNode>(linalg.get());
    IndexExpr indexA = rewrite(sub->a, indices);
    IndexExpr indexB = rewrite(sub->b, indices);
    return new SubNode(indexA, indexB);
  } else if (isa<LinalgMatMulNode>(linalg.get())) {
    const LinalgMatMulNode* mul = to<LinalgMatMulNode>(linalg.get());
    IndexVar index = getUniqueIndex();
    IndexExpr indexA = rewrite(mul->a, {indices[0], index});
    IndexExpr indexB = rewrite(mul->b, {index, indices[1]});
    return new MulNode(indexA, indexB);
  } else if (isa<LinalgVarNode>(linalg.get())) {
    const LinalgVarNode* var = to<LinalgVarNode>(linalg.get());
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
