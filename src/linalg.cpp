#include "taco/linalg.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/linalg_notation/linalg_notation_nodes.h"
#include "taco/linalg_notation/linalg_rewriter.h"

using namespace std;

namespace taco {

LinalgBase::LinalgBase(string name, Type tensorType, Datatype dtype, std::vector<int> dims, Format format, bool isColVec) :
  LinalgExpr(TensorVar(name, tensorType, format), isColVec, new TensorBase(name, dtype, dims, format)), name(name),
  tensorType(tensorType), idxcount(0) {
}

LinalgAssignment LinalgBase::operator=(const LinalgExpr& expr) {
  taco_iassert(isa<LinalgVarNode>(this->ptr));
  TensorVar var = to<LinalgVarNode>(this->get())->tensorVar;

  taco_uassert(var.getOrder() == expr.getOrder()) << "LHS (" << var.getOrder() << ") and RHS (" << expr.getOrder()
                                                      << ") of linalg assignment must match order";
  if (var.getOrder() == 1)
    taco_uassert(this->isColVector() == expr.isColVector()) << "RHS and LHS of linalg assignment must match vector type";

  LinalgAssignment assignment = LinalgAssignment(var, expr);
  this->assignment = assignment;
  this->rewrite();
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
    string name = "i" + to_string(i);
    IndexVar indexVar(name);
    result.push_back(indexVar);
  }
  idxcount += order;
  return result;
}

IndexVar LinalgBase::getUniqueIndex() {
  int loc = idxcount % indexVarNameList.size();
  int num = idxcount / indexVarNameList.size();

  string indexVarName;
  if (num == 0)
    indexVarName = indexVarNameList.at(loc);
  else
    indexVarName = indexVarNameList.at(loc) + to_string(num);

  idxcount += 1;
  IndexVar result(indexVarName);
  return result;
}

IndexExpr LinalgBase::rewrite(LinalgExpr linalg, vector<IndexVar> indices) {
  return IndexExpr();
}

IndexStmt rewrite(LinalgStmt linalg) {
  return IndexStmt();
}

IndexStmt LinalgBase::rewrite() {
  if (this->assignment.defined()) {
    auto linalgRewriter = new LinalgRewriter();
    //linalgRewriter->setLiveIndices(indices);
    IndexStmt stmt = linalgRewriter->rewrite(*this);
    this->indexAssignment = stmt;
    return stmt;
  }
  return IndexStmt();
}

std::ostream& operator<<(std::ostream& os, const LinalgBase& linalg) {
  LinalgAssignment assignment = linalg.getAssignment();

  // If TensorBase exists, print the storage
  if (linalg.tensorBase != nullptr) {
    return os << *(linalg.tensorBase) << endl;
  }

  if (!assignment.defined()) return os << getNode(linalg)->tensorVar.getName();
  LinalgNotationPrinter printer(os);
  printer.print(assignment);
  return os;
}

}
