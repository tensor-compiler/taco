#include "taco/linalg_notation/linalg_rewriter.h"

#include "taco/linalg_notation/linalg_notation_nodes.h"
#include "taco/index_notation/index_notation_nodes.h"

using namespace std;
using namespace taco;

class LinalgRewriter::Visitor : public LinalgNotationVisitorStrict {
public:
  Visitor(LinalgRewriter* rewriter ) : rewriter(rewriter) {}
  IndexExpr rewrite(LinalgExpr linalgExpr) {
    this->expr = IndexExpr();
    LinalgNotationVisitorStrict::visit(linalgExpr);
    return this->expr;
  }
  IndexStmt rewrite(LinalgStmt linalgStmt) {
    this->stmt = IndexStmt();
    LinalgNotationVisitorStrict::visit(linalgStmt);
    return this->stmt;
  }
private:
  LinalgRewriter* rewriter;
  IndexExpr expr;
  IndexStmt stmt;
  using LinalgNotationVisitorStrict::visit;
  void visit(const LinalgSubNode* node)         { expr = rewriter->rewriteSub(node); }
  void visit(const LinalgAddNode* node)         { expr = rewriter->rewriteAdd(node); }
  void visit(const LinalgElemMulNode* node)     { expr = rewriter->rewriteElemMul(node); }
  void visit(const LinalgMatMulNode* node)      { expr = rewriter->rewriteMatMul(node); }
  void visit(const LinalgDivNode* node)         { expr = rewriter->rewriteDiv(node); }
  void visit(const LinalgNegNode* node)         { expr = rewriter->rewriteNeg(node); }
  void visit(const LinalgTransposeNode* node)   { expr = rewriter->rewriteTranspose(node); }
  void visit(const LinalgLiteralNode* node)     { expr = rewriter->rewriteLiteral(node); }
  void visit(const LinalgVarNode* node)         { expr = rewriter->rewriteVar(node); }
  void visit(const LinalgTensorBaseNode* node)  { expr = rewriter->rewriteTensorBase(node); }
  void visit(const LinalgAssignmentNode* node)  { stmt = rewriter->rewriteAssignment(node); }

};

LinalgRewriter::LinalgRewriter() : visitor(new Visitor(this)) {
}

IndexExpr LinalgRewriter::rewriteSub(const LinalgSubNode* sub) {
  IndexExpr indexA = rewrite(sub->a);
  IndexExpr indexB = rewrite(sub->b);
  return new SubNode(indexA, indexB);
}

IndexExpr LinalgRewriter::rewriteAdd(const LinalgAddNode* add) {
  IndexExpr indexA = rewrite(add->a);
  IndexExpr indexB = rewrite(add->b);
  return new AddNode(indexA, indexB);
}

IndexExpr LinalgRewriter::rewriteElemMul(const LinalgElemMulNode* elemMul) {
  IndexExpr indexA = rewrite(elemMul->a);
  IndexExpr indexB = rewrite(elemMul->b);
  return new MulNode(indexA, indexB);
}

IndexExpr LinalgRewriter::rewriteMatMul(const LinalgMatMulNode *matMul) {
  IndexVar index = getUniqueIndex();
  vector<IndexVar> indicesA;
  vector<IndexVar> indicesB;
  if (matMul->a.getOrder() == 2 && matMul->b.getOrder() == 2) {
    indicesA = {liveIndices[0], index};
    indicesB = {index, liveIndices[1]};
  }
  else if (matMul->a.getOrder() == 1 && matMul->b.getOrder() == 2) {
    indicesA = {index};
    indicesB = {index, liveIndices[0]};
  }
  else if (matMul->a.getOrder() == 2 && matMul->b.getOrder() == 1) {
    indicesA = {liveIndices[0], index};
    indicesB = {index};
  }
  else if (matMul->a.getOrder() == 1 && matMul->a.isColVector() && matMul->b.getOrder() == 1) {
    indicesA = {liveIndices[0]};
    indicesB = {liveIndices[1]};
  } else if (matMul->a.getOrder() == 0) {
    indicesA = {};
    indicesB = liveIndices;
  } else if (matMul->b.getOrder() == 0) {
    indicesA = liveIndices;
    indicesB = {};
  } else {
    indicesA = {index};
    indicesB = {index};
  }
  liveIndices = indicesA;
  IndexExpr indexA = rewrite(matMul->a);
  liveIndices = indicesB;
  IndexExpr indexB = rewrite(matMul->b);
  return new MulNode(indexA, indexB);
}

IndexExpr LinalgRewriter::rewriteDiv(const LinalgDivNode *div) {
  IndexExpr indexA = rewrite(div->a);
  IndexExpr indexB = rewrite(div->b);
  return new DivNode(indexA, indexB);
}

IndexExpr LinalgRewriter::rewriteNeg(const LinalgNegNode *neg) {
  IndexExpr index = rewrite(neg->a);
  return new NegNode(index);
}

IndexExpr LinalgRewriter::rewriteTranspose(const LinalgTransposeNode *transpose) {
  if (transpose->a.getOrder() == 2) {
    liveIndices = {liveIndices[1], liveIndices[0]};
    return rewrite(transpose->a);
  }
  else if (transpose->a.getOrder() == 1) {
    liveIndices = {liveIndices[0]};
    return rewrite(transpose->a);
  }
  liveIndices = {};
  return rewrite(transpose->a);
}

IndexExpr LinalgRewriter::rewriteLiteral(const LinalgLiteralNode *lit) {
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
}

IndexExpr LinalgRewriter::rewriteVar(const LinalgVarNode *var) {
  return new AccessNode(var->tensorVar, liveIndices);
}

IndexExpr LinalgRewriter::rewriteTensorBase(const LinalgTensorBaseNode *node) {
  return node->tensorBase->operator()(liveIndices);
}

IndexVar LinalgRewriter::getUniqueIndex() {
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

IndexStmt LinalgRewriter::rewriteAssignment(const LinalgAssignmentNode *node) {
  return IndexStmt();
}

void LinalgRewriter::setLiveIndices(std::vector<IndexVar> indices) {
  liveIndices = indices;
}

IndexExpr LinalgRewriter::rewrite(LinalgExpr linalgExpr) {
  return visitor->rewrite(linalgExpr);
}

IndexStmt LinalgRewriter::rewrite(LinalgBase linalgBase) {
  TensorVar tensor = linalgBase.getAssignment().getLhs();

  vector<IndexVar> indices = {};
  if (tensor.getOrder() == 1) {
    indices.push_back(getUniqueIndex());
  } else if (tensor.getOrder() == 2) {
    indices.push_back(getUniqueIndex());
    indices.push_back(getUniqueIndex());
  }

  Access lhs = Access(tensor, indices);

  liveIndices = indices;
  auto rhs = rewrite(linalgBase.getAssignment().getRhs());

  if(linalgBase.tensorBase != nullptr) {
    linalgBase.tensorBase->operator()(indices) = rhs;
  }

  Assignment indexAssign = Assignment(lhs, rhs);
  return indexAssign;
}
