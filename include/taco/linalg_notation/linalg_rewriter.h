#ifndef TACO_LINALG_REWRITER_H
#define TACO_LINALG_REWRITER_H

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <taco/index_notation/index_notation.h>

#include "taco/lower/iterator.h"
#include "taco/util/scopedset.h"
#include "taco/util/uncopyable.h"
#include "taco/ir_tags.h"

namespace taco {

class TensorVar;

class IndexVar;

class IndexExpr;

class LinalgBase;

class LinalgRewriter : public util::Uncopyable {
public:
  LinalgRewriter();

  virtual ~LinalgRewriter() = default;

  /// Lower an index statement to an IR function.
  IndexExpr rewrite(LinalgBase linalgBase);

//  void setLiveIndices(std::vector<IndexVar> indices);
protected:

  virtual IndexExpr rewriteSub(const LinalgSubNode* sub);

  virtual IndexExpr rewriteAdd(const LinalgAddNode* add);

  virtual IndexExpr rewriteElemMul(const LinalgElemMulNode* elemMul);

  virtual IndexExpr rewriteMatMul(const LinalgMatMulNode* matMul);

  virtual IndexExpr rewriteDiv(const LinalgDivNode* div);

  virtual IndexExpr rewriteNeg(const LinalgNegNode* neg);

  virtual IndexExpr rewriteTranspose(const LinalgTransposeNode* transpose);

  virtual IndexExpr rewriteLiteral(const LinalgLiteralNode* literal);

  virtual IndexExpr rewriteVar(const LinalgVarNode* var);

  virtual IndexExpr rewriteTensorBase(const LinalgTensorBaseNode* node);

  virtual IndexStmt rewriteAssignment(const LinalgAssignmentNode* node);

  IndexExpr rewrite(LinalgExpr linalgExpr);

private:
  std::vector<IndexVar> liveIndices;

  int idxcount;
  std::vector<std::string> indexVarNameList = {"i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"};

  IndexVar getUniqueIndex();

  class Visitor;
  friend class Visitor;
  std::shared_ptr<Visitor> visitor;
};

}   // namespace taco
#endif //TACO_LINALG_REWRITER_H
