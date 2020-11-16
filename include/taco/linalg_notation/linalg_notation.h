//
// Created by Olivia Hsu on 10/30/20.
//

#ifndef TACO_LINALG_NOTATION_H
#define TACO_LINALG_NOTATION_H
#include <ostream>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <utility>

#include "taco/format.h"
#include "taco/error.h"
#include "taco/util/intrusive_ptr.h"
#include "taco/util/comparable.h"
#include "taco/type.h"
#include "taco/ir/ir.h"
#include "taco/codegen/module.h"

#include "taco/ir_tags.h"
#include "taco/lower/iterator.h"
#include "taco/index_notation/provenance_graph.h"

#include "taco/linalg_notation/linalg_notation_nodes_abstract.h"
#include "taco/linalg.h"

#include "taco/tensor.h"

namespace taco {

class Type;

class Dimension;

class Format;

class Schedule;

class TensorVar;

class LinalgBase;

class LinalgExpr;

class LinalgAssignment;

class Access;

struct LinalgVarNode;
struct LinalgLiteralNode;
struct LinalgNegNode;
struct LinalgTransposeNode;
struct LinalgAddNode;
struct LinalgSubNode;
struct LinalgMatMulNode;
struct LinalgElemMulNode;
struct LinalgDivNode;
struct LinalgUnaryExprNode;
struct LinalgBinaryExprNode;

class LinalgExprVisitorStrict;


class LinalgExpr : public util::IntrusivePtr<const LinalgExprNode> {
public:
  LinalgExpr() : util::IntrusivePtr<const LinalgExprNode>(nullptr) {}

  LinalgExpr(const LinalgExprNode *n) : util::IntrusivePtr<const LinalgExprNode>(n) {}

  /// Construct a scalar tensor access.
  /// ```
  /// A(i,j) = b;
  /// ```
  explicit LinalgExpr(TensorVar);

  LinalgExpr(TensorVar, bool isColVec, TensorBase* tensorBase);

  explicit LinalgExpr(TensorBase* _tensorBase, bool isColVec=false);

  LinalgExpr(TensorVar var, bool isColVec);
  /// Consturct an integer literal.
  /// ```
  /// A(i,j) = 1;
  /// ```
  LinalgExpr(char);

  LinalgExpr(int8_t);

  LinalgExpr(int16_t);

  LinalgExpr(int32_t);

  LinalgExpr(int64_t);

  /// Consturct an unsigned integer literal.
  /// ```
  /// A(i,j) = 1u;
  /// ```
  LinalgExpr(uint8_t);

  LinalgExpr(uint16_t);

  LinalgExpr(uint32_t);

  LinalgExpr(uint64_t);

  /// Consturct double literal.
  /// ```
  /// A(i,j) = 1.0;
  /// ```
  LinalgExpr(float);

  LinalgExpr(double);

  /// Construct complex literal.
  /// ```
  /// A(i,j) = complex(1.0, 1.0);
  /// ```
  LinalgExpr(std::complex<float>);

  LinalgExpr(std::complex<double>);

  Datatype getDataType() const;
  int getOrder() const;
  bool isColVector() const;
  void setColVector(bool) const;

  /// Visit the linalg expression's sub-expressions.
  void accept(LinalgExprVisitorStrict *) const;

  /// Print the index expression.
  friend std::ostream &operator<<(std::ostream &, const LinalgExpr &);

  TensorBase *tensorBase;
};

/// Compare two index expressions by value.
bool equals(LinalgExpr, LinalgExpr);

/// Construct and returns an expression that negates this expression.
LinalgExpr operator-(const LinalgExpr&);

/// Add two linear algebra expressions.
LinalgExpr operator+(const LinalgExpr&, const LinalgExpr&);

/// Subtract a linear algebra expressions from another.
LinalgExpr operator-(const LinalgExpr&, const LinalgExpr&);

/// Matrix Multiply two linear algebra expressions.
LinalgExpr operator*(const LinalgExpr&, const LinalgExpr&);

/// Divide a linear expression by another.
LinalgExpr operator/(const LinalgExpr&, const LinalgExpr&);

/// Element-wise multiply two linear algebra expressions
// FIXME: May want to be consistent with eigen library in c++ and change to cmul
LinalgExpr elemMul(const LinalgExpr& lhs, const LinalgExpr& rhs);

/// Construct and returns an expression that transposes this expression
// FIXME: May want to change this with '^T' in the future
LinalgExpr transpose(const LinalgExpr& lhs);
//LinalgExpr operator^(const LinalgExpr&, const T);

/// Check to make sure operators are legal (shape-wise)
int getMatMulOrder(const LinalgExpr &lhs, const LinalgExpr &rhs);

void checkCompatibleShape(const LinalgExpr &lhs, const LinalgExpr &rhs);
/// A an index statement computes a tensor.  The index statements are
/// assignment, forall, where, multi, and sequence.
class LinalgStmt : public util::IntrusivePtr<const LinalgStmtNode> {
public:
  LinalgStmt();
  LinalgStmt(const LinalgStmtNode* n);

  /// Visit the tensor expression
  void accept(LinalgStmtVisitorStrict *) const;
};

class LinalgAssignment : public LinalgStmt {
public:
  LinalgAssignment() = default;
  LinalgAssignment(const LinalgAssignmentNode*);

  /// Create an assignment.
  LinalgAssignment(TensorVar lhs, LinalgExpr rhs);

  /// Return the assignment's left-hand side.
  TensorVar getLhs() const;

  /// Return the assignment's right-hand side.
  LinalgExpr getRhs() const;

  typedef LinalgAssignmentNode Node;
};

}

#endif //TACO_LINALG_NOTATION_H
