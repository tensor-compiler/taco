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

namespace taco {

class Type;

class Dimension;

class Format;

class Schedule;

class TensorVar;

class LinalgExpr;

class Assignment;

class Access;

struct VarNode;
struct LiteralNode;
struct NegNode;
struct TransposeNode;
struct AddNode;
struct SubNode;
struct MatMulNode;
struct ElemMulNode;
struct DivNode;
struct UnaryExprNode;
struct BinaryExprNode;

class LinalgExprVisitorStrict;


class LinalgExpr : public util::IntrusivePtr<const LinalgExprNode> {
public:
  LinalgExpr() : util::IntrusivePtr<const LinalgExprNode>(nullptr) {}

  LinalgExpr(const LinalgExprNode *n) : util::IntrusivePtr<const LinalgExprNode>(n) {}

  /// Construct a scalar tensor access.
  /// ```
  /// A(i,j) = b;
  /// ```
  LinalgExpr(TensorVar);

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

  /// Visit the linalg expression's sub-expressions.
  void accept(LinalgExprVisitorStrict *) const;

  /// Print the index expression.
  friend std::ostream &operator<<(std::ostream &, const LinalgExpr &);
};

/// Compare two index expressions by value.
bool equals(LinalgExpr, LinalgExpr);

/// Construct and returns an expression that negates this expression.
LinalgExpr operator-(const LinalgExpr&);

/// Add two linear algebra expressions.
LinalgExpr operator+(const LinalgExpr&, const LinalgExpr&);

/// Subtract a linear algebra expressions from another.
LinalgExpr operator-(const LinalgExpr&, const LinalgExpr&);

/// Multiply two linear algebra expressions.
LinalgExpr operator*(const LinalgExpr&, const LinalgExpr&);

/// Divide a linear expression by another.
LinalgExpr operator/(const LinalgExpr&, const LinalgExpr&);

}

#endif //TACO_LINALG_NOTATION_H
