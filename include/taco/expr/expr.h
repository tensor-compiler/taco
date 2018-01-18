#ifndef TACO_EXPR_H
#define TACO_EXPR_H

#include <ostream>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <map>

#include "taco/error.h"
#include "taco/util/intrusive_ptr.h"
#include "taco/util/comparable.h"
#include "taco/util/uncopyable.h"

namespace taco {
class Type;
class Dimension;
class Format;

class IndexExpr;
class TensorVar;
class IndexVar;
class Schedule;
class OperatorSplit;
class ExprVisitorStrict;
struct AccessNode;


/// A node of an index expression tree.
struct ExprNode : public util::Manageable<ExprNode>, private util::Uncopyable {
public:
  ExprNode();
  virtual ~ExprNode() = default;
  virtual void accept(ExprVisitorStrict*) const = 0;
  virtual void print(std::ostream& os) const = 0;

  /// Split the expression.
  void splitOperator(IndexVar old, IndexVar left, IndexVar right);

  /// Returns the expression's operator splits.
  const std::vector<OperatorSplit>& getOperatorSplits() const;

private:
  std::shared_ptr<std::vector<OperatorSplit>> operatorSplits;
};


/// A tensor index expression describes a tensor computation as a scalar
/// expression where tensors are indexed by index variables (`IndexVar`).  The
/// index variables range over the tensor dimensions they index, and the scalar
/// expression is evaluated at every point in the resulting iteration space.
/// Index variables that are not used to index the result/left-hand-side are
/// called summation variables and are summed over. Some examples:
///
/// ```
/// // Matrix addition
/// A(i,j) = B(i,j) + C(i,j);
///
/// // Tensor addition (order-3 tensors)
/// A(i,j,k) = B(i,j,k) + C(i,j,k);
///
/// // Matrix-vector multiplication
/// a(i) = B(i,j) * c(j);
///
/// // Tensor-vector multiplication (order-3 tensor)
/// A(i,j) = B(i,j,k) * c(k);
///
/// // Matricized tensor times Khatri-Rao product (MTTKRP) from data analytics
/// A(i,j) = B(i,k,l) * C(k,j) * D(l,j);
/// ```
///
/// @see IndexVar Index into index expressions.
/// @see TensorVar Operands of index expressions.
class IndexExpr : public util::IntrusivePtr<const ExprNode> {
public:
  typedef ExprNode Node;

  IndexExpr() : util::IntrusivePtr<const ExprNode>(nullptr) {}
  IndexExpr(const ExprNode* n) : util::IntrusivePtr<const ExprNode>(n) {}

  /// Consturcts an integer literal.
  /// ```
  /// A(i,j) = 1;
  /// ```
  IndexExpr(int);

  /// Consturcts double literal.
  /// ```
  /// A(i,j) = 1.0;
  /// ```
  IndexExpr(double);

  /// Consturcts float literal.
  /// ```
  /// A(i,j) = 1.0f;
  /// ```
  IndexExpr(float);

  /// Constructs and returns an expression that negates this expression.
  /// ```
  /// A(i,j) = -B(i,j);
  /// ```
  IndexExpr operator-();

  /// Split the given index variable `old` into two index variables, `left` and
  /// `right`, at this expression.  This operation only has an effect for binary
  /// expressions. The `left` index variable computes the left-hand-side of the
  /// expression and stores the result in a temporary workspace. The `right`
  /// index variable computes the whole expression, substituting the
  /// left-hand-side for the workspace.
  void splitOperator(IndexVar old, IndexVar left, IndexVar right);

  /// Add two index expressions.
  /// ```
  /// A(i,j) = B(i,j) + C(i,j);
  /// ```
  friend IndexExpr operator+(const IndexExpr&, const IndexExpr&);

  /// Subtract an index expressions from another.
  /// ```
  /// A(i,j) = B(i,j) - C(i,j);
  /// ```
  friend IndexExpr operator-(const IndexExpr&, const IndexExpr&);

  /// Multiply two index expressions.
  /// ```
  /// A(i,j) = B(i,j) * C(i,j);  // Component-wise multiplication
  /// ```
  friend IndexExpr operator*(const IndexExpr&, const IndexExpr&);

  /// Divide an index expression by another.
  /// ```
  /// A(i,j) = B(i,j) / C(i,j);  // Component-wise division
  /// ```
  friend IndexExpr operator/(const IndexExpr&, const IndexExpr&);

  /// Returns the schedule of the index expression.
  const Schedule& getSchedule() const;

  /// Visit the index expression's sub-expressions.
  void accept(ExprVisitorStrict *) const;

  /// Print the index expression.
  friend std::ostream& operator<<(std::ostream&, const IndexExpr&);
};

/// Compare two expressions by value.
bool equals(IndexExpr, IndexExpr);


/// An index expression that represents a tensor access, such as `A(i,j))`.
/// Access expressions are returned when calling the overloaded operator() on
/// a `TensorVar`.  Access expressions can also be assigned an expression, which
/// happens when they occur on the left-hand-side of an assignment.
///
/// @see TensorVar Calling `operator()` on a `TensorVar` returns an `Assign`.
class Access : public IndexExpr {
public:
  typedef AccessNode Node;

  Access() = default;
  Access(const Node* n);
  Access(const TensorVar& tensorVar, const std::vector<IndexVar>& indices={});

  /// Return the Access expression's TensorVar.
  const TensorVar &getTensorVar() const;

  /// Returns the index variables used to index into the Access's TensorVar.
  const std::vector<IndexVar>& getIndexVars() const;

  /// Assign the result of an expression to a left-hand-side tensor access.
  void operator=(const IndexExpr&);
  void operator=(const Access&);

  /// Accumulate the result of an expression to a left-hand-side tensor access.
  /// ```
  /// a(i) += B(i,j) * c(j);
  /// ```
  void operator+=(const IndexExpr&);
  void operator+=(const Access&);

private:
  const Node* getPtr() const;
};


/// Index variables are used to index into tensors in index expressions, and
/// they represent iteration over the tensor modes they index into.
class IndexVar : public util::Comparable<IndexVar> {
public:
  IndexVar();
  IndexVar(const std::string& name);

  /// Returns the name of the index variable.
  std::string getName() const;

  friend bool operator==(const IndexVar&, const IndexVar&);
  friend bool operator<(const IndexVar&, const IndexVar&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const IndexVar&);


/// A tensor variable in an index expression, which can either be an operand
/// or the result of the expression.
class TensorVar : public util::Comparable<TensorVar> {
public:
  TensorVar();
  TensorVar(const Type& type);
  TensorVar(const std::string& name, const Type& type);
  TensorVar(const Type& type, const Format& format);
  TensorVar(const std::string& name, const Type& type, const Format& format);

  /// Returns the name of the tensor variable.
  std::string getName() const;

  /// Returns the order of the tensor (number of modes).
  int getOrder() const;

  /// Returns the type of the tensor variable.
  const Type& getType() const;

  /// Returns the format of the tensor variable.
  const Format& getFormat() const;

  /// Returns the free variables used to access this variable on the
  /// left-hand-side of the expression
  const std::vector<IndexVar>& getFreeVars() const;

  /// Returns the right-hand-side of the expression that computes the tensor,
  /// which is undefined if the tensor is not computed.
  const IndexExpr& getIndexExpr() const;

  /// Returns true if the result of the index expression is accumulated into
  /// the var and false if it is stored.
  bool isAccumulating() const;

  /// Returns the schedule of the tensor var, which describes how to compile
  /// and execute it's expression.
  const Schedule& getSchedule() const;

  /// Assign an index expression to the tensor var, with the given free vars
  /// denoting the indexing on the left-hand-side.
  void setIndexExpression(std::vector<IndexVar> freeVars, IndexExpr indexExpr,
                          bool accumulate=false);

  /// Create an index expression that accesses (reads) this tensor.
  const Access operator()(const std::vector<IndexVar>& indices) const;

  /// Create an index expression that accesses (reads or writes) this tensor.
  Access operator()(const std::vector<IndexVar>& indices);

  /// Create an index expression that accesses (reads) this tensor.
  template <typename... IndexVars>
  const Access operator()(const IndexVars&... indices) const {
    return static_cast<const TensorVar*>(this)->operator()({indices...});
  }

  /// Create an index expression that accesses (reads or writes) this tensor.
  template <typename... IndexVars>
  Access operator()(const IndexVars&... indices) {
    return this->operator()({indices...});
  }

  friend bool operator==(const TensorVar&, const TensorVar&);
  friend bool operator<(const TensorVar&, const TensorVar&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const TensorVar&);
std::set<IndexVar> getIndexVars(const TensorVar&);
std::map<IndexVar,Dimension> getIndexVarRanges(const TensorVar&);


/// Simplify an expression by setting the `exhausted` IndexExprs to zero.
IndexExpr simplify(const IndexExpr& expr, const std::set<Access>& exhausted);
  
}
#endif
