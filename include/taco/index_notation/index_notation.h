#ifndef TACO_INDEX_NOTATION_H
#define TACO_INDEX_NOTATION_H

#include <ostream>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <map>

#include "taco/error.h"
#include "taco/util/intrusive_ptr.h"
#include "taco/util/comparable.h"
#include "taco/type.h"

#include "taco/index_notation/index_notation_nodes_abstract.h"

namespace taco {

class Type;
class Dimension;
class Format;
class Schedule;

class TensorVar;
class IndexVar;

class IndexExpr;
class Assignment;
class Access;

struct AccessNode;
struct ReductionNode;

struct AssignmentNode;
struct ForallNode;
struct WhereNode;

/// A tensor index expression describes a tensor computation as a scalar
/// expression where tensors are indexed by index variables (`IndexVar`).  The
/// index variables range over the tensor dimensions they index, and the scalar
/// expression is evaluated at every point in the resulting iteration space.
/// Index variables that are not used to index the result/left-hand-side are
/// called summation variables and are summed over. Some examples:
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

  /// Construct a scalar tensor access.
  /// ```
  /// A(i,j) = b;
  /// ```
  IndexExpr(TensorVar);

  /// Consturct an integer literal.
  /// ```
  /// A(i,j) = 1;
  /// ```
  IndexExpr(long long);

  /// Consturct an unsigned integer literal.
  /// ```
  /// A(i,j) = 1u;
  /// ```
  IndexExpr(unsigned long long);

  /// Consturct double literal.
  /// ```
  /// A(i,j) = 1.0;
  /// ```
  IndexExpr(double);

  /// Construct complex literal.
  /// ```
  /// A(i,j) = complex(1.0, 1.0);
  /// ```
  IndexExpr(std::complex<double>);

  /// Split the given index variable `old` into two index variables, `left` and
  /// `right`, at this expression.  This operation only has an effect for binary
  /// expressions. The `left` index variable computes the left-hand-side of the
  /// expression and stores the result in a temporary workspace. The `right`
  /// index variable computes the whole expression, substituting the
  /// left-hand-side for the workspace.
  void splitOperator(IndexVar old, IndexVar left, IndexVar right);

  DataType getDataType() const;
  
  /// Returns the schedule of the index expression.
  const Schedule& getSchedule() const;

  /// Visit the index expression's sub-expressions.
  void accept(IndexExprVisitorStrict *) const;

  /// Print the index expression.
  friend std::ostream& operator<<(std::ostream&, const IndexExpr&);
};

/// Compare two expressions by value.
bool equals(IndexExpr, IndexExpr);

/// Construct and returns an expression that negates this expression.
/// ```
/// A(i,j) = -B(i,j);
/// ```
IndexExpr operator-(const IndexExpr&);

/// Add two index expressions.
/// ```
/// A(i,j) = B(i,j) + C(i,j);
/// ```
IndexExpr operator+(const IndexExpr&, const IndexExpr&);

/// Subtract an index expressions from another.
/// ```
/// A(i,j) = B(i,j) - C(i,j);
/// ```
IndexExpr operator-(const IndexExpr&, const IndexExpr&);

/// Multiply two index expressions.
/// ```
/// A(i,j) = B(i,j) * C(i,j);  // Component-wise multiplication
/// ```
IndexExpr operator*(const IndexExpr&, const IndexExpr&);

/// Divide an index expression by another.
/// ```
/// A(i,j) = B(i,j) / C(i,j);  // Component-wise division
/// ```
IndexExpr operator/(const IndexExpr&, const IndexExpr&);


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
  /// ```
  /// a(i) = b(i) * c(i);
  /// ```
  Assignment operator=(const IndexExpr&);

  /// Must override the default Access operator=, otherwise it is a copy.
  Assignment operator=(const Access&);

  /// Must disambiguate TensorVar as it can be implicitly converted to IndexExpr
  /// and AccesExpr.
  Assignment operator=(const TensorVar&);

  /// Accumulate the result of an expression to a left-hand-side tensor access.
  /// ```
  /// a(i) += B(i,j) * c(j);
  /// ```
  Assignment operator+=(const IndexExpr&);

private:
  const Node* getPtr() const;
};


/// A reduction over the components indexed by the reduction variable.
class Reduction : public IndexExpr {
public:
  typedef ReductionNode Node;

  Reduction(const Node*);
  Reduction(IndexExpr op, IndexVar var, IndexExpr expr);

private:
  const Node* getPtr();
};


/// A an index statement computes a tensor.  The index statements are assignment
/// forall, and where.
class IndexStmt : public util::IntrusivePtr<const IndexStmtNode> {
public:
  IndexStmt();
  IndexStmt(const IndexStmtNode* n);

  /// Visit the tensor expression
  void accept(IndexNotationVisitorStrict *) const;
};

std::ostream& operator<<(std::ostream&, const IndexStmt&);


/// An assignment statement assigns an index expression to the locations in a
/// tensor given by an lhs access expression.
class Assignment : public IndexStmt {
public:
  Assignment(const AssignmentNode*);

  /// Create an assignment. Can specify an optional operator `op` that turns the
  /// assignment into a compound assignment, e.g. `+=`.
  Assignment(TensorVar tensor, std::vector<IndexVar> indices, IndexExpr expr,
             IndexExpr op = IndexExpr());

  Access getLhs() const;
  IndexExpr getRhs() const;

private:
  const AssignmentNode* getPtr() const;
};


/// A forall statement binds an index variable to values and evaluates the
/// sub-statement for each of these values.
class Forall : public IndexStmt {
public:
  Forall(const ForallNode*);
  Forall(IndexVar indexVar, IndexStmt stmt);

  IndexVar getIndexVar() const;
  IndexStmt getStmt() const;

private:
  const ForallNode* getPtr() const;
};


/// A where statment has a producer statement that binds a tensor variable in
/// the environment of a consumer statement.
class Where : public IndexStmt {
public:
  Where(const WhereNode*);
  Where(IndexStmt consumer, IndexStmt producer);

  IndexStmt getConsumer();
  IndexStmt getProducer();

private:
  const WhereNode* getPtr() const;
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

  /// Set the name of the tensor variable.
  void setName(std::string name);

  /// Assign an index expression to the tensor var, with the given free vars
  /// denoting the indexing on the left-hand-side.
  void setIndexExpression(std::vector<IndexVar> freeVars, IndexExpr indexExpr,
                          bool accumulate=false);

  /// Create an index expression that accesses (reads) this tensor.
  const Access operator()(const std::vector<IndexVar>& indices) const;

  /// Create an index expression that accesses (reads) this tensor.
  template <typename... IndexVars>
  const Access operator()(const IndexVars&... indices) const {
    return static_cast<const TensorVar*>(this)->operator()({indices...});
  }

  /// Create an index expression that accesses (reads or writes) this tensor.
  Access operator()(const std::vector<IndexVar>& indices);

  /// Create an index expression that accesses (reads or writes) this tensor.
  template <typename... IndexVars>
  Access operator()(const IndexVars&... indices) {
    return this->operator()({indices...});
  }

  /// Assign an expression to a scalar tensor.
  Assignment operator=(const IndexExpr&);

  /// Add an expression to a scalar tensor.
  Assignment operator+=(const IndexExpr&);

  friend bool operator==(const TensorVar&, const TensorVar&);
  friend bool operator<(const TensorVar&, const TensorVar&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const TensorVar&);


Reduction sum(IndexVar i, IndexExpr expr);
Forall forall(IndexVar i, IndexStmt expr);
Where where(IndexStmt consumer, IndexStmt producer);

/// Get all index variables in the expression
std::vector<IndexVar> getIndexVars(const IndexExpr&);

/// Get all index variable used to compute the tensor.
std::set<IndexVar> getIndexVars(const TensorVar&);

std::map<IndexVar,Dimension> getIndexVarRanges(const TensorVar&);

/// Simplify an expression by setting the `zeroed` IndexExprs to zero.
IndexExpr simplify(const IndexExpr& expr, const std::set<Access>& zeroed);

std::set<IndexVar> getVarsWithoutReduction(const IndexExpr& expr);

/// Verify that the expression is well formed.
bool verify(const IndexExpr& expr, const std::vector<IndexVar>& free);

/// Verifies that the variable's expression is well formed.
bool verify(const TensorVar& var);

/// Verify that an expression is formatted so that we can apply Einstein's
/// summation convention, meaning a sum of products: a*...*b + ... + c*...*d
/// with no explicit reductions.
bool isEinsum(IndexExpr);

/// Apply Einstein's summation convention to the expression and return the
/// result, meaning non-free variables are summed over their term.  Returns an
/// undefined index expression if einsum does not apply to the expression.
IndexExpr einsum(const IndexExpr& expr, const std::vector<IndexVar>& free={});

/// Apply Einstein's summation convention to the var's expression and return the
/// result, meaning non-free variables are summed over their term.  Returns an
/// undefined index expression if einsum does not apply to the expression.
IndexExpr einsum(const TensorVar& var);

}
#endif
