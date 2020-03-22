#ifndef TACO_ITERATION_ALGEBRA_H
#define TACO_ITERATION_ALGEBRA_H

#include "taco/index_notation/index_notation.h"
#include "taco/util/uncopyable.h"
#include "taco/util/comparable.h"
#include "taco/util/intrusive_ptr.h"

namespace taco {

class IterationAlgebraVisitorStrict;
class IndexExpr;

struct IterationAlgebraNode;
struct RegionNode;
struct ComplementNode;
struct IntersectNode;
struct UnionNode;

/// The iteration algebra class describes a set expression composed of complements, intersections and unions on
/// IndexExprs to describe the spaces in a Venn Diagram where computation will occur.
/// This algebra is used to generate merge lattices to co-iterate over tensors in an expression.
class IterationAlgebra : public util::IntrusivePtr<const IterationAlgebraNode> {
public:
  IterationAlgebra();
  IterationAlgebra(const IterationAlgebraNode* n);
  IterationAlgebra(IndexExpr expr);

  void accept(IterationAlgebraVisitorStrict* v) const;
};

std::ostream& operator<<(std::ostream&, const IterationAlgebra&);

/// A region in an Iteration space. Given a Tensor A, this produces values everywhere the tensorVar or access is defined.
class Region: public IterationAlgebra {
public:
  Region();
  Region(IndexExpr expr);
  Region(const RegionNode*);
};

/// This complements an iteration space algebra expression. Thus, it will flip the segments that are produced and
/// omitted in the input segment.
/// Example: Given a segment A which produces values where A is defined and omits values outside of A,
///          complement(A) will not compute where A is defined but compute over the background of A.
class Complement: public IterationAlgebra {
public:
  Complement(IterationAlgebra alg);
  Complement(const ComplementNode* n);
};

/// This intersects two iteration space algebra expressions. This instructs taco to compute over areas where BOTH
/// set expressions produce values and ignore all other segments.
///
/// Examples
///
/// Given two tensors A and B:
/// Intersect(A, B) will produce values where both A and B are defined. An example of an operation with this property
/// is multiplication where both tensors are sparse over 0.
///
/// Intersect(Complement(A), B) will produce values where only B is defined. This pattern can be useful for filtering
/// one tensor based on the values of another.
class Intersect: public IterationAlgebra {
public:
  Intersect(IterationAlgebra, IterationAlgebra);
  Intersect(const IterationAlgebraNode*);
};

/// This takes the union of two iteration space algebra expressions. This instructs taco to compute over areas where
/// either set expression produces a value or both set expressions produce a value and ignore all other segments.
///
/// Examples
///
/// Given two tensors A and B:
/// Union(A, B) will produce values where either A or B is defined. Addition is an example of a union operator.
///
/// Union(Complement(A), B) will produce values wherever A is not defined. In the places A is not defined, the compiler
/// will replace the value of A in the indexExpression with the fill value of the tensor A. Likewise, when B is not
/// defined, the compiler will replace the value of B in the index expression with the fill value of B.
class Union: public IterationAlgebra {
public:
  Union(IterationAlgebra, IterationAlgebra);
  Union(const IterationAlgebraNode*);
};

/// A node in the iteration space algebra
struct IterationAlgebraNode: public util::Manageable<IterationAlgebraNode>,
                             private util::Uncopyable {
public:
  IterationAlgebraNode() {}

  virtual ~IterationAlgebraNode() = default;
  virtual void accept(IterationAlgebraVisitorStrict*) const = 0;
};

/// A binary node in the iteration space algebra. Used for Unions and Intersects
struct BinaryIterationAlgebraNode: public IterationAlgebraNode  {
  IterationAlgebra a;
  IterationAlgebra b;
protected:
  BinaryIterationAlgebraNode(IterationAlgebra a, IterationAlgebra b) : IterationAlgebraNode(), a(a), b(b) {}
};

/// A node which is wrapped by Region. @see Region
struct RegionNode: public IterationAlgebraNode {
public:
  RegionNode() : IterationAlgebraNode() {}
  RegionNode(IndexExpr expr) : IterationAlgebraNode(), expr_(expr) {}
  void accept(IterationAlgebraVisitorStrict*) const;
  const IndexExpr expr() const;
private:
  IndexExpr expr_;
};

/// A node which is wrapped by Complement. @see Complement
struct ComplementNode: public IterationAlgebraNode {
  IterationAlgebra a;
public:
  ComplementNode(IterationAlgebra a) : IterationAlgebraNode(), a(a) {}

  void accept(IterationAlgebraVisitorStrict*) const;
};

/// A node which is wrapped by Intersect. @see Intersect
struct IntersectNode: public BinaryIterationAlgebraNode {
public:
  IntersectNode(IterationAlgebra a, IterationAlgebra b) : BinaryIterationAlgebraNode(a, b) {}

  void accept(IterationAlgebraVisitorStrict*) const;

  const std::string algebraString() const;
};

/// A node which is wrapped by Union. @see Union
struct UnionNode: public BinaryIterationAlgebraNode {
public:
  UnionNode(IterationAlgebra a, IterationAlgebra b) : BinaryIterationAlgebraNode(a, b) {}

  void accept(IterationAlgebraVisitorStrict*) const;

  const std::string algebraString() const;
};

/// Visits an iteration space algebra expression
class IterationAlgebraVisitorStrict {
public:
  virtual ~IterationAlgebraVisitorStrict() {}
  void visit(const IterationAlgebra& alg);

  virtual void visit(const RegionNode*) = 0;
  virtual void visit(const ComplementNode*) = 0;
  virtual void visit(const IntersectNode*) = 0;
  virtual void visit(const UnionNode*) = 0;
};

// Default Iteration Algebra visitor
class IterationAlgebraVisitor : public IterationAlgebraVisitorStrict {
public:
  virtual ~IterationAlgebraVisitor() {}
  using IterationAlgebraVisitorStrict::visit;

  virtual void visit(const RegionNode* n);
  virtual void visit(const ComplementNode*);
  virtual void visit(const IntersectNode*);
  virtual void visit(const UnionNode*);
};

/// Rewrites an iteration algebra expression
class IterationAlgebraRewriterStrict : public IterationAlgebraVisitorStrict {
public:
  virtual ~IterationAlgebraRewriterStrict() {}
  IterationAlgebra rewrite(IterationAlgebra);

protected:
  /// Assign new algebra in visit method to replace the algebra nodes visited
  IterationAlgebra alg;

  using IterationAlgebraVisitorStrict::visit;

  virtual void visit(const RegionNode*) = 0;
  virtual void visit(const ComplementNode*) = 0;
  virtual void visit(const IntersectNode*) = 0;
  virtual void visit(const UnionNode*) = 0;
};

class IterationAlgebraRewriter : public IterationAlgebraRewriterStrict {
public:
  virtual ~IterationAlgebraRewriter() {}

protected:
  using IterationAlgebraRewriterStrict::visit;

  virtual void visit(const RegionNode* n);
  virtual void visit(const ComplementNode*);
  virtual void visit(const IntersectNode*);
  virtual void visit(const UnionNode*);
};

/// Returns true if algebra e is of type E.
template <typename E>
inline bool isa(const IterationAlgebraNode* e) {
  return e != nullptr && dynamic_cast<const E*>(e) != nullptr;
}

/// Casts the algebraNode e to type E.
template <typename E>
inline const E* to(const IterationAlgebraNode* e) {
  taco_iassert(isa<E>(e)) <<
                          "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
  return static_cast<const E*>(e);
}

/// Return true if the iteration algebra is of the given subtype.  The subtypes
/// are Region, Complement, Union and Intersect.
template <typename SubType> bool isa(IterationAlgebra);

/// Casts the iteration algebra to the given subtype. Assumes S is a subtype and
/// the subtypes are Region, Complement, Union and Intersect.
template <typename SubType> SubType to(IterationAlgebra);

/// Returns true if the structure of the iteration algebra is the same.
/// This means that intersections, unions, complements and regions appear
/// in the same places but the IndexExpressions these operations are applied
/// to are not necessarily the same.
bool algStructureEqual(const IterationAlgebra&, const IterationAlgebra&);

/// Returns true if the iterations algebras passed in have the same structure
/// and the Index Expressions that they operate on are the same.
bool algEqual(const IterationAlgebra&, const IterationAlgebra&);

/// Applies demorgan's laws to the algebra passed in and returns a new algebra
/// which describes the same space but with complements appearing only around region
/// nodes.
IterationAlgebra applyDemorgan(IterationAlgebra alg);

/// Rewrites the algebra to replace the IndexExprs in the algebra with new index exprs as
/// specified by the input map. If the map does not contain an indexExpr, it is kept the
/// same as the input algebra.
IterationAlgebra replaceAlgIndexExprs(IterationAlgebra alg, const std::map<IndexExpr, IndexExpr>&);
}


#endif // TACO_ITERATION_ALGEBRA_H
