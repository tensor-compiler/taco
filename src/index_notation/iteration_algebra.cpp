#include "taco/index_notation/iteration_algebra.h"
#include "taco/index_notation/iteration_algebra_printer.h"

namespace taco {

// Iteration Algebra Definitions

IterationAlgebra::IterationAlgebra() : util::IntrusivePtr<const IterationAlgebraNode>(nullptr) {}
IterationAlgebra::IterationAlgebra(const IterationAlgebraNode* n) : util::IntrusivePtr<const IterationAlgebraNode>(n) {}
IterationAlgebra::IterationAlgebra(IndexExpr expr) : IterationAlgebra(new RegionNode(expr)) {}

void IterationAlgebra::accept(IterationAlgebraVisitorStrict *v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const IterationAlgebra& algebra) {
  if(!algebra.defined()) return os << "{}";
  IterationAlgebraPrinter printer(os);
  printer.print(algebra);
  return os;
}

// Definitions for Iteration Algebra

// Region
Region::Region() : IterationAlgebra(new RegionNode) {}
Region::Region(IndexExpr expr) : IterationAlgebra(expr) {}
Region::Region(const taco::RegionNode *n) : IterationAlgebra(n) {}

// Complement
Complement::Complement(const ComplementNode* n): IterationAlgebra(n) {}
Complement::Complement(IterationAlgebra alg) : Complement(new ComplementNode(alg)) {}

// Intersect
Intersect::Intersect(IterationAlgebra a, IterationAlgebra b) : Intersect(new IntersectNode(a, b)) {}
Intersect::Intersect(const IterationAlgebraNode* n) : IterationAlgebra(n) {}

// Union
Union::Union(IterationAlgebra a, IterationAlgebra b) : Union(new UnionNode(a, b)) {}
Union::Union(const IterationAlgebraNode* n) : IterationAlgebra(n) {}

// Node method definitions start here:

// Definitions for RegionNode
void RegionNode::accept(IterationAlgebraVisitorStrict *v) const {
  v->visit(this);
}

const IndexExpr RegionNode::indexExpr() const {
  return expr;
}

// Definitions for ComplementNode
void ComplementNode::accept(IterationAlgebraVisitorStrict *v) const {
  v->visit(this);
}

// Definitions for IntersectNode
void IntersectNode::accept(IterationAlgebraVisitorStrict *v) const {
  v->visit(this);
}

const std::string IntersectNode::algebraString() const {
  return "*";
}

// Definitions for UnionNode
void UnionNode::accept(IterationAlgebraVisitorStrict *v) const {
  v->visit(this);
}

const std::string UnionNode::algebraString() const {
  return "U";
}

// Visitor definitions start here:

// IterationAlgebraVisitorStrict definitions
void IterationAlgebraVisitorStrict::visit(const IterationAlgebra &alg) {
  alg.accept(this);
}

// Default IterationAlgebraVisitor definitions
void IterationAlgebraVisitor::visit(const RegionNode *n) {
}

void IterationAlgebraVisitor::visit(const ComplementNode *n) {
  n->a.accept(this);
}

void IterationAlgebraVisitor::visit(const IntersectNode *n) {
  n->a.accept(this);
  n->b.accept(this);
}

void IterationAlgebraVisitor::visit(const UnionNode *n) {
  n->a.accept(this);
  n->b.accept(this);
}

// IterationAlgebraRewriter definitions start here:
IterationAlgebra IterationAlgebraRewriterStrict::rewrite(IterationAlgebra iter_alg) {
  if(iter_alg.defined()) {
    iter_alg.accept(this);
    alg = iter_alg;
  }
  else {
    iter_alg = IterationAlgebra();
  }

  alg = IterationAlgebra();
  return iter_alg;
}

// Default IterationAlgebraRewriter definitions
void IterationAlgebraRewriter::visit(const RegionNode *n) {
  alg = n;
}

void IterationAlgebraRewriter::visit(const ComplementNode *n) {
  IterationAlgebra a = rewrite(n->a);
  if(n-> a == a) {
    alg = n;
  } else {
    alg = new ComplementNode(a);
  }
}

void IterationAlgebraRewriter::visit(const IntersectNode *n) {
  IterationAlgebra a = rewrite(n->a);
  IterationAlgebra b = rewrite(n->b);

  if(n->a == a && n->b == b) {
    alg = n;
  } else {
    alg = new IntersectNode(a, b);
  }
}

void IterationAlgebraRewriter::visit(const UnionNode *n) {
  IterationAlgebra a = rewrite(n->a);
  IterationAlgebra b = rewrite(n->b);

  if(n->a == a && n->b == b) {
    alg = n;
  } else {
    alg = new UnionNode(a, b);
  }
}
}