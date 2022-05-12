#include "taco/util/collections.h"
#include "taco/index_notation/iteration_algebra.h"
#include "taco/index_notation/iteration_algebra_printer.h"

namespace taco {

// Iteration Algebra Definitions

IterationAlgebra::IterationAlgebra() : IterationAlgebra(nullptr) {}
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

template <> bool isa<Region>(IterationAlgebra alg) {
  return isa<RegionNode>(alg.ptr);
}

template <> Region to<Region>(IterationAlgebra alg) {
  taco_iassert(isa<Region>(alg));
  return Region(to<RegionNode>(alg.ptr));
}

// Complement
Complement::Complement(const ComplementNode* n): IterationAlgebra(n) {
}

Complement::Complement(IterationAlgebra alg) : Complement(new ComplementNode(alg)) {
}


template <> bool isa<Complement>(IterationAlgebra alg) {
  return isa<ComplementNode>(alg.ptr);
}

template <> Complement to<Complement>(IterationAlgebra alg) {
  taco_iassert(isa<Complement>(alg));
  return Complement(to<ComplementNode>(alg.ptr));
}

// Intersect
Intersect::Intersect(IterationAlgebra a, IterationAlgebra b) : Intersect(new IntersectNode(a, b)) {}
Intersect::Intersect(const IterationAlgebraNode* n) : IterationAlgebra(n) {}

template <> bool isa<Intersect>(IterationAlgebra alg) {
  return isa<IntersectNode>(alg.ptr);
}

template <> Intersect to<Intersect>(IterationAlgebra alg) {
  taco_iassert(isa<Intersect>(alg));
  return Intersect(to<IntersectNode>(alg.ptr));
}

// Union
Union::Union(IterationAlgebra a, IterationAlgebra b) : Union(new UnionNode(a, b)) {}
Union::Union(const IterationAlgebraNode* n) : IterationAlgebra(n) {}

template <> bool isa<Union>(IterationAlgebra alg) {
  return isa<UnionNode>(alg.ptr);
}

template <> Union to<Union>(IterationAlgebra alg) {
  taco_iassert(isa<Union>(alg));
  return Union(to<UnionNode>(alg.ptr));
}


// Node method definitions start here:

// Definitions for RegionNode
void RegionNode::accept(IterationAlgebraVisitorStrict *v) const {
  v->visit(this);
}

const IndexExpr RegionNode::expr() const {
  return expr_;
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
    iter_alg = alg;
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

struct AlgComparer : public IterationAlgebraVisitorStrict {

  bool eq = false;
  IterationAlgebra bAlg;
  bool checkIndexExprs;

  explicit AlgComparer(bool checkIndexExprs) : checkIndexExprs(checkIndexExprs) {
  }

  bool compare(const IterationAlgebra& a, const IterationAlgebra& b) {
    bAlg = b;
    a.accept(this);
    return eq;
  }

  void visit(const RegionNode* node) {
    if(!isa<RegionNode>(bAlg.ptr)) {
      eq = false;
      return;
    }

    auto bnode = to<RegionNode>(bAlg.ptr);
    if (checkIndexExprs && !equals(node->expr(), bnode->expr())) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const ComplementNode* node) {
    if (!isa<ComplementNode>(bAlg.ptr)) {
      eq = false;
      return;
    }

    auto bNode = to<ComplementNode>(bAlg.ptr);
    eq = AlgComparer(checkIndexExprs).compare(node->a, bNode->a);
  }

  template<typename T>
  bool binaryCheck(const T* anode, IterationAlgebra b) {
    if (!isa<T>(b.ptr)) {
      return false;
    }
    auto bnode = to<T>(b.ptr);
    return AlgComparer(checkIndexExprs).compare(anode->a, bnode->a) &&
           AlgComparer(checkIndexExprs).compare(anode->b, bnode->b);
  }


  void visit(const IntersectNode* node) {
    eq = binaryCheck(node, bAlg);
  }

  void visit(const UnionNode* node) {
    eq = binaryCheck(node, bAlg);
  }

};

bool algStructureEqual(const IterationAlgebra& a, const IterationAlgebra& b) {
  return AlgComparer(false).compare(a, b);
}

bool algEqual(const IterationAlgebra& a, const IterationAlgebra& b) {
  return AlgComparer(true).compare(a, b);
}

class DeMorganApplier : public IterationAlgebraRewriterStrict {

  void visit(const RegionNode* n) {
    alg = Complement(n);
  }

  void visit(const ComplementNode* n) {
    alg = applyDemorgan(n->a);
  }

  template<typename Node, typename ComplementedNode>
  IterationAlgebra binaryVisit(Node n) {
    IterationAlgebra a = applyDemorgan(Complement(n->a));
    IterationAlgebra b = applyDemorgan(Complement(n->b));
    return new ComplementedNode(a, b);
  }

  void visit(const IntersectNode* n) {
    alg = binaryVisit<decltype(n), UnionNode>(n);
  }

  void visit(const UnionNode* n) {
    alg = binaryVisit<decltype(n), IntersectNode>(n);
  }
};

struct DeMorganDispatcher : public IterationAlgebraRewriter {

  using IterationAlgebraRewriter::visit;

  void visit(const ComplementNode *n) {
    alg = DeMorganApplier().rewrite(n->a);
  }
};

IterationAlgebra applyDemorgan(IterationAlgebra alg) {
  return DeMorganDispatcher().rewrite(alg);
}

class IndexExprReplacer : public IterationAlgebraRewriter {

public:
  IndexExprReplacer(const std::map<IndexExpr, IndexExpr>& substitutions) : substitutions(substitutions) {
  }

private:
  using IterationAlgebraRewriter::visit;

  void visit(const RegionNode* node) {
    if (util::contains(substitutions, node->expr())) {
      alg = new RegionNode(substitutions.at(node->expr()));
      return;
    }
    alg = node;
  }

  const std::map<IndexExpr, IndexExpr> substitutions;
};

IterationAlgebra replaceAlgIndexExprs(IterationAlgebra alg, const std::map<IndexExpr, IndexExpr>& substitutions) {
  return IndexExprReplacer(substitutions).rewrite(alg);
}

}