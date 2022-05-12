#include "taco/index_notation/index_notation.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <utility>
#include <set>
#include <taco/ir/simplify.h>
#include "lower/mode_access.h"

#include "error/error_checks.h"
#include "taco/error/error_messages.h"
#include "taco/type.h"
#include "taco/format.h"

#include "taco/index_notation/properties.h"
#include "taco/index_notation/intrinsic.h"
#include "taco/index_notation/schedule.h"
#include "taco/index_notation/transformations.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_printer.h"
#include "taco/ir/ir.h"
#include "taco/codegen/module.h"
#include "taco/tensor.h"

#include "taco/util/name_generator.h"
#include "taco/util/scopedset.h"
#include "taco/util/scopedmap.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"
#include "taco/util/functions.h"
#include "taco/util/env.h"

using namespace std;

namespace taco {

// class IndexExpr
IndexExpr::IndexExpr(TensorVar var) 
    : IndexExpr(new AccessNode(var,{},{},false)) {
}

IndexExpr::IndexExpr(char val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(int8_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(int16_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(int32_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(int64_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(uint8_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(uint16_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(uint32_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(uint64_t val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(float val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(double val) : IndexExpr(new LiteralNode(val)) {
}

IndexExpr::IndexExpr(std::complex<float> val) :IndexExpr(new LiteralNode(val)){
}

IndexExpr::IndexExpr(std::complex<double> val) :IndexExpr(new LiteralNode(val)){
}

Datatype IndexExpr::getDataType() const {
  return const_cast<IndexExprNode*>(this->ptr)->getDataType();
}

void IndexExpr::workspace(IndexVar i, IndexVar iw, std::string name) {
//  const_cast<IndexExprNode*>(this->ptr)->splitOperator(i, i, iw);
}

void IndexExpr::workspace(IndexVar i, IndexVar iw, Format format, string name) {
//  const_cast<IndexExprNode*>(this->ptr)->splitOperator(i, i, iw);
}

void IndexExpr::workspace(IndexVar i, IndexVar iw, TensorVar workspace) {
//  const_cast<IndexExprNode*>(this->ptr)->splitOperator(i, i, iw);
//  const_cast<IndexExprNode*>(this->ptr)->workspace(i, iw, workspace);
  this->ptr->setWorkspace(i, iw, workspace);
}

void IndexExpr::accept(IndexExprVisitorStrict *v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const IndexExpr& expr) {
  if (!expr.defined()) return os << "IndexExpr()";
  IndexNotationPrinter printer(os);
  printer.print(expr);
  return os;
}

static bool checkRegionDefinitions(const CallNode* anode, const CallNode* bnode) {
  // Check region definitions
  if (anode->regionDefinitions.size() != bnode->regionDefinitions.size()) {
    return false;
  }

  auto& aDefs = anode->regionDefinitions;
  auto& bDefs = bnode->regionDefinitions;
  for (auto itA = aDefs.begin(), itB = bDefs.begin(); itA != aDefs.end(); ++itA, ++itB) {
    if(itA->first != itB->first) {
      return false;
    }

    std::vector<IndexExpr> aArgs;
    std::vector<IndexExpr> bArgs;
    for(int idx : itA->first) {
      taco_iassert((size_t)idx < anode->args.size()); // We already know anode->args.size == bnode->args.size
      aArgs.push_back(anode->args[idx]);
      bArgs.push_back(bnode->args[idx]);
    }

    // TODO lower and check IR
    if(!util::targetPtrEqual(itA->second, itB->second)) {
      return false;
    }
  }

  return true;
}

/// Checks if the iteration algebra structure is the same and the ordering of the index expressions
/// nested under regions is the same for each op node.
static bool checkIterationAlg(const CallNode* anode, const CallNode* bnode) {
  // Check IterationAlgebra structures
  if(!algStructureEqual(anode->iterAlg, bnode->iterAlg)) {
    return false;
  }

  struct OrderChecker : public IterationAlgebraVisitor {
    explicit OrderChecker(const CallNode* op) : op(op) {}

    std::vector<size_t>& check() {
      op->iterAlg.accept(this);
      return ordering;
    }

    using IterationAlgebraVisitor::visit;

    void visit(const RegionNode* region) {
      const IndexExpr& e = region->expr();
      auto it = std::find(op->args.begin(), op->args.end(), e);
      taco_iassert(it != op->args.end()) << "Iteration algebra region expressions must be in arguments";
      size_t loc = it - op->args.begin();
      ordering.push_back(loc);
    }

    std::vector<size_t> ordering;
    const CallNode* op;
  };

  std::vector<size_t> aOrdering = OrderChecker(anode).check();
  std::vector<size_t> bOrdering = OrderChecker(bnode).check();
  return aOrdering == bOrdering;
}

struct Isomorphic : public IndexNotationVisitorStrict {
  bool eq = false;
  IndexExpr bExpr;
  IndexStmt bStmt;
  std::map<TensorVar,TensorVar> isoATensor, isoBTensor;
  std::map<IndexVar,IndexVar> isoAVar, isoBVar;

  bool check(IndexExpr a, IndexExpr b) {
    if (!a.defined() && !b.defined()) {
      return true;
    }
    if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
      return false;
    }
    this->bExpr = b;
    a.accept(this);
    return eq;
  }

  bool check(IndexStmt a, IndexStmt b) {
    if (!a.defined() && !b.defined()) {
      return true;
    }
    if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
      return false;
    }
    this->bStmt = b;
    a.accept(this);
    return eq;
  }

  bool check(TensorVar a, TensorVar b) {
    if (!util::contains(isoBTensor, a) && !util::contains(isoATensor, b)) {
      if (a.getType() != b.getType() || a.getFormat() != b.getFormat()) {
        return false;
      }
      isoBTensor.insert({a, b});
      isoATensor.insert({b, a});
      return true;
    }
    if (!util::contains(isoBTensor, a) || !util::contains(isoATensor, b)) {
      return false;
    }
    return (isoBTensor[a] == b) && (isoATensor[b] == a);
  }

  bool check(IndexVar a, IndexVar b) {
    if (!util::contains(isoBVar, a) && !util::contains(isoAVar, b)) {
      isoBVar.insert({a, b});
      isoAVar.insert({b, a});
      return true;
    }
    if (!util::contains(isoBVar, a) || !util::contains(isoAVar, b)) {
      return false;
    }
    return (isoBVar[a] == b) && (isoAVar[b] == a);
  }

  using IndexNotationVisitorStrict::visit;

  void visit(const IndexVarNode* anode) {
    if(!isa<IndexVarNode>(bExpr.ptr)) {
      eq = false;
      return;
    }

    auto bnode = to<IndexVarNode>(bExpr.ptr);
    if(anode != bnode) {
      eq = false;
      return;
    }

    eq = true;
  }

  void visit(const AccessNode* anode) {
    if (!isa<AccessNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AccessNode>(bExpr.ptr);
    if (!check(anode->tensorVar, bnode->tensorVar)) {
      eq = false;
      return;
    }
    if (anode->indexVars.size() != bnode->indexVars.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->indexVars.size(); i++) {
      if (!check(anode->indexVars[i], bnode->indexVars[i])) {
        eq = false;
        return;
      }
    }
    if (anode->isAccessingStructure != bnode->isAccessingStructure ||
        anode->windowedModes != bnode->windowedModes) {
      eq = false;
      return;
    }
    if (anode->indexSetModes.size() != bnode->indexSetModes.size()) {
      eq = false;
      return;
    }
    for (auto aset = anode->indexSetModes.begin(), bset = bnode->indexSetModes.begin(); aset != anode->indexSetModes.end(); ++aset, ++bset) {
      if (aset->first != bset->first || *aset->second.set != *bset->second.set) {
        eq = false;
        return;
      }
    }
    eq = true;
  }

  void visit(const LiteralNode* anode) {
    if (!isa<LiteralNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<LiteralNode>(bExpr.ptr);
    if (anode->getDataType() != bnode->getDataType()) {
      eq = false;
      return;
    }
    if (memcmp(anode->val,bnode->val,anode->getDataType().getNumBytes()) != 0) {
      eq = false;
      return;
    }
    eq = true;
  }

  template <class T>
  bool unaryIsomorphic(const T* anode, IndexExpr b) {
    if (!isa<T>(b.ptr)) {
      return false;
    }
    auto bnode = to<T>(b.ptr);
    if (!check(anode->a, bnode->a)) {
      return false;
    }
    return true;
  }

  void visit(const NegNode* anode) {
    eq = unaryIsomorphic(anode, bExpr);
  }

  void visit(const SqrtNode* anode) {
    eq = unaryIsomorphic(anode, bExpr);
  }

  template <class T>
  bool binaryIsomorphic(const T* anode, IndexExpr b) {
    if (!isa<T>(b.ptr)) {
      return false;
    }
    auto bnode = to<T>(b.ptr);
    if (!check(anode->a, bnode->a) || !check(anode->b, bnode->b)) {
      return false;
    }
    return true;
  }

  void visit(const AddNode* anode) {
    eq = binaryIsomorphic(anode, bExpr);
  }

  void visit(const SubNode* anode) {
    eq = binaryIsomorphic(anode, bExpr);
  }

  void visit(const MulNode* anode) {
    eq = binaryIsomorphic(anode, bExpr);
  }

  void visit(const DivNode* anode) {
    eq = binaryIsomorphic(anode, bExpr);
  }

  void visit(const CastNode* anode) {
    if (!isa<CastNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CastNode>(bExpr.ptr);
    if (anode->getDataType() != bnode->getDataType() ||
        !check(anode->a, bnode->a)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const CallIntrinsicNode* anode) {
    if (!isa<CallIntrinsicNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CallIntrinsicNode>(bExpr.ptr);
    if (anode->func->getName() != bnode->func->getName() ||
        anode->args.size() != bnode->args.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->args.size(); ++i) {
      if (!check(anode->args[i], bnode->args[i])) {
        eq = false;
        return;
      }
    }
    eq = true;
  }

  void visit(const ReductionNode* anode) {
    if (!isa<ReductionNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<ReductionNode>(bExpr.ptr);
    if (!check(anode->op, bnode->op) ||
        !check(anode->var, bnode->var) ||
        !check(anode->a, bnode->a)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const AssignmentNode* anode) {
    if (!isa<AssignmentNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AssignmentNode>(bStmt.ptr);
    if (!check(anode->lhs, bnode->lhs) ||
        !check(anode->rhs, bnode->rhs) ||
        !check(anode->op, bnode->op)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const YieldNode* anode) {
    if (!isa<YieldNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<YieldNode>(bStmt.ptr);
    if (anode->indexVars.size() != bnode->indexVars.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->indexVars.size(); i++) {
      if (!check(anode->indexVars[i], bnode->indexVars[i])) {
        eq = false;
        return;
      }
    }
    if (!check(anode->expr, bnode->expr)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const ForallNode* anode) {
    if (!isa<ForallNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<ForallNode>(bStmt.ptr);
    if (!check(anode->indexVar, bnode->indexVar) ||
        !check(anode->stmt, bnode->stmt) ||
        anode->parallel_unit != bnode->parallel_unit ||
        anode->output_race_strategy != bnode->output_race_strategy ||
        anode->unrollFactor != bnode->unrollFactor) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const WhereNode* anode) {
    if (!isa<WhereNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<WhereNode>(bStmt.ptr);
    if (!check(anode->consumer, bnode->consumer) ||
        !check(anode->producer, bnode->producer)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const SequenceNode* anode) {
    if (!isa<SequenceNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<SequenceNode>(bStmt.ptr);
    if (!check(anode->definition, bnode->definition) ||
        !check(anode->mutation, bnode->mutation)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const AssembleNode* anode) {
    if (!isa<AssembleNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AssembleNode>(bStmt.ptr);
    if (!check(anode->queries, bnode->queries) ||
        !check(anode->compute, bnode->compute)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const MultiNode* anode) {
    if (!isa<MultiNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<MultiNode>(bStmt.ptr);
    if (!check(anode->stmt1, bnode->stmt1) ||
        !check(anode->stmt2, bnode->stmt2)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const SuchThatNode* anode) {
    if (!isa<SuchThatNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<SuchThatNode>(bStmt.ptr);
    if (!check(anode->stmt, bnode->stmt) ||
         anode->predicate != bnode->predicate) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const CallNode* anode) {
    if (!isa<CallNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CallNode>(bExpr.ptr);

    // Properties
    if (anode->properties.size() != bnode->properties.size()) {
      eq = false;
      return;
    }

    for(const auto& a_prop : anode->properties) {
      bool found = false;
      for(const auto& b_prop : bnode->properties) {
        if(a_prop.equals(b_prop)) {
          found = true;
          break;
        }
      }
      if (!found) {
        eq = false;
        return;
      }
    }

    // Exhausted regions
    if (anode->definedRegions != bnode->definedRegions) {
      eq = false;
      return;
    }

    // Lower function
    // TODO: For now just check that the function pointers are the same.
    // TODO (rawnh): This check is broken. The retrieved function pointers are null
    //  when attempting to dereference them. The original code attempted to use
    //  util::targetPtrEqual.
    if (util::getFromEnv("TACO_ISOMORPHIC_HACK", "0") == "0") {
      if (&anode->defaultLowerFunc != &bnode->defaultLowerFunc) {
        eq = false;
        return;
      }
    } else {
      // If the hack is enabled, check that names are the same.
      if (anode->name != bnode->name) {
        eq = false;
        return;
      }
    }

    // Check arguments
    if (anode->args.size() != bnode->args.size()) {
      eq = false;
      return;
    }

    for (size_t i = 0; i < anode->args.size(); ++i) {
      if (!check(anode->args[i], bnode->args[i])) {
        eq = false;
        return;
      }
    }

    // Algebra
    if (!checkIterationAlg(anode, bnode)) {
      eq = false;
      return;
    }

    // Special definitions
    eq = checkRegionDefinitions(anode, bnode);
  }
};

bool isomorphic(IndexExpr a, IndexExpr b) {
  if (!a.defined() && !b.defined()) {
    return true;
  }
  if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
    return false;
  }
  return Isomorphic().check(a,b);
}

bool isomorphic(IndexStmt a, IndexStmt b) {
  if (!a.defined() && !b.defined()) {
    return true;
  }
  if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
    return false;
  }
  return Isomorphic().check(a,b);
}

struct Equals : public IndexNotationVisitorStrict {
  bool eq = false;
  IndexExpr bExpr;
  IndexStmt bStmt;

  bool check(IndexExpr a, IndexExpr b) {
    this->bExpr = b;
    a.accept(this);
    return eq;
  }

  bool check(IndexStmt a, IndexStmt b) {
    this->bStmt = b;
    a.accept(this);
    return eq;
  }

  using IndexNotationVisitorStrict::visit;

  void visit(const IndexVarNode* anode) {
    if(!isa<IndexVarNode>(bExpr.ptr)) {
      eq = false;
      return;
    }

    auto bnode = to<IndexVarNode>(bExpr.ptr);
    if(anode != bnode) {
      eq = false;
      return;
    }

    eq = true;
  }

  void visit(const AccessNode* anode) {
    if (!isa<AccessNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AccessNode>(bExpr.ptr);
    if (anode->tensorVar != bnode->tensorVar) {
      eq = false;
      return;
    }
    if (anode->indexVars.size() != bnode->indexVars.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->indexVars.size(); i++) {
      if (anode->indexVars[i] != bnode->indexVars[i]) {
        eq = false;
        return;
      }
    }
    if (anode->isAccessingStructure != bnode->isAccessingStructure ||
        anode->windowedModes != bnode->windowedModes ||
        anode->indexSetModes != bnode->indexSetModes) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const LiteralNode* anode) {
    if (!isa<LiteralNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<LiteralNode>(bExpr.ptr);
    if (anode->getDataType() != bnode->getDataType()) {
      eq = false;
      return;
    }
    if (memcmp(anode->val,bnode->val,anode->getDataType().getNumBytes()) != 0) {
      eq = false;
      return;
    }
    eq = true;
  }

  template <class T>
  bool unaryEquals(const T* anode, IndexExpr b) {
    if (!isa<T>(b.ptr)) {
      return false;
    }
    auto bnode = to<T>(b.ptr);
    if (!equals(anode->a, bnode->a)) {
      return false;
    }
    return true;
  }

  void visit(const NegNode* anode) {
    eq = unaryEquals(anode, bExpr);
  }

  void visit(const SqrtNode* anode) {
    eq = unaryEquals(anode, bExpr);
  }

  template <class T>
  bool binaryEquals(const T* anode, IndexExpr b) {
    if (!isa<T>(b.ptr)) {
      return false;
    }
    auto bnode = to<T>(b.ptr);
    if (!equals(anode->a, bnode->a) || !equals(anode->b, bnode->b)) {
      return false;
    }
    return true;
  }

  void visit(const AddNode* anode) {
    eq = binaryEquals(anode, bExpr);
  }

  void visit(const SubNode* anode) {
    eq = binaryEquals(anode, bExpr);
  }

  void visit(const MulNode* anode) {
    eq = binaryEquals(anode, bExpr);
  }

  void visit(const DivNode* anode) {
    eq = binaryEquals(anode, bExpr);
  }

  void visit(const CastNode* anode) {
    if (!isa<CastNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CastNode>(bExpr.ptr);
    if (anode->getDataType() != bnode->getDataType() ||
        !equals(anode->a, bnode->a)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const CallNode* anode) {
    if (!isa<CallNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CallNode>(bExpr.ptr);

    // Properties
    if (anode->properties.size() != bnode->properties.size()) {
      eq = false;
      return;
    }

    for(const auto& a_prop : anode->properties) {
      bool found = false;
      for(const auto& b_prop : bnode->properties) {
        if(a_prop.equals(b_prop)) {
          found = true;
          break;
        }
      }
      if (!found) {
        eq = false;
        return;
      }
    }

    // Exhausted regions
    if (anode->definedRegions != bnode->definedRegions) {
      eq = false;
      return;
    }

    // Lower function
    // TODO: For now just check that the function pointers are the same.
    if(!util::targetPtrEqual(anode->defaultLowerFunc, bnode->defaultLowerFunc)) {
      eq = false;
      return;
    }

    // Check arguments
    if (anode->args.size() != bnode->args.size()) {
      eq = false;
      return;
    }

    for (size_t i = 0; i < anode->args.size(); ++i) {
      if (!equals(anode->args[i], bnode->args[i])) {
        eq = false;
        return;
      }
    }

    // Algebra
    if (!checkIterationAlg(anode, bnode)) {
      eq = false;
      return;
    }

    // Special definitions
    eq = checkRegionDefinitions(anode, bnode);
  }

  void visit(const CallIntrinsicNode* anode) {
    if (!isa<CallIntrinsicNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<CallIntrinsicNode>(bExpr.ptr);
    if (anode->func->getName() != bnode->func->getName() ||
        anode->args.size() != bnode->args.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->args.size(); ++i) {
      if (!equals(anode->args[i], bnode->args[i])) {
        eq = false;
        return;
      }
    }
    eq = true;
  }

  void visit(const ReductionNode* anode) {
    if (!isa<ReductionNode>(bExpr.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<ReductionNode>(bExpr.ptr);
    if (!equals(anode->op, bnode->op) ||
        anode->var != bnode->var ||
        !equals(anode->a, bnode->a)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const AssignmentNode* anode) {
    if (!isa<AssignmentNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AssignmentNode>(bStmt.ptr);
    if (!equals(anode->lhs, bnode->lhs) || !equals(anode->rhs, bnode->rhs) ||
        !equals(anode->op, bnode->op)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const YieldNode* anode) {
    if (!isa<YieldNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<YieldNode>(bStmt.ptr);
    if (anode->indexVars.size() != bnode->indexVars.size()) {
      eq = false;
      return;
    }
    for (size_t i = 0; i < anode->indexVars.size(); i++) {
      if (anode->indexVars[i] != bnode->indexVars[i]) {
        eq = false;
        return;
      }
    }
    if (!equals(anode->expr, bnode->expr)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const ForallNode* anode) {
    if (!isa<ForallNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<ForallNode>(bStmt.ptr);
    if (anode->indexVar != bnode->indexVar ||
        !equals(anode->stmt, bnode->stmt) ||
        anode->parallel_unit != bnode->parallel_unit ||
        anode->output_race_strategy != bnode->output_race_strategy ||
        anode->unrollFactor != bnode->unrollFactor) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const WhereNode* anode) {
    if (!isa<WhereNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<WhereNode>(bStmt.ptr);
    if (!equals(anode->consumer, bnode->consumer) ||
        !equals(anode->producer, bnode->producer)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const SequenceNode* anode) {
    if (!isa<SequenceNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<SequenceNode>(bStmt.ptr);
    if (!equals(anode->definition, bnode->definition) ||
        !equals(anode->mutation, bnode->mutation)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const AssembleNode* anode) {
    if (!isa<AssembleNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<AssembleNode>(bStmt.ptr);
    if (!equals(anode->queries, bnode->queries) ||
        !equals(anode->compute, bnode->compute)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const MultiNode* anode) {
    if (!isa<MultiNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<MultiNode>(bStmt.ptr);
    if (!equals(anode->stmt1, bnode->stmt1) ||
        !equals(anode->stmt2, bnode->stmt2)) {
      eq = false;
      return;
    }
    eq = true;
  }

  void visit(const SuchThatNode* anode) {
    if (!isa<SuchThatNode>(bStmt.ptr)) {
      eq = false;
      return;
    }
    auto bnode = to<SuchThatNode>(bStmt.ptr);
    if (anode->predicate != bnode->predicate ||
        !equals(anode->stmt, bnode->stmt)) {
      eq = false;
      return;
    }
    eq = true;
  }
};

bool equals(IndexExpr a, IndexExpr b) {
  if (!a.defined() && !b.defined()) {
    return true;
  }
  if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
    return false;
  }
  return Equals().check(a,b);
}

bool equals(IndexStmt a, IndexStmt b) {
  if (!a.defined() && !b.defined()) {
    return true;
  }
  if ((a.defined() && !b.defined()) || (!a.defined() && b.defined())) {
    return false;
  }
  return Equals().check(a,b);
}

IndexExpr operator-(const IndexExpr& expr) {
  return new NegNode(expr.ptr);
}

IndexExpr operator+(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new AddNode(lhs, rhs);
}

IndexExpr operator-(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new SubNode(lhs, rhs);
}

IndexExpr operator*(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new MulNode(lhs, rhs);
}

IndexExpr operator/(const IndexExpr& lhs, const IndexExpr& rhs) {
  return new DivNode(lhs, rhs);
}


// class Access
Access::Access(const AccessNode* n) : IndexExpr(n) {
}

Access::Access(const TensorVar& tensor, const std::vector<IndexVar>& indices,
               const std::map<int, std::shared_ptr<IndexVarIterationModifier>>& modifiers,
               bool isAccessingStructure)
    : Access(new AccessNode(tensor, indices, modifiers, isAccessingStructure)) {
}

const TensorVar& Access::getTensorVar() const {
  return getNode(*this)->tensorVar;
}

const std::vector<IndexVar>& Access::getIndexVars() const {
  return getNode(*this)->indexVars;
}

bool Access::isAccessingStructure() const {
  return getNode(*this)->isAccessingStructure;
}

bool Access::hasWindowedModes() const {
  return !getNode(*this)->windowedModes.empty();
}

bool Access::isModeWindowed(int mode) const {
  auto node = getNode(*this);
  return node->windowedModes.find(mode) != node->windowedModes.end();
}

int Access::getWindowLowerBound(int mode) const {
  taco_iassert(this->isModeWindowed(mode));
  return getNode(*this)->windowedModes.at(mode).lo;
}

int Access::getWindowUpperBound(int mode) const {
  taco_iassert(this->isModeWindowed(mode));
  return getNode(*this)->windowedModes.at(mode).hi;
}

int Access::getWindowSize(int mode) const {
  taco_iassert(this->isModeWindowed(mode));
  auto w = getNode(*this)->windowedModes.at(mode);
  return (w.hi - w.lo) / w.stride;
}

int Access::getStride(int mode) const {
  taco_iassert(this->isModeWindowed(mode));
  return getNode(*this)->windowedModes.at(mode).stride;
}

bool operator==(const Access& a, const Access& b) {
  // Short-circuit for when the Access pointers are the same.
  if (getNode(a) == getNode(b)) {
    return true;
  }
  if (a.getTensorVar() != b.getTensorVar()) {
    return false;
  }
  if (a.getIndexVars() != b.getIndexVars()) {
    return false;
  }
  if (getNode(a)->windowedModes != getNode(b)->windowedModes) {
    return false;
  }
  if (getNode(a)->indexSetModes != getNode(b)->indexSetModes) {
    return false;
  }
  return true;
}

bool operator<(const Access& a, const Access& b) {
  // First branch on tensorVar.
  if (a.getTensorVar() != b.getTensorVar()) {
    return a.getTensorVar() < b.getTensorVar();
  }

  // Then branch on the indexVars used in the access.
  if (a.getIndexVars() != b.getIndexVars()) {
    return a.getIndexVars() < b.getIndexVars();
  }

  // Branch on the windows.
  if (getNode(a)->windowedModes < getNode(b)->windowedModes) {
    return getNode(a)->windowedModes < getNode(b)->windowedModes;
  }

  // Finally, branch on the index set.
  return getNode(a)->indexSetModes < getNode(b)->indexSetModes;
}

bool Access::hasIndexSetModes() const {
  return !getNode(*this)->indexSetModes.empty();
}

bool Access::isModeIndexSet(int mode) const {
  auto node = getNode(*this);
  return util::contains(node->indexSetModes, mode);
}

TensorVar Access::getModeIndexSetTensor(int mode) const {
  taco_iassert(this->isModeIndexSet(mode));
  return getNode(*this)->indexSetModes.at(mode).tensor.getTensorVar();
}

const std::vector<int>& Access::getIndexSet(int mode) const {
  taco_iassert(this->isModeIndexSet(mode));
  return *getNode(*this)->indexSetModes.at(mode).set;
}

static void check(Assignment assignment) {
  auto lhs = assignment.getLhs();
  auto tensorVar = lhs.getTensorVar();
  auto freeVars = lhs.getIndexVars();
  auto indexExpr = assignment.getRhs();
  auto shape = tensorVar.getType().getShape();

  // If the LHS access has any windowed modes, use the dimensions of those
  // windows as the shape, rather than the shape of the underlying tensor.
  if (lhs.hasWindowedModes() || lhs.hasIndexSetModes()) {
    vector<Dimension> dims(shape.getOrder());
    for (int i = 0; i < shape.getOrder();i++) {
      dims[i] = shape.getDimension(i);
      if (lhs.isModeWindowed(i)) {
        dims[i] = Dimension(lhs.getWindowSize(i));
      } else if (lhs.isModeIndexSet(i)) {
        dims[i] = Dimension(lhs.getIndexSet(i).size());
      }
    }
    shape = Shape(dims);
  }

  auto typecheck = error::dimensionsTypecheck(freeVars, indexExpr, shape);
  taco_uassert(typecheck.first) << error::expr_dimension_mismatch << " " << typecheck.second;
}

Assignment Access::operator=(const IndexExpr& expr) {
  TensorVar result = getTensorVar();
  Assignment assignment = Assignment(*this, expr);
  check(assignment);
  const_cast<AccessNode*>(getNode(*this))->setAssignment(assignment);
  return assignment;
}

Assignment Access::operator=(const Access& expr) {
  return operator=(static_cast<IndexExpr>(expr));
}

Assignment Access::operator=(const TensorVar& var) {
  return operator=(Access(var));
}

Assignment Access::operator+=(const IndexExpr& expr) {
  TensorVar result = getTensorVar();
  Assignment assignment = Assignment(
    result,
    getIndexVars(),
    expr,
    Add(),
    // Include any windows on LHS index vars.
    getNode(*this)->packageModifiers()
  );
  // check(assignment); TODO: fix check for precompute
  const_cast<AccessNode*>(getNode(*this))->setAssignment(assignment);
  return assignment;
}

template <> bool isa<Access>(IndexExpr e) {
  return isa<AccessNode>(e.ptr);
}

template <> Access to<Access>(IndexExpr e) {
  taco_iassert(isa<Access>(e));
  return Access(to<AccessNode>(e.ptr));
}


// class Literal
Literal::Literal(const LiteralNode* n) : IndexExpr(n) {
}

Literal::Literal(bool val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(unsigned char val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(unsigned short val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(unsigned int val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(unsigned long val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(unsigned long long val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(char val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(short val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(int val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(long val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(long long val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(int8_t val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(float val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(double val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(std::complex<float> val) : Literal(new LiteralNode(val)) {
}

Literal::Literal(std::complex<double> val) : Literal(new LiteralNode(val)) {
}

Literal Literal::zero(Datatype type) {
  switch (type.getKind()) {
    case Datatype::Bool:        return Literal(false);
    case Datatype::UInt8:       return Literal(uint8_t(0));
    case Datatype::UInt16:      return Literal(uint16_t(0));
    case Datatype::UInt32:      return Literal(uint32_t(0));
    case Datatype::UInt64:      return Literal(uint64_t(0));
    case Datatype::Int8:        return Literal(int8_t(0));
    case Datatype::Int16:       return Literal(int16_t(0));
    case Datatype::Int32:       return Literal(int32_t(0));
    case Datatype::Int64:       return Literal(int64_t(0));
    case Datatype::Float32:     return Literal(float(0.0));
    case Datatype::Float64:     return Literal(double(0.0));
    case Datatype::Complex64:   return Literal(std::complex<float>());
    case Datatype::Complex128:  return Literal(std::complex<double>());
    default:                    taco_ierror << "unsupported type";
  };

  return Literal();
}

template <typename T> T Literal::getVal() const {
  return getNode(*this)->getVal<T>();
}
template bool Literal::getVal() const;
template unsigned char Literal::getVal() const;
template unsigned short Literal::getVal() const;
template unsigned int Literal::getVal() const;
template unsigned long Literal::getVal() const;
template unsigned long long Literal::getVal() const;
template char Literal::getVal() const;
template short Literal::getVal() const;
template int Literal::getVal() const;
template long Literal::getVal() const;
template long long Literal::getVal() const;
template int8_t Literal::getVal() const;
template float Literal::getVal() const;
template double Literal::getVal() const;
template std::complex<float> Literal::getVal() const;
template std::complex<double> Literal::getVal() const;

void* Literal::getValPtr() {
  return getNode(*this)->val;
}

template <> bool isa<Literal>(IndexExpr e) {
  return isa<LiteralNode>(e.ptr);
}

template <> Literal to<Literal>(IndexExpr e) {
  taco_iassert(isa<Literal>(e));
  return Literal(to<LiteralNode>(e.ptr));
}


// class Neg
Neg::Neg(const NegNode* n) : IndexExpr(n) {
}

Neg::Neg(IndexExpr a) : Neg(new NegNode(a)) {
}

IndexExpr Neg::getA() const {
  return getNode(*this)->a;
}

template <> bool isa<Neg>(IndexExpr e) {
  return isa<NegNode>(e.ptr);
}

template <> Neg to<Neg>(IndexExpr e) {
  taco_iassert(isa<Neg>(e));
  return Neg(to<NegNode>(e.ptr));
}


// class Add
Add::Add() : Add(new AddNode) {
}

Add::Add(const AddNode* n) : IndexExpr(n) {
}

Add::Add(IndexExpr a, IndexExpr b) : Add(new AddNode(a, b)) {
}

IndexExpr Add::getA() const {
  return getNode(*this)->a;
}

IndexExpr Add::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<Add>(IndexExpr e) {
  return isa<AddNode>(e.ptr);
}

template <> Add to<Add>(IndexExpr e) {
  taco_iassert(isa<Add>(e));
  return Add(to<AddNode>(e.ptr));
}


// class Sub
Sub::Sub() : Sub(new SubNode) {
}

Sub::Sub(const SubNode* n) : IndexExpr(n) {
}

Sub::Sub(IndexExpr a, IndexExpr b) : Sub(new SubNode(a, b)) {
}

IndexExpr Sub::getA() const {
  return getNode(*this)->a;
}

IndexExpr Sub::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<Sub>(IndexExpr e) {
  return isa<SubNode>(e.ptr);
}

template <> Sub to<Sub>(IndexExpr e) {
  taco_iassert(isa<Sub>(e));
  return Sub(to<SubNode>(e.ptr));
}


// class Mul
Mul::Mul() : Mul(new MulNode) {
}

Mul::Mul(const MulNode* n) : IndexExpr(n) {
}

Mul::Mul(IndexExpr a, IndexExpr b) : Mul(new MulNode(a, b)) {
}

IndexExpr Mul::getA() const {
  return getNode(*this)->a;
}

IndexExpr Mul::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<Mul>(IndexExpr e) {
  return isa<MulNode>(e.ptr);
}

template <> Mul to<Mul>(IndexExpr e) {
  taco_iassert(isa<Mul>(e));
  return Mul(to<MulNode>(e.ptr));
}


// class Div
Div::Div() : Div(new DivNode) {
}

Div::Div(const DivNode* n) : IndexExpr(n) {
}

Div::Div(IndexExpr a, IndexExpr b) : Div(new DivNode(a, b)) {
}

IndexExpr Div::getA() const {
  return getNode(*this)->a;
}

IndexExpr Div::getB() const {
  return getNode(*this)->b;
}

template <> bool isa<Div>(IndexExpr e) {
  return isa<DivNode>(e.ptr);
}

template <> Div to<Div>(IndexExpr e) {
  taco_iassert(isa<Div>(e));
  return Div(to<DivNode>(e.ptr));
}


// class Sqrt
Sqrt::Sqrt(const SqrtNode* n) : IndexExpr(n) {
}

Sqrt::Sqrt(IndexExpr a) : Sqrt(new SqrtNode(a)) {
}

IndexExpr Sqrt::getA() const {
  return getNode(*this)->a;
}

template <> bool isa<Sqrt>(IndexExpr e) {
  return isa<SqrtNode>(e.ptr);
}

template <> Sqrt to<Sqrt>(IndexExpr e) {
  taco_iassert(isa<Sqrt>(e));
  return Sqrt(to<SqrtNode>(e.ptr));
}


// class Cast
Cast::Cast(const CastNode* n) : IndexExpr(n) {
}

Cast::Cast(IndexExpr a, Datatype newType) : Cast(new CastNode(a, newType)) {
}

IndexExpr Cast::getA() const {
  return getNode(*this)->a;
}

template <> bool isa<Cast>(IndexExpr e) {
  return isa<CastNode>(e.ptr);
}

template <> Cast to<Cast>(IndexExpr e) {
  taco_iassert(isa<Cast>(e));
  return Cast(to<CastNode>(e.ptr));
}

// class Call, most construction should happen from tensor_operator.h
Call::Call(const CallNode* n) : IndexExpr(n) {
}

Call::Call(const CallNode *n, std::string name) : IndexExpr(n), name(name) {
}

const std::vector<IndexExpr>& Call::getArgs() const {
  return getNode(*this)->args;
}

const CallNode::OpImpl Call::getFunc() const {
  return getNode(*this)->defaultLowerFunc;
}

const IterationAlgebra& Call::getAlgebra() const {
  return getNode(*this)->iterAlg;
}

const std::vector<Property>& Call::getProperties() const {
  return getNode(*this)->properties;
}

const std::string Call::getName() const {
  return getNode(*this)->name;
}

const std::map<std::vector<int>, CallNode::OpImpl> Call::getDefs() const {
  return getNode(*this)->regionDefinitions;
}

const std::vector<int>& Call::getDefinedArgs() const {
  return getNode(*this)->definedRegions;
}


template <> bool isa<Call>(IndexExpr e) {
  return isa<CallNode>(e.ptr);
}

template <> Call to<Call>(IndexExpr e) {
  taco_iassert(isa<Call>(e));
  return Call(to<CallNode>(e.ptr));
}

// class CallIntrinsic
CallIntrinsic::CallIntrinsic(const CallIntrinsicNode* n) : IndexExpr(n) {
}

CallIntrinsic::CallIntrinsic(const std::shared_ptr<Intrinsic>& func,
                             const std::vector<IndexExpr>& args)
    : CallIntrinsic(new CallIntrinsicNode(func, args)) {
}

const Intrinsic& CallIntrinsic::getFunc() const {
  return *(getNode(*this)->func);
}

const std::vector<IndexExpr>& CallIntrinsic::getArgs() const {
  return getNode(*this)->args;
}

template <> bool isa<CallIntrinsic>(IndexExpr e) {
  return isa<CallIntrinsicNode>(e.ptr);
}

template <> CallIntrinsic to<CallIntrinsic>(IndexExpr e) {
  taco_iassert(isa<CallIntrinsic>(e));
  return CallIntrinsic(to<CallIntrinsicNode>(e.ptr));
}

IndexExpr mod(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<ModIntrinsic>(), {a, b});
}

IndexExpr abs(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AbsIntrinsic>(), {a});
}

IndexExpr pow(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<PowIntrinsic>(), {a, b});
}

IndexExpr square(IndexExpr a) {
  return CallIntrinsic(std::make_shared<SquareIntrinsic>(), {a});
}

IndexExpr cube(IndexExpr a) {
  return CallIntrinsic(std::make_shared<CubeIntrinsic>(), {a});
}

IndexExpr sqrt(IndexExpr a) {
  return CallIntrinsic(std::make_shared<SqrtIntrinsic>(), {a});
}

IndexExpr cbrt(IndexExpr a) {
  return CallIntrinsic(std::make_shared<CbrtIntrinsic>(), {a});
}

IndexExpr exp(IndexExpr a) {
  return CallIntrinsic(std::make_shared<ExpIntrinsic>(), {a});
}

IndexExpr log(IndexExpr a) {
  return CallIntrinsic(std::make_shared<LogIntrinsic>(), {a});
}

IndexExpr log10(IndexExpr a) {
  return CallIntrinsic(std::make_shared<Log10Intrinsic>(), {a});
}

IndexExpr sin(IndexExpr a) {
  return CallIntrinsic(std::make_shared<SinIntrinsic>(), {a});
}

IndexExpr cos(IndexExpr a) {
  return CallIntrinsic(std::make_shared<CosIntrinsic>(), {a});
}

IndexExpr tan(IndexExpr a) {
  return CallIntrinsic(std::make_shared<TanIntrinsic>(), {a});
}

IndexExpr asin(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AsinIntrinsic>(), {a});
}

IndexExpr acos(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AcosIntrinsic>(), {a});
}

IndexExpr atan(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AtanIntrinsic>(), {a});
}

IndexExpr atan2(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<Atan2Intrinsic>(), {a, b});
}

IndexExpr sinh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<SinhIntrinsic>(), {a});
}

IndexExpr cosh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<CoshIntrinsic>(), {a});
}

IndexExpr tanh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<TanhIntrinsic>(), {a});
}

IndexExpr asinh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AsinhIntrinsic>(), {a});
}

IndexExpr acosh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AcoshIntrinsic>(), {a});
}

IndexExpr atanh(IndexExpr a) {
  return CallIntrinsic(std::make_shared<AtanhIntrinsic>(), {a});
}

IndexExpr gt(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<GtIntrinsic>(), {a, b});
}

IndexExpr lt(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<LtIntrinsic>(), {a, b});
}

IndexExpr gte(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<GteIntrinsic>(), {a, b});
}

IndexExpr lte(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<LteIntrinsic>(), {a, b});
}

IndexExpr eq(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<EqIntrinsic>(), {a, b});
}

IndexExpr neq(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<NeqIntrinsic>(), {a, b});
}

IndexExpr max(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<MaxIntrinsic>(), {a, b});
}

IndexExpr min(IndexExpr a, IndexExpr b) {
  return CallIntrinsic(std::make_shared<MinIntrinsic>(), {a, b});
}

IndexExpr heaviside(IndexExpr a, IndexExpr b) {
  if (!b.defined()) {
    b = Literal::zero(a.getDataType());
  }
  return CallIntrinsic(std::make_shared<HeavisideIntrinsic>(), {a, b});
}

IndexExpr Not(IndexExpr a) {
  return CallIntrinsic(std::make_shared<NotIntrinsic>(), {a});
}


// class Reduction
Reduction::Reduction(const ReductionNode* n) : IndexExpr(n) {
}

Reduction::Reduction(IndexExpr op, IndexVar var, IndexExpr expr)
    : Reduction(new ReductionNode(op, var, expr)) {
}

IndexExpr Reduction::getOp() const {
  return getNode(*this)->op;
}

IndexVar Reduction::getVar() const {
  return getNode(*this)->var;
}

IndexExpr Reduction::getExpr() const {
  return getNode(*this)->a;
}

Reduction sum(IndexVar i, IndexExpr expr) {
  return Reduction(Add(), i, expr);
}

template <> bool isa<Reduction>(IndexExpr s) {
  return isa<ReductionNode>(s.ptr);
}

template <> Reduction to<Reduction>(IndexExpr s) {
  taco_iassert(isa<Reduction>(s));
  return Reduction(to<ReductionNode>(s.ptr));
}


// class IndexStmt
IndexStmt::IndexStmt() : util::IntrusivePtr<const IndexStmtNode>(nullptr) {
}

IndexStmt::IndexStmt(const IndexStmtNode* n)
    : util::IntrusivePtr<const IndexStmtNode>(n) {
}

void IndexStmt::accept(IndexStmtVisitorStrict *v) const {
  ptr->accept(v);
}

std::vector<IndexVar> IndexStmt::getIndexVars() const {
  vector<IndexVar> vars;;
  set<IndexVar> seen;
  match(*this,
    std::function<void(const AssignmentNode*,Matcher*)>([&](
        const AssignmentNode* op, Matcher* ctx) {
      for (auto& var : op->lhs.getIndexVars()) {
        if (!util::contains(seen, var)) {
          vars.push_back(var);
          seen.insert(var);
        }
      }
      ctx->match(op->rhs);
    }),
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
      for (auto& var : op->indexVars) {
        if (!util::contains(seen, var)) {
          vars.push_back(var);
          seen.insert(var);
        }
      }
    })
  );
  return vars;
}

map<IndexVar,Dimension> IndexStmt::getIndexVarDomains() const {
  map<IndexVar, Dimension> indexVarDomains;
  match(*this,
    std::function<void(const AssignmentNode*,Matcher*)>([](
        const AssignmentNode* op, Matcher* ctx) {
      ctx->match(op->lhs);
      ctx->match(op->rhs);
    }),
    function<void(const AccessNode*)>([&indexVarDomains](const AccessNode* op) {
      auto& type = op->tensorVar.getType();
      auto& vars = op->indexVars;
      for (size_t i = 0; i < vars.size(); i++) {
        if (!util::contains(indexVarDomains, vars[i])) {
          indexVarDomains.insert({vars[i], type.getShape().getDimension(i)});
        }
        else {
          taco_iassert(indexVarDomains.at(vars[i]) ==
                       type.getShape().getDimension(i))
              << "Index variable used to index incompatible dimensions";
        }
      }
    })
  );

  return indexVarDomains;
}

IndexStmt IndexStmt::concretizeScheduled(ProvenanceGraph provGraph, vector<IndexVar> forallIndexVarList) const {
  IndexStmt stmt = *this;
  string r;
  if (isEinsumNotation(stmt, &r)) {
    stmt = makeReductionNotationScheduled(stmt, provGraph);
  }
  if (isReductionNotationScheduled(stmt, provGraph, &r)) {
    stmt = makeConcreteNotationScheduled(stmt, provGraph, forallIndexVarList);
  }
  return stmt;
}

IndexStmt IndexStmt::concretize() const {
  IndexStmt stmt = *this;
  if (isEinsumNotation(stmt)) {
    stmt = makeReductionNotation(stmt);
  }
  if (isReductionNotation(stmt)) {
    stmt = makeConcreteNotation(stmt);
  }
  return stmt;
}

IndexStmt IndexStmt::split(IndexVar i, IndexVar i1, IndexVar i2, size_t splitFactor) const {
  IndexVarRel rel = IndexVarRel(new SplitRelNode(i, i1, i2, splitFactor));
  string reason;

  // Add predicate to concrete index notation
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  // Replace all occurrences of i with nested i1, i2
  transformed = Transformation(ForAllReplace({i}, {i1, i2})).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}

IndexStmt IndexStmt::divide(IndexVar i, IndexVar i1, IndexVar i2, size_t splitFactor) const {
  IndexVarRel rel = IndexVarRel(new DivideRelNode(i, i1, i2, splitFactor));
  string reason;

  // Add predicate to concrete index notation.
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  // Replace all occurrences of i with nested i1, i2.
  transformed = Transformation(ForAllReplace({i}, {i1, i2})).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}

IndexStmt IndexStmt::precompute(IndexExpr expr, std::vector<IndexVar> i_vars,
                                std::vector<IndexVar> iw_vars, TensorVar workspace) const {

  IndexStmt transformed = *this;
  string reason;

 taco_uassert(i_vars.size() == iw_vars.size()) << "The precompute transformation requires"
                                               << "i_vars and iw_vars to be the same size";
 for (int l = 0; l < (int) i_vars.size(); l++) {
    IndexVar i = i_vars.at(l);
    IndexVar iw = iw_vars.at(l);

    if (i != iw) {
      IndexVarRel rel = IndexVarRel(new PrecomputeRelNode(i, iw));
      transformed = Transformation(AddSuchThatPredicates({rel})).apply(transformed, &reason);
      if (!transformed.defined()) {
        taco_uerror << reason;
      }
    }
  }

  transformed = Transformation(Precompute(expr, i_vars, iw_vars, workspace)).apply(transformed, &reason);

  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

IndexStmt IndexStmt::precompute(IndexExpr expr, IndexVar i, IndexVar iw, TensorVar workspace) const {
  std::vector<IndexVar> i_vars{i};
  std::vector<IndexVar> iw_vars{iw};
  return precompute(expr, i_vars, iw_vars, workspace);
}

IndexStmt IndexStmt::reorder(taco::IndexVar i, taco::IndexVar j) const {
  string reason;
  IndexStmt transformed = Reorder(i, j).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

IndexStmt IndexStmt::reorder(std::vector<IndexVar> reorderedvars) const {
  string reason;
  IndexStmt transformed = Reorder(reorderedvars).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

IndexStmt IndexStmt::parallelize(IndexVar i, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy) const {
  string reason;
  IndexStmt transformed = Parallelize(i, parallel_unit, output_race_strategy).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

IndexStmt IndexStmt::pos(IndexVar i, IndexVar ipos, Access access) const {
  // check access is contained in stmt
  bool foundAccess = false;
  for (Access argAccess : getArgumentAccesses(*this)) {
    if (argAccess.getTensorVar() == access.getTensorVar() && argAccess.getIndexVars() == access.getIndexVars()) {
      foundAccess = true;
      break;
    }
  }
  if (!foundAccess) {
    taco_uerror << "Access: " << access << " does not appear in index statement as an argument";
  }

  // check access is correct
  ProvenanceGraph provGraph = ProvenanceGraph(*this);
  vector<IndexVar> underivedParentAncestors = provGraph.getUnderivedAncestors(i);
  size_t max_mode = 0;
  for (IndexVar underived : underivedParentAncestors) {
    size_t mode_index = 0; // which of the access index vars match?
    for (auto var : access.getIndexVars()) {
      if (var == underived) {
        break;
      }
      mode_index++;
    }
    if (mode_index > max_mode) max_mode = mode_index;
  }
  if ((size_t)max_mode >= access.getIndexVars().size()) {
    taco_uerror << "Index variable " << i << " does not appear in access: " << access;
  }

  int mode = access.getTensorVar().getFormat().getModeOrdering()[max_mode];
  if (access.getTensorVar().getFormat().getModeFormats()[mode] == Dense) {
    taco_uerror << "Pos transformation is not valid for dense formats, the coordinate space should be transformed instead";
  }

  IndexVarRel rel = IndexVarRel(new PosRelNode(i, ipos, access));
  string reason;

  // Add predicate to concrete index notation
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  // Replace all occurrences of i with ipos
  transformed = Transformation(ForAllReplace({i}, {ipos})).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}

IndexStmt IndexStmt::fuse(IndexVar i, IndexVar j, IndexVar f) const {
  IndexVarRel rel = IndexVarRel(new FuseRelNode(i, j, f));
  string reason;

  // Add predicate to concrete index notation
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  // Replace all occurrences of i, j with f
  transformed = Transformation(ForAllReplace({i,j}, {f})).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}

IndexStmt IndexStmt::bound(IndexVar i, IndexVar i1, size_t bound, BoundType bound_type) const {
  IndexVarRel rel = IndexVarRel(new BoundRelNode(i, i1, bound, bound_type));
  string reason;

  // Add predicate to concrete index notation
  IndexStmt transformed = Transformation(AddSuchThatPredicates({rel})).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  // Replace all occurrences of i with i1
  transformed = Transformation(ForAllReplace({i}, {i1})).apply(transformed, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }

  return transformed;
}

IndexStmt IndexStmt::unroll(IndexVar i, size_t unrollFactor) const {
  struct UnrollLoop : IndexNotationRewriter {
    using IndexNotationRewriter::visit;
    IndexVar i;
    size_t unrollFactor;
    UnrollLoop(IndexVar i, size_t unrollFactor) : i(i), unrollFactor(unrollFactor) {}

    void visit(const ForallNode* node) {
      if (node->indexVar == i) {
        stmt = Forall(i, rewrite(node->stmt), node->parallel_unit, node->output_race_strategy, unrollFactor);
      }
      else {
        IndexNotationRewriter::visit(node);
      }
    }
  };
  return UnrollLoop(i, unrollFactor).rewrite(*this);
}

IndexStmt IndexStmt::assemble(TensorVar result, AssembleStrategy strategy,
                              bool separatelySchedulable) const {
  string reason;
  IndexStmt transformed = 
      SetAssembleStrategy(result, strategy, 
                          separatelySchedulable).apply(*this, &reason);
  if (!transformed.defined()) {
    taco_uerror << reason;
  }
  return transformed;
}

std::ostream& operator<<(std::ostream& os, const IndexStmt& expr) {
  if (!expr.defined()) return os << "IndexStmt()";
  IndexNotationPrinter printer(os);
  printer.print(expr);
  return os;
}

// class Assignment
Assignment::Assignment(const AssignmentNode* n) : IndexStmt(n) {
}

Assignment::Assignment(Access lhs, IndexExpr rhs, IndexExpr op)
    : Assignment(new AssignmentNode(lhs, rhs, op)) {
}

Assignment::Assignment(TensorVar tensor, vector<IndexVar> indices,
                       IndexExpr rhs, IndexExpr op,
                       const std::map<int, std::shared_ptr<IndexVarIterationModifier>>& modifiers)
    : Assignment(Access(tensor, indices, modifiers), rhs, op) {
}

Access Assignment::getLhs() const {
  return getNode(*this)->lhs;
}

IndexExpr Assignment::getRhs() const {
  return getNode(*this)->rhs;
}

IndexExpr Assignment::getOperator() const {
  return getNode(*this)->op;
}

const std::vector<IndexVar>& Assignment::getFreeVars() const {
  return getLhs().getIndexVars();
}

std::vector<IndexVar> Assignment::getReductionVars() const {
  vector<IndexVar> freeVars = getLhs().getIndexVars();
  set<IndexVar> seen(freeVars.begin(), freeVars.end());
  vector<IndexVar> reductionVars;
  match(getRhs(),
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
    for (auto& var : op->indexVars) {
      if (!util::contains(seen, var)) {
        reductionVars.push_back(var);
        seen.insert(var);
      }
    }
    })
  );
  return reductionVars;
}

template <> bool isa<Assignment>(IndexStmt s) {
  return isa<AssignmentNode>(s.ptr);
}

template <> Assignment to<Assignment>(IndexStmt s) {
  taco_iassert(isa<Assignment>(s));
  return Assignment(to<AssignmentNode>(s.ptr));
}


// class Yield
Yield::Yield(const YieldNode* n) : IndexStmt(n) {
}

Yield::Yield(const std::vector<IndexVar>& indexVars, IndexExpr expr)
    : Yield(new YieldNode(indexVars, expr)) {
}

const std::vector<IndexVar>& Yield::getIndexVars() const {
  return getNode(*this)->indexVars;
}

IndexExpr Yield::getExpr() const {
  return getNode(*this)->expr;
}


// class Forall
Forall::Forall(const ForallNode* n) : IndexStmt(n) {
}

Forall::Forall(IndexVar indexVar, IndexStmt stmt)
    : Forall(indexVar, stmt, ParallelUnit::NotParallel, OutputRaceStrategy::IgnoreRaces) {
}

Forall::Forall(IndexVar indexVar, IndexStmt stmt, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy, size_t unrollFactor)
        : Forall(new ForallNode(indexVar, stmt, parallel_unit, output_race_strategy, unrollFactor)) {
}

IndexVar Forall::getIndexVar() const {
  return getNode(*this)->indexVar;
}

IndexStmt Forall::getStmt() const {
  return getNode(*this)->stmt;
}

ParallelUnit Forall::getParallelUnit() const {
  return getNode(*this)->parallel_unit;
}

OutputRaceStrategy Forall::getOutputRaceStrategy() const {
  return getNode(*this)->output_race_strategy;
}

size_t Forall::getUnrollFactor() const {
  return getNode(*this)->unrollFactor;
}

Forall forall(IndexVar i, IndexStmt stmt) {
  return Forall(i, stmt);
}

Forall forall(IndexVar i, IndexStmt stmt, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy, size_t unrollFactor) {
  return Forall(i, stmt, parallel_unit, output_race_strategy, unrollFactor);
}

template <> bool isa<Forall>(IndexStmt s) {
  return isa<ForallNode>(s.ptr);
}

template <> Forall to<Forall>(IndexStmt s) {
  taco_iassert(isa<Forall>(s));
  return Forall(to<ForallNode>(s.ptr));
}


// class Where
Where::Where(const WhereNode* n) : IndexStmt(n) {
}

Where::Where(IndexStmt consumer, IndexStmt producer)
    : Where(new WhereNode(consumer, producer)) {
}

IndexStmt Where::getConsumer() {
  return getNode(*this)->consumer;
}


IndexStmt Where::getProducer() {
  return getNode(*this)->producer;
}

TensorVar Where::getResult() {
  return getResultAccesses(getConsumer()).first[0].getTensorVar();
}

TensorVar Where::getTemporary() {
  return getResultAccesses(getProducer()).first[0].getTensorVar();
}

Where where(IndexStmt consumer, IndexStmt producer) {
  return Where(consumer, producer);
}

template <> bool isa<Where>(IndexStmt s) {
  return isa<WhereNode>(s.ptr);
}

template <> Where to<Where>(IndexStmt s) {
  taco_iassert(isa<Where>(s));
  return Where(to<WhereNode>(s.ptr));
}


// class Sequence
Sequence::Sequence(const SequenceNode* n) :IndexStmt(n) {
}

Sequence::Sequence(IndexStmt definition, IndexStmt mutation)
    : Sequence(new SequenceNode(definition, mutation)) {
}

IndexStmt Sequence::getDefinition() const {
  return getNode(*this)->definition;
}

IndexStmt Sequence::getMutation() const {
  return getNode(*this)->mutation;
}

Sequence sequence(IndexStmt definition, IndexStmt mutation) {
  return Sequence(definition, mutation);
}

template <> bool isa<Sequence>(IndexStmt s) {
  return isa<SequenceNode>(s.ptr);
}

template <> Sequence to<Sequence>(IndexStmt s) {
  taco_iassert(isa<Sequence>(s));
  return Sequence(to<SequenceNode>(s.ptr));
}


// class Assemble
Assemble::Assemble(const AssembleNode* n) :IndexStmt(n) {
}

Assemble::Assemble(IndexStmt queries, IndexStmt compute, 
                   AttrQueryResults results)
    : Assemble(new AssembleNode(queries, compute, results)) {
}

IndexStmt Assemble::getQueries() const {
  return getNode(*this)->queries;
}

IndexStmt Assemble::getCompute() const {
  return getNode(*this)->compute;
}

const Assemble::AttrQueryResults& Assemble::getAttrQueryResults() const {
  return getNode(*this)->results;
}

Assemble assemble(IndexStmt queries, IndexStmt compute, 
                  Assemble::AttrQueryResults results) {
  return Assemble(queries, compute, results);
}

template <> bool isa<Assemble>(IndexStmt s) {
  return isa<AssembleNode>(s.ptr);
}

template <> Assemble to<Assemble>(IndexStmt s) {
  taco_iassert(isa<Assemble>(s));
  return Assemble(to<AssembleNode>(s.ptr));
}


// class Multi
Multi::Multi(const MultiNode* n) : IndexStmt(n) {
}

Multi::Multi(IndexStmt stmt1, IndexStmt stmt2)
    : Multi(new MultiNode(stmt1, stmt2)) {
}

IndexStmt Multi::getStmt1() const {
  return getNode(*this)->stmt1;
}

IndexStmt Multi::getStmt2() const {
  return getNode(*this)->stmt2;
}

Multi multi(IndexStmt stmt1, IndexStmt stmt2) {
  return Multi(stmt1, stmt2);
}

template <> bool isa<Multi>(IndexStmt s) {
  return isa<MultiNode>(s.ptr);
}

template <> Multi to<Multi>(IndexStmt s) {
  taco_iassert(isa<Multi>(s));
  return Multi(to<MultiNode>(s.ptr));
}

// class SuchThat
SuchThat::SuchThat(const SuchThatNode* n) : IndexStmt(n) {
}

SuchThat::SuchThat(IndexStmt stmt, std::vector<IndexVarRel> predicate)
        : SuchThat(new SuchThatNode(stmt, predicate)) {
}

IndexStmt SuchThat::getStmt() const {
  return getNode(*this)->stmt;
}

std::vector<IndexVarRel> SuchThat::getPredicate() const {
  return getNode(*this)->predicate;
}

SuchThat suchthat(IndexStmt stmt, std::vector<IndexVarRel> predicate) {
  return SuchThat(stmt, predicate);
}

template <> bool isa<SuchThat>(IndexStmt s) {
  return isa<SuchThatNode>(s.ptr);
}

template <> SuchThat to<SuchThat>(IndexStmt s) {
  taco_iassert(isa<SuchThat>(s));
  return SuchThat(to<SuchThatNode>(s.ptr));
}

// class IndexVar
IndexVar::IndexVar() : IndexVar(util::uniqueName('i')) {}

IndexVar::IndexVar(const std::string& name) : IndexVar(name, Datatype::Int32) {}

IndexVar::IndexVar(const std::string& name, const Datatype& type) : IndexVar(new IndexVarNode(name, type)) {}

IndexVar::IndexVar(const IndexVarNode* n) : IndexExpr(n) {}

template <> bool isa<IndexVar>(IndexExpr e) {
  return isa<IndexVarNode>(e.ptr);
}

template <> IndexVar to<IndexVar>(IndexExpr e) {
  taco_iassert(isa<IndexVar>(e));
  return IndexVar(to<IndexVarNode>(e.ptr));
}

std::string IndexVar::getName() const {
  return getNode(*this)->getName();
}

WindowedIndexVar IndexVar::operator()(int lo, int hi, int stride) {
  return WindowedIndexVar(*this, lo, hi, stride);
}

IndexSetVar IndexVar::operator()(std::vector<int>&& indexSet) {
  return IndexSetVar(*this, indexSet);
}

IndexSetVar IndexVar::operator()(std::vector<int>& indexSet) {
  return IndexSetVar(*this, indexSet);
}

bool operator==(const IndexVar& a, const IndexVar& b) {
  return *getNode(a) == *getNode(b);
}

bool operator<(const IndexVar& a, const IndexVar& b) {
  return *getNode(a) < *getNode(b);
}

bool operator!=(const IndexVar& a , const IndexVar& b) {
  return *getNode(a) != *getNode(b);
}

bool operator>=(const IndexVar& a, const IndexVar& b) {
  return *getNode(a) >= *getNode(b);
}

bool operator<=(const IndexVar& a, const IndexVar& b) {
  return *getNode(a) <= *getNode(b);
}

bool operator>(const IndexVar& a , const IndexVar& b) {
  return *getNode(a) > *getNode(b);
}

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<IndexVarInterface>& var) {
  std::stringstream ss;
  IndexVarInterface::match(var, [&](std::shared_ptr<IndexVar> ivar) {
    ss << *ivar;
  }, [&](std::shared_ptr<WindowedIndexVar> wvar) {
    ss << *wvar;
  }, [&](std::shared_ptr<IndexSetVar> svar) {
    ss << *svar;
  });
  return os << ss.str();
}

std::ostream& operator<<(std::ostream& os, const IndexVar& var) {
  return os << var.getName();
}

std::ostream& operator<<(std::ostream& os, const WindowedIndexVar& var) {
  return os << var.getIndexVar();
}

std::ostream& operator<<(std::ostream& os, const IndexSetVar& var) {
  return os << var.getIndexVar();
}

WindowedIndexVar::WindowedIndexVar(IndexVar base, int lo, int hi, int stride) : content( new Content){
  this->content->base = base;
  this->content->lo = lo;
  this->content->hi = hi;
  this->content->stride = stride;
}

IndexVar WindowedIndexVar::getIndexVar() const {
  return this->content->base;
}

int WindowedIndexVar::getLowerBound() const {
  return this->content->lo;
}

int WindowedIndexVar::getUpperBound() const {
  return this->content->hi;
}

int WindowedIndexVar::getStride() const {
  return this->content->stride;
}

int WindowedIndexVar::getWindowSize() const {
  return (this->content->hi - this->content->lo) / this->content->stride;
}

IndexSetVar::IndexSetVar(IndexVar base, std::vector<int> indexSet): content (new Content) {
  this->content->base = base;
  this->content->indexSet = indexSet;
}

IndexVar IndexSetVar::getIndexVar() const {
  return this->content->base;
}

const std::vector<int>& IndexSetVar::getIndexSet() const {
  return this->content->indexSet;
}

// class TensorVar
struct TensorVar::Content {
  int id;
  string name;
  Type type;
  Format format;
  Schedule schedule;
  Literal fill;
};

TensorVar::TensorVar() : content(nullptr) {
}

static Format createDenseFormat(const Type& type) {
  return Format(vector<ModeFormatPack>(type.getOrder(), ModeFormat(Dense)));
}

TensorVar::TensorVar(const Type& type, const Literal& fill)
: TensorVar(type, createDenseFormat(type), fill) {
}

TensorVar::TensorVar(const std::string& name, const Type& type, const Literal& fill)
: TensorVar(-1, name, type, createDenseFormat(type), fill) {
}

TensorVar::TensorVar(const Type& type, const Format& format, const Literal& fill)
    : TensorVar(-1, util::uniqueName('A'), type, format, fill) {
}

TensorVar::TensorVar(const string& name, const Type& type, const Format& format, const Literal& fill)
    : TensorVar(-1, name, type, format, fill) {
}

TensorVar::TensorVar(const int& id, const string& name, const Type& type, const Format& format, const Literal& fill)
    : content(new Content) {
  content->id = id;
  content->name = name;
  content->type = type;
  content->format = format;
  content->fill = fill.defined()? fill : Literal::zero(type.getDataType());
}

int TensorVar::getId() const {
  return content->id;
}

std::string TensorVar::getName() const {
  return content->name;
}

int TensorVar::getOrder() const {
  return content->type.getShape().getOrder();
}

const Type& TensorVar::getType() const {
  return content->type;
}

const Format& TensorVar::getFormat() const {
  return content->format;
}

const Schedule& TensorVar::getSchedule() const {
  struct GetSchedule : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    Schedule schedule;
    void visit(const BinaryExprNode* expr) {
      auto workspace = expr->getWorkspace();
      if (workspace.defined()) {
        schedule.addPrecompute(workspace);
      }
    }
  };
  GetSchedule getSchedule;
  content->schedule.clearPrecomputes();
  getSchedule.schedule = content->schedule;
  return content->schedule;
}

const Literal& TensorVar::getFill() const {
  return content->fill;
}

void TensorVar::setFill(const Literal &fill) {
  content->fill = fill;
}

void TensorVar::setName(std::string name) {
  content->name = name;
}

bool TensorVar::defined() const {
  return content != nullptr;
}

const Access TensorVar::operator()(const std::vector<IndexVar>& indices) const {
  taco_uassert((int)indices.size() == getOrder()) <<
      "A tensor of order " << getOrder() << " must be indexed with " <<
      getOrder() << " variables, but is indexed with:  " << util::join(indices);
  return Access(new AccessNode(*this, indices, {}, false));
}

Access TensorVar::operator()(const std::vector<IndexVar>& indices) {
  taco_uassert((int)indices.size() == getOrder()) <<
      "A tensor of order " << getOrder() << " must be indexed with " <<
      getOrder() << " variables, but is indexed with:  " << util::join(indices);
  return Access(new AccessNode(*this, indices, {}, false));
}

Assignment TensorVar::operator=(IndexExpr expr) {
  taco_uassert(getOrder() == 0)
      << "Must use index variable on the left-hand-side when assigning an "
      << "expression to a non-scalar tensor.";
  Assignment assignment = Assignment(*this, {}, expr);
  check(assignment);
  return assignment;
}

Assignment TensorVar::operator+=(IndexExpr expr) {
  taco_uassert(getOrder() == 0)
      << "Must use index variable on the left-hand-side when assigning an "
      << "expression to a non-scalar tensor.";
  Assignment assignment = Assignment(*this, {}, expr, new AddNode);
  check(assignment);
  return assignment;
}

bool operator==(const TensorVar& a, const TensorVar& b) {
  return a.content == b.content;
}

bool operator<(const TensorVar& a, const TensorVar& b) {
  return a.content < b.content;
}

std::ostream& operator<<(std::ostream& os, const TensorVar& var) {
  return os << var.getName() << " : " << var.getType();
}


static bool isValid(Assignment assignment, string* reason) {
  if (reason == nullptr) {
    INIT_REASON(reason);
  }
  auto rhs = assignment.getRhs();
  auto lhs = assignment.getLhs();
  auto result = lhs.getTensorVar();
  auto freeVars = lhs.getIndexVars();
  auto shape = result.getType().getShape();

  // If the LHS access has any windowed modes, use the dimensions of those
  // windows as the shape, rather than the shape of the underlying tensor.
  if (lhs.hasWindowedModes() || lhs.hasIndexSetModes()) {
    vector<Dimension> dims(shape.getOrder());
    for (int i = 0; i < shape.getOrder();i++) {
      dims[i] = shape.getDimension(i);
      if (lhs.isModeWindowed(i)) {
        dims[i] = Dimension(lhs.getWindowSize(i));
      } else if (lhs.isModeIndexSet(i)) {
        dims[i] = Dimension(lhs.getIndexSet(i).size());
      }
    }
    shape = Shape(dims);
  }

  auto typecheck = error::dimensionsTypecheck(freeVars, rhs, shape);
  if (!typecheck.first) {
    *reason = error::expr_dimension_mismatch + " " + typecheck.second;
    return false;
  }
  return true;
}

// functions
bool isEinsumNotation(IndexStmt stmt, std::string* reason) {
  INIT_REASON(reason);

  if (!isa<Assignment>(stmt)) {
    *reason = "Einsum notation statements must be assignments.";
    return false;
  }

  if (!isValid(to<Assignment>(stmt), reason)) {
    return false;
  }

  // Einsum notation until proved otherwise
  bool isEinsum = true;

  // Additions are not allowed under the first multiplication
  bool mulnodeVisited = false;

  match(stmt,
    std::function<void(const AddNode*,Matcher*)>([&](const AddNode* op,
                                                     Matcher* ctx) {
      if (mulnodeVisited) {
        *reason = "additions in einsum notation must not be nested under "
                  "multiplications";
        isEinsum = false;
      }
      else {
        ctx->match(op->a);
        ctx->match(op->b);
      }
    }),
    std::function<void(const SubNode*,Matcher*)>([&](const SubNode* op,
                                                     Matcher* ctx) {
      if (mulnodeVisited) {
        *reason = "subtractions in einsum notation must not be nested under "
                  "multiplications";
        isEinsum = false;
      }
      else {
        ctx->match(op->a);
        ctx->match(op->b);
      }
    }),
    std::function<void(const MulNode*,Matcher*)>([&](const MulNode* op,
                                                     Matcher* ctx) {
      bool topMulNode = !mulnodeVisited;
      mulnodeVisited = true;
      ctx->match(op->a);
      ctx->match(op->b);
      if (topMulNode) {
        mulnodeVisited = false;
      }
    }),
    std::function<void(const BinaryExprNode*)>([&](const BinaryExprNode* op) {
      *reason = "einsum notation may not contain " + op->getOperatorString() +
                " operations";
      isEinsum = false;
    }),
    std::function<void(const ReductionNode*)>([&](const ReductionNode* op) {
      *reason = "einsum notation may not contain reductions";
      isEinsum = false;
    })
  );
  return isEinsum;
}

bool isReductionNotation(IndexStmt stmt, std::string* reason) {
  INIT_REASON(reason);

  if (!isa<Assignment>(stmt)) {
    *reason = "reduction notation statements must be assignments";
    return false;
  }

  if (!isValid(to<Assignment>(stmt), reason)) {
    return false;
  }

  // Reduction notation until proved otherwise
  bool isReduction = true;

  util::ScopedSet<IndexVar> boundVars; 
  vector<IndexVar> boundVarsList;
  for (auto& var : to<Assignment>(stmt).getFreeVars()) {
    boundVars.insert({var});
    boundVarsList.push_back(var);
  }

  match(stmt,
    std::function<void(const ReductionNode*,Matcher*)>([&](
        const ReductionNode* op, Matcher* ctx) {
      boundVars.scope();
      boundVars.insert({op->var});
      ctx->match(op->a);
      boundVars.unscope();
    }),
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
      for (auto& var : op->indexVars) {
        if (!boundVars.contains(var)) {
          *reason = "all reduction variables in reduction notation must be "
                    "bound by a reduction expression";
          isReduction = false;
        }
      }
    })
  );
  return isReduction;
}

bool isReductionNotationScheduled(IndexStmt stmt, ProvenanceGraph provGraph, std::string* reason) {
  INIT_REASON(reason);

  if (!isa<Assignment>(stmt)) {
    *reason = "reduction notation statements must be assignments";
    return false;
  }

  if (!isValid(to<Assignment>(stmt), reason)) {
    return false;
  }

  // Reduction notation until proved otherwise
  bool isReduction = true;

  util::ScopedSet<IndexVar> boundVars; 
  vector<IndexVar> boundVarsList;
  for (auto& var : to<Assignment>(stmt).getFreeVars()) {
    boundVars.insert({var});
    boundVarsList.push_back(var);
  }

  match(stmt,
        std::function<void(const ReductionNode*,Matcher*)>([&](
          const ReductionNode* op, Matcher* ctx) {
          boundVars.scope();
          boundVars.insert({op->var});
          ctx->match(op->a);
          boundVars.unscope();
        }),
        std::function<void(const AccessNode*)>([&](const AccessNode* op) {
          for (auto& var : op->indexVars) {
            if (!boundVars.contains(var)) {
              // This detects to see if one of the boundVars is an ancestor of var
              // or if boundVars is a descendant of var given the Provenance Graph.
              // If either of these are true, then the statement is still in reduction notation.
              if (provGraph.isFullyDerived(var)) {
                auto ancestors = provGraph.getUnderivedAncestors(var);
                for (auto& ancestor: ancestors) {
                  if (boundVars.contains(ancestor)) {
                    isReduction = true;
                  }
                }
              } else {
                auto descendants = provGraph.getFullyDerivedDescendants(var);
                for (auto& descendant : descendants) {
                  if (boundVars.contains(descendant)) {
                    isReduction = true;
                  }
                }
              }
                  *reason = "all reduction variables in reduction notation must be "
                            "bound by a reduction expression";
              isReduction = false;
            }
          }
        })
  );
  return isReduction;
}

bool isConcreteNotation(IndexStmt stmt, std::string* reason) {
  taco_iassert(stmt.defined()) << "the index statement is undefined";
  INIT_REASON(reason);

  // Concrete notation until proved otherwise
  bool isConcrete = true;

  bool inWhereProducer = false;
  bool inWhereConsumer = false;
  util::ScopedSet<IndexVar> boundVars; 
  std::set<IndexVar> definedVars; // used to check if all variables recoverable TODO: need to actually use scope like above

  ProvenanceGraph provGraph = ProvenanceGraph(stmt);

  match(stmt,
    std::function<void(const ForallNode*,Matcher*)>([&](const ForallNode* op,
                                                        Matcher* ctx) {
      boundVars.scope();
      boundVars.insert({op->indexVar});
      definedVars.insert(op->indexVar);
      ctx->match(op->stmt);
      boundVars.unscope();
    }),
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
      for (auto& var : op->indexVars) {
        // non underived variables may appear in temporaries, but we don't check these
        if (!boundVars.contains(var) && provGraph.isUnderived(var) &&
           (provGraph.isFullyDerived(var) || !provGraph.isRecoverable(var, definedVars))) {
          *reason = "all variables in concrete notation must be bound by a "
                    "forall statement";
          isConcrete = false;
        }
      }
    }),
    std::function<void(const IndexVarNode*)>([&](const IndexVarNode* op) {
      IndexVar var(op);
      if (!boundVars.contains(var) && provGraph.isUnderived(var) &&
         (provGraph.isFullyDerived(var) || !provGraph.isRecoverable(var, definedVars)))  {
        *reason = "index variables used in compute statements must be nested under a forall";
        isConcrete = false;
      }
    }),
    std::function<void(const WhereNode*,Matcher*)>([&](const WhereNode* op, Matcher* ctx) {
      bool alreadyInProducer = inWhereProducer;
      inWhereProducer = true;
      ctx->match(op->producer);
      if (!alreadyInProducer) inWhereProducer = false;
      bool alreadyInConsumer = inWhereConsumer;
      inWhereConsumer = true;
      ctx->match(op->consumer);
      if (!alreadyInConsumer) inWhereConsumer = false;
    }),
    std::function<void(const AssignmentNode*,Matcher*)>([&](
        const AssignmentNode* op, Matcher* ctx) {
      if(!inWhereConsumer && !inWhereProducer && !isValid(Assignment(op), reason)) { // TODO: fix check for precompute
        isConcrete = false;
        return;
      }

      // Handles derived vars on RHS with underived vars on LHS.
      Assignment assignPtrWrapper = Assignment(op);
      std::vector<IndexVar> possibleReductionVars = assignPtrWrapper.getReductionVars();
      std::vector<IndexVar> freeVars = assignPtrWrapper.getFreeVars();
      std::set<IndexVar> freeVarsSet(freeVars.begin(), freeVars.end());

      int numReductionVars = 0;
      for(const auto& reductionVar : possibleReductionVars) {
        std::vector<IndexVar> underivedParents = provGraph.getUnderivedAncestors(reductionVar);
        for(const auto& parent : underivedParents) {
          if(!util::contains(freeVarsSet, parent)) {
            ++numReductionVars;
          }
        }
      }
      // allow introducing precompute loops where we set a temporary to values instead of +=
      if (numReductionVars > 0 &&
          op->op == IndexExpr() && !inWhereProducer) {
        *reason = "reduction variables in concrete notation must be dominated "
                  "by compound assignments (such as +=)";
        isConcrete = false;
        return;
      }

      ctx->match(op->lhs);
      ctx->match(op->rhs);
    }),
    std::function<void(const ReductionNode*)>([&](const ReductionNode* op) {
      *reason = "concrete notation cannot contain reduction nodes";
      isConcrete = false;
    }),
    std::function<void(const SuchThatNode*)>([&](const SuchThatNode* op) {
      const string failed_reason = "concrete notation cannot contain nested SuchThat nodes";
      if (!isa<SuchThat>(stmt)) {
        *reason = failed_reason;
        isConcrete = false;
        return;
      }
      SuchThat firstSuchThat = to<SuchThat>(stmt);
      if (firstSuchThat != op) {
        *reason = failed_reason;
        isConcrete = false;
        return;
      }
    })
  );
  return isConcrete;
}

Assignment makeReductionNotation(Assignment assignment) {
  IndexExpr expr = assignment.getRhs();
  std::vector<IndexVar> free = assignment.getLhs().getIndexVars();
  if (!isEinsumNotation(assignment)) {
    return assignment;
  }

  struct MakeReductionNotation : IndexNotationRewriter {
    MakeReductionNotation(const std::vector<IndexVar>& free)
        : free(free.begin(), free.end()){}

    std::set<IndexVar> free;
    bool onlyOneTerm;

    IndexExpr addReductions(IndexExpr expr) {
      auto vars = getIndexVars(expr);
      for (auto& var : util::reverse(vars)) {
        if (!util::contains(free, var)) {
          expr = sum(var,expr);
        }
      }
      return expr;
    }

    IndexExpr einsum(const IndexExpr& expr) {
      onlyOneTerm = true;
      IndexExpr einsumexpr = rewrite(expr);

      if (onlyOneTerm) {
        einsumexpr = addReductions(einsumexpr);
      }

      return einsumexpr;
    }

    using IndexNotationRewriter::visit;

    void visit(const AddNode* op) {
      // Sum every reduction variables over each term
      onlyOneTerm = false;

      IndexExpr a = addReductions(op->a);
      IndexExpr b = addReductions(op->b);
      if (a == op->a && b == op->b) {
        expr = op;
      }
      else {
        expr = new AddNode(a, b);
      }
    }

    void visit(const SubNode* op) {
      // Sum every reduction variables over each term
      onlyOneTerm = false;

      IndexExpr a = addReductions(op->a);
      IndexExpr b = addReductions(op->b);
      if (a == op->a && b == op->b) {
        expr = op;
      }
      else {
        expr = new SubNode(a, b);
      }
    }
  };
  return Assignment(assignment.getLhs(),
                    MakeReductionNotation(free).einsum(expr),
                    assignment.getOperator());
}

IndexStmt makeReductionNotation(IndexStmt stmt) {
  taco_iassert(isEinsumNotation(stmt));
  return makeReductionNotation(to<Assignment>(stmt));
}

// Replace other reductions with where and forall statements
struct ReplaceReductionsWithWheres : IndexNotationRewriter {
  using IndexNotationRewriter::visit;

  Reduction reduction;
  TensorVar t;

  void visit(const AssignmentNode* node) {
    reduction = Reduction();
    t = TensorVar();

    IndexExpr rhs = rewrite(node->rhs);

    // nothing was rewritten
    if (rhs == node->rhs) {
      stmt = node;
      return;
    }

    taco_iassert(t.defined() && reduction.defined());
    IndexStmt consumer = Assignment(node->lhs, rhs, node->op);
    IndexStmt producer = forall(reduction.getVar(),
                                Assignment(t, reduction.getExpr(),
                                           reduction.getOp()));
    stmt = where(rewrite(consumer), rewrite(producer));
  }

  void visit(const ReductionNode* node) {
    // only rewrite one reduction at a time
    if (reduction.defined()) {
      expr = node;
      return;
    }

    reduction = node;
    t = TensorVar("t" + util::toString(node->var),
                  node->getDataType());
    expr = t;
  }
};

IndexStmt makeConcreteNotation(IndexStmt stmt) {
  std::string reason;
  taco_iassert(isReductionNotation(stmt, &reason))
      << "Not reduction notation: " << stmt << std::endl << reason;
  taco_iassert(isa<Assignment>(stmt));

  // Free variables and reductions covering the whole rhs become top level loops
  vector<IndexVar> freeVars = to<Assignment>(stmt).getFreeVars();

  struct RemoveTopLevelReductions : IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    void visit(const AssignmentNode* node) {
      // Easiest to just walk down the reduction node until we find something
      // that's not a reduction
      vector<IndexVar> topLevelReductions;
      IndexExpr rhs = node->rhs;
      IndexExpr reductionOp;
      while (isa<Reduction>(rhs)) {
        Reduction reduction = to<Reduction>(rhs);
        // Hack: explicit reductions with user defined functions shouldn't be rewritten.
        if (util::getFromEnv("TACO_CONCRETIZE_HACK", "0") != "0" && isa<Call>(reduction.getOp())) {
          break;
        }
        topLevelReductions.push_back(reduction.getVar());
        rhs = reduction.getExpr();
        reductionOp = reduction.getOp();
      }

      if (rhs != node->rhs) {
        stmt = Assignment(node->lhs, rhs, reductionOp);
        for (auto& i : util::reverse(topLevelReductions)) {
          stmt = forall(i, stmt);
        }
      }
      else {
        stmt = node;
      }
    }
  };
  stmt = RemoveTopLevelReductions().rewrite(stmt);

  for (auto& i : util::reverse(freeVars)) {
    stmt = forall(i, stmt);
  }

  stmt = ReplaceReductionsWithWheres().rewrite(stmt);
  return stmt;
}

Assignment makeReductionNotationScheduled(Assignment assignment, ProvenanceGraph provGraph) {
  IndexExpr expr = assignment.getRhs();
  std::vector<IndexVar> free = assignment.getLhs().getIndexVars();
  if (!isEinsumNotation(assignment)) {
    return assignment;
  }

  struct MakeReductionNotation : IndexNotationRewriter {
    MakeReductionNotation(const std::vector<IndexVar>& free, ProvenanceGraph provGraph)
      : free(free.begin(), free.end()), provGraph(provGraph){}

    std::set<IndexVar> free;
    ProvenanceGraph provGraph; 
    bool onlyOneTerm;

    IndexExpr addReductions(IndexExpr expr) {
      auto vars = getIndexVars(expr);
      for (auto& var : util::reverse(vars)) {

        if (!util::contains(free, var)) {
          bool shouldReduce = true;
          /// Do not add a reduction node if mismatch is between a fully derived indexVar and its ancestor
          if (provGraph.isFullyDerived(var)) {
            for (auto& f: free) {
              if (provGraph.isDerivedFrom(var, f)) {
                shouldReduce = false;
              }
            }
          } else {
            for (auto& f: free) {
              if (provGraph.isDerivedFrom(f, var)) {
                shouldReduce = false;
              }
            }
          }
          if (shouldReduce)
            expr = sum(var,expr);
        }
      }
      return expr;
    }

    IndexExpr einsum(const IndexExpr& expr) {
      onlyOneTerm = true;
      IndexExpr einsumexpr = rewrite(expr);

      if (onlyOneTerm) {
        einsumexpr = addReductions(einsumexpr);
      }

      return einsumexpr;
    }

    using IndexNotationRewriter::visit;

    void visit(const AddNode* op) {
      // Sum every reduction variables over each term
      onlyOneTerm = false;

      IndexExpr a = addReductions(op->a);
      IndexExpr b = addReductions(op->b);
      if (a == op->a && b == op->b) {
        expr = op;
      }
      else {
        expr = new AddNode(a, b);
      }
    }

    void visit(const SubNode* op) {
      // Sum every reduction variables over each term
      onlyOneTerm = false;

      IndexExpr a = addReductions(op->a);
      IndexExpr b = addReductions(op->b);
      if (a == op->a && b == op->b) {
        expr = op;
      }
      else {
        expr = new SubNode(a, b);
      }
    }
  };
  return Assignment(assignment.getLhs(),
                    MakeReductionNotation(free, provGraph).einsum(expr),
                    assignment.getOperator());
}

IndexStmt makeReductionNotationScheduled(IndexStmt stmt, ProvenanceGraph provGraph) {
  taco_iassert(isEinsumNotation(stmt));
  return makeReductionNotationScheduled(to<Assignment>(stmt), provGraph);
}

IndexStmt makeConcreteNotationScheduled(IndexStmt stmt, ProvenanceGraph provGraph, 
                                        vector<IndexVar> forallIndexVars) {
  std::string reason;
  taco_iassert(isReductionNotationScheduled(stmt, provGraph, &reason))
    << "Not reduction notation: " << stmt << std::endl << reason;
  taco_iassert(isa<Assignment>(stmt));

  // Free variables and reductions covering the whole rhs become top level loops
  vector<IndexVar> freeVars = to<Assignment>(stmt).getFreeVars();
  vector<IndexVar> reductionAndFreeVars;

  struct RemoveTopLevelReductions : IndexNotationRewriter {
    using IndexNotationRewriter::visit;
    vector<IndexVar> forallIndexVars;
    vector<IndexVar> reductionAndFreeVars;

    RemoveTopLevelReductions(vector<IndexVar> forallIndexVars) : forallIndexVars(forallIndexVars) {}

    void visit(const AssignmentNode* node) {
      // Easiest to just walk down the reduction node until we find something
      // that's not a reduction
      vector<IndexVar> topLevelReductions;
      IndexExpr rhs = node->rhs;
      while (isa<Reduction>(rhs)) {
        Reduction reduction = to<Reduction>(rhs);
        topLevelReductions.push_back(reduction.getVar());
        rhs = reduction.getExpr();
      }

      if (rhs != node->rhs) {
        stmt = Assignment(node->lhs, rhs, Add());
        if (forallIndexVars.empty()) {
          for (auto &i : util::reverse(topLevelReductions)) {
            stmt = forall(i, stmt);
          }
        } else {
          reductionAndFreeVars.insert(reductionAndFreeVars.end(), topLevelReductions.begin(), 
                                      topLevelReductions.end());
        }
      }
      else {
        stmt = node;
      }
    }
  };
  auto rewriter = RemoveTopLevelReductions(forallIndexVars);
  stmt = rewriter.rewrite(stmt);
  reductionAndFreeVars = rewriter.reductionAndFreeVars;
  // This gets the list of indexVars on the rhs of an assignment
  // TODO: check to make sure that we want to get ALL rhs indexVars (not just the upper level)
  vector<IndexVar> rhsVars;
  match(stmt,
        function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
          for (auto &i : op->indexVars) {
            if (std::find(rhsVars.begin(), rhsVars.end(), i) == rhsVars.end()) {
              rhsVars.push_back(i);
            }
          }
        }),
        function<void(const AssignmentNode*, Matcher*)>([&](const AssignmentNode* op, Matcher* ctx) {
          ctx->match(op->rhs);
        })
  );

  // Emit the freeVars as foralls if the freeVars are fully derived
  // else emit the fully derived descendant of the freeVar found in rhsVars
  if (forallIndexVars.empty()) {
    for (auto &i : util::reverse(freeVars)) {
      if (provGraph.isFullyDerived(i))
        stmt = forall(i, stmt);
      else {
        auto derivedVars = provGraph.getFullyDerivedDescendants(i);
        IndexVar derivedI = *rhsVars.begin();
        for (auto &derivedVar : derivedVars) {
          if (std::find(rhsVars.begin(), rhsVars.end(), derivedVar) != rhsVars.end()) {
            derivedI = derivedVar;
          }
        }
        stmt = forall(derivedI, stmt);
      }
    }
  } else {
    reductionAndFreeVars.insert(reductionAndFreeVars.end(), freeVars.begin(), freeVars.end());
    for (auto &i : util::reverse(forallIndexVars)) {
      if (std::find(reductionAndFreeVars.begin(), reductionAndFreeVars.end(), i) != reductionAndFreeVars.end())
        stmt = forall(i, stmt);
      else {
        auto ancestorVars = provGraph.getUnderivedAncestors(i);
        IndexVar ancestorI = *reductionAndFreeVars.begin();
        for (auto &ancestorVar : ancestorVars) {
          if (std::find(reductionAndFreeVars.begin(), reductionAndFreeVars.end(), ancestorVar) 
              != reductionAndFreeVars.end()) {
            stmt = forall(i, stmt);
          }
        }
      }
    }
  }

  stmt = ReplaceReductionsWithWheres().rewrite(stmt);
  return stmt;
}

vector<TensorVar> getResults(IndexStmt stmt) {
  vector<TensorVar> result;
  set<TensorVar> collected;

  for (auto& access : getResultAccesses(stmt).first) {
    TensorVar tensor = access.getTensorVar();
    taco_iassert(!util::contains(collected, tensor));
    collected.insert(tensor);
    result.push_back(tensor);
  }

  return result;
}


vector<TensorVar> getArguments(IndexStmt stmt) {
  vector<TensorVar> result;
  set<TensorVar> collected;

  for (auto& access : getArgumentAccesses(stmt)) {
    TensorVar tensor = access.getTensorVar();
    if (!util::contains(collected, tensor)) {
      collected.insert(tensor);
      result.push_back(tensor);
    }
    // The arguments will include any index sets on this tensor
    // argument as well.
    if (access.hasIndexSetModes()) {
      for (size_t i = 0; i < access.getIndexVars().size(); i++) {
        if (access.isModeIndexSet(i)) {
          auto t = access.getModeIndexSetTensor(i);
          if (!util::contains(collected, t)) {
            collected.insert(t);
            result.push_back(t);
          }
        }
      }
    }
  }

  return result;
}

bool allForFreeLoopsBeforeAllReductionLoops(IndexStmt stmt) {

    struct LoopOrderGetter : IndexNotationVisitor {

      std::vector<IndexVar> loopOrder;
      std::set<IndexVar> freeVars;

      using IndexNotationVisitor::visit;

      void visit(const AssignmentNode *op) {
        for (const auto &var : op->lhs.getIndexVars()) {
          freeVars.insert(var);
        }
        IndexNotationVisitor::visit(op);
      }

      void visit(const ForallNode *op) {
        loopOrder.push_back(op->indexVar);
        IndexNotationVisitor::visit(op);
      }
    };


    LoopOrderGetter getter;
    getter.visit(stmt);

    bool seenReductionVar = false;
    for (auto &var : getter.loopOrder) {
      if (util::contains(getter.freeVars, var)) {
        if (seenReductionVar) {
          // A reduction loop came before a loop over a free var
          return false;
        }
      } else {
        seenReductionVar = true;
      }
    }
    return true;
  }

std::map<Forall, Where> getTemporaryLocations(IndexStmt stmt) {
  map<Forall, Where> temporaryLocs;
  Forall f = Forall();
  match(stmt,
        function<void(const ForallNode*, Matcher*)>([&](const ForallNode* op, Matcher* ctx) {
          f = op;
          ctx->match(op->stmt);
        }),
          function<void(const WhereNode*, Matcher*)>([&](const WhereNode* w, Matcher* ctx) {
            if (!(f == IndexStmt()))
              temporaryLocs.insert({f, Where(w)});
          })
        );
  return temporaryLocs;
}


std::vector<TensorVar> getTemporaries(IndexStmt stmt) {
  vector<TensorVar> temporaries;
  bool firstAssignment = true;
  match(stmt,
    function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
      // Ignore the first assignment as its lhs is the result and not a temp.
      if (firstAssignment) {
        firstAssignment = false;
        return;
      }
      temporaries.push_back(op->lhs.getTensorVar());
    }),
    function<void(const SequenceNode*,Matcher*)>([&](const SequenceNode* op,
                                                     Matcher* ctx) {
      if (firstAssignment) {
        ctx->match(op->definition);
        firstAssignment = true;
        ctx->match(op->mutation);
      }
      else {
        ctx->match(op->definition);
        ctx->match(op->mutation);
      }
    }),
    function<void(const MultiNode*,Matcher*)>([&](const MultiNode* op,
                                                  Matcher* ctx) {
      if (firstAssignment) {
        ctx->match(op->stmt1);
        firstAssignment = true;
        ctx->match(op->stmt2);
      }
      else {
        ctx->match(op->stmt1);
        ctx->match(op->stmt2);
      }
    }),
    function<void(const WhereNode*,Matcher*)>([&](const WhereNode* op,
                                                  Matcher* ctx) {
      ctx->match(op->consumer);
      ctx->match(op->producer);
    }),
    function<void(const AssembleNode*,Matcher*)>([&](const AssembleNode* op,
                                                  Matcher* ctx) {
      ctx->match(op->compute);
      if (op->queries.defined()) {
        ctx->match(op->queries);
      }
    })
  );
  return temporaries;
}


std::vector<TensorVar> getAttrQueryResults(IndexStmt stmt) {
  std::vector<TensorVar> results;
  match(stmt,
    function<void(const AssembleNode*,Matcher*)>([&](const AssembleNode* op,
                                                  Matcher* ctx) {
      const auto queryResults = getResults(op->queries);
      results.insert(results.end(), queryResults.begin(), queryResults.end());
      if (op->queries.defined()) {
        ctx->match(op->queries);
      }
      ctx->match(op->compute);
    })
  );
  return results;
}


std::vector<TensorVar> getAssembledByUngroupedInsertion(IndexStmt stmt) {
  std::vector<TensorVar> results;
  match(stmt,
    function<void(const AssembleNode*,Matcher*)>([&](const AssembleNode* op,
                                                  Matcher* ctx) {
      for (const auto& result : op->results) {
        results.push_back(result.first);
      }
      if (op->queries.defined()) {
        ctx->match(op->queries);
      }
      ctx->match(op->compute);
    })
  );
  return results;
}


std::vector<TensorVar> getTensorVars(IndexStmt stmt) {
  vector<TensorVar> results = getResults(stmt);
  vector<TensorVar> arguments = getArguments(stmt);
  vector<TensorVar> temps = getTemporaries(stmt);
  return util::combine(results, util::combine(arguments, temps));
}


pair<vector<Access>,set<Access>> getResultAccesses(IndexStmt stmt)
{
  vector<Access> result;
  set<Access> reduced;

  match(stmt,
    function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
      taco_iassert(!util::contains(result, op->lhs));
      result.push_back(op->lhs);
      if (op->op.defined()) {
        reduced.insert(op->lhs);
      }
    }),
    function<void(const WhereNode*,Matcher*)>([&](const WhereNode* op,
                                                  Matcher* ctx) {
      ctx->match(op->consumer);
    }),
    function<void(const SequenceNode*,Matcher*)>([&](const SequenceNode* op,
                                                     Matcher* ctx) {
      ctx->match(op->definition);
    }),
    function<void(const AssembleNode*,Matcher*)>([&](const AssembleNode* op,
                                                     Matcher* ctx) {
      ctx->match(op->compute);
    })
  );
  return {result, reduced};
}


std::vector<Access> getArgumentAccesses(IndexStmt stmt)
{
  vector<Access> result;
  set<TensorVar> temporaries = util::toSet(getTemporaries(stmt));

  match(stmt,
    function<void(const AccessNode*)>([&](const AccessNode* n) {
      if (util::contains(temporaries, n->tensorVar)) {
        return;
      }
      result.push_back(n);
    }),
    function<void(const AssignmentNode*,Matcher*)>([&](const AssignmentNode* n,
                                                       Matcher* ctx) {
      ctx->match(n->rhs);
    })
  );

  return result;
}

// Return corresponding underived indexvars
struct GetIndexVars : IndexNotationVisitor {
  GetIndexVars(ProvenanceGraph provGraph) : provGraph(provGraph) {}
  vector<IndexVar> indexVars;
  set<IndexVar> seen;
  ProvenanceGraph provGraph;

  using IndexNotationVisitor::visit;

  void add(const vector<IndexVar>& vars) {
    for (auto& var : vars) {
      std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(var);
      for (auto &underived : underivedAncestors) {
        if (!util::contains(seen, underived)) {
          seen.insert(underived);
          indexVars.push_back(underived);
        }
      }
    }
  }

  void visit(const ForallNode* node) {
    add({node->indexVar});
    IndexNotationVisitor::visit(node->stmt);
  }

  void visit(const AccessNode* node) {
    add(node->indexVars);
  }

  void visit(const AssignmentNode* node) {
    add(node->lhs.getIndexVars());
    IndexNotationVisitor::visit(node->rhs);
  }
};

vector<IndexVar> getIndexVars(IndexStmt stmt) {
  GetIndexVars visitor = GetIndexVars(ProvenanceGraph(stmt));
  stmt.accept(&visitor);
  return visitor.indexVars;
}


vector<IndexVar> getIndexVars(IndexExpr expr) {
  GetIndexVars visitor = GetIndexVars(ProvenanceGraph());
  expr.accept(&visitor);
  return visitor.indexVars;
}

std::vector<IndexVar> getReductionVars(IndexStmt stmt) {
  const auto provGraph = ProvenanceGraph(stmt);

  std::vector<IndexVar> reductionVars, scopedVars, producerScopedVars, 
                        consumerScopedVars;
  match(stmt,
    function<void(const ForallNode*,Matcher*)>([&](const ForallNode* op, 
                                                   Matcher* ctx) {
      const auto indexVars = provGraph.getUnderivedAncestors(op->indexVar);
      for (const auto& iv : indexVars) {
        scopedVars.push_back(iv);
      }
      ctx->match(op->stmt);
      for (size_t i = 0; i < indexVars.size(); ++i) {
        scopedVars.pop_back();
      }
    }),
    function<void(const WhereNode*,Matcher*)>([&](const WhereNode* op,
                                                  Matcher* ctx) {
      const auto oldProducerScopedVars = producerScopedVars;
      producerScopedVars = scopedVars;
      ctx->match(op->producer);
      producerScopedVars = oldProducerScopedVars;

      const auto oldConsumerScopedVars = consumerScopedVars;
      consumerScopedVars = scopedVars;
      ctx->match(op->consumer);
      consumerScopedVars = oldConsumerScopedVars;
    }),
    function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
      auto freeVars = op->lhs.getIndexVars();
      util::append(freeVars, producerScopedVars);

      auto seen = util::toSet(freeVars);
      match(op->rhs,
        std::function<void(const AccessNode*)>([&](const AccessNode* op) {
          for (const auto& var : op->indexVars) {
            if (!util::contains(seen, var)) {
              reductionVars.push_back(var);
              seen.insert(var);
            }
          }
        })
      );
      for (const auto& var : consumerScopedVars) {
        if (!util::contains(seen, var)) {
          reductionVars.push_back(var);
          seen.insert(var);
        }
      }
    })
  );
  return reductionVars;
}

vector<ir::Expr> createVars(const vector<TensorVar>& tensorVars,
                            map<TensorVar, ir::Expr>* vars, 
                            bool isParameter) {
  taco_iassert(vars != nullptr);
  vector<ir::Expr> irVars;
  for (auto& var : tensorVars) {
    ir::Expr irVar = ir::Var::make(var.getName(), var.getType().getDataType(),
                                   true, true, isParameter);
    irVars.push_back(irVar);
    vars->insert({var, irVar});
  }
  return irVars;
}

std::map<TensorVar,ir::Expr> createIRTensorVars(IndexStmt stmt)
{
  std::map<TensorVar,ir::Expr> tensorVars;

  // Create result and parameter variables
  vector<TensorVar> results = getResults(stmt);
  vector<TensorVar> arguments = getArguments(stmt);
  vector<TensorVar> temporaries = getTemporaries(stmt);

  // Create variables for index sets on result tensors.
  for (auto& access : getResultAccesses(stmt).first) {
    // Any accesses that have index sets will be added.
    if (access.hasIndexSetModes()) {
      for (size_t i = 0; i < access.getIndexVars().size(); i++) {
        if (access.isModeIndexSet(i)) {
          auto t = access.getModeIndexSetTensor(i);
          if (tensorVars.count(t) == 0) {
            ir::Expr irVar = ir::Var::make(t.getName(), t.getType().getDataType(), true, true, true);
            tensorVars.insert({t, irVar});
          }
        }
      }
    }
  }

  // Convert tensor results, arguments and temporaries to IR variables
  map<TensorVar, ir::Expr> resultVars;
  vector<ir::Expr> resultsIR = createVars(results, &resultVars);
  tensorVars.insert(resultVars.begin(), resultVars.end());
  vector<ir::Expr> argumentsIR = createVars(arguments, &tensorVars);
  vector<ir::Expr> temporariesIR = createVars(temporaries, &tensorVars);

  return tensorVars;
}

struct Zero : public IndexNotationRewriterStrict {
public:
  Zero(const set<Access>& zeroed) : zeroed(zeroed) {}

private:
  using IndexExprRewriterStrict::visit;

  set<Access> zeroed;

  /// Temporary variables whose assignment has become zero.  These are therefore
  /// zero at every access site.
  set<TensorVar> zeroedVars;

  void visit(const AccessNode* op) {
    if (util::contains(zeroed, op) ||
        util::contains(zeroedVars, op->tensorVar)) {
      expr = IndexExpr();
    }
    else {
      expr = op;
    }
  }

  void visit(const LiteralNode* op) {
    expr = op;
  }

  void visit(const IndexVarNode* op) {
    expr = op;
  }

  template <class T>
  IndexExpr visitUnaryOp(const T *op) {
    IndexExpr a = rewrite(op->a);
    if (!a.defined()) {
      return IndexExpr();
    }
    else if (a == op->a) {
      return op;
    }
    else {
      return new T(a);
    }
  }

  void visit(const NegNode* op) {
    expr = visitUnaryOp(op);
  }

  void visit(const SqrtNode* op) {
    expr = visitUnaryOp(op);
  }

  template <class T>
  IndexExpr visitDisjunctionOp(const T *op) {
    IndexExpr a = rewrite(op->a);
    IndexExpr b = rewrite(op->b);
    if (!a.defined() && !b.defined()) {
      return IndexExpr();
    }
    else if (!a.defined()) {
      return b;
    }
    else if (!b.defined()) {
      return a;
    }
    else if (a == op->a && b == op->b) {
      return op;
    }
    else {
      return new T(a, b);
    }
  }

  template <class T>
  IndexExpr visitConjunctionOp(const T *op) {
    IndexExpr a = rewrite(op->a);
    IndexExpr b = rewrite(op->b);
    if (!a.defined() || !b.defined()) {
      return IndexExpr();
    }
    else if (a == op->a && b == op->b) {
      return op;
    }
    else {
      return new T(a, b);
    }
  }

  void visit(const AddNode* op) {
    expr = visitDisjunctionOp(op);
  }

  void visit(const SubNode* op) {
    IndexExpr a = rewrite(op->a);
    IndexExpr b = rewrite(op->b);
    if (!a.defined() && !b.defined()) {
      expr = IndexExpr();
    }
    else if (!a.defined()) {
      expr = -b;
    }
    else if (!b.defined()) {
      expr = a;
    }
    else if (a == op->a && b == op->b) {
      expr = op;
    }
    else {
      expr = new SubNode(a, b);
    }
  }

  void visit(const MulNode* op) {
    expr = visitConjunctionOp(op);
  }

  void visit(const DivNode* op) {
    expr = visitConjunctionOp(op);
  }

  void visit(const CastNode* op) {
    IndexExpr a = rewrite(op->a);
    if (!a.defined()) {
      expr = IndexExpr();
    }
    else if (a == op->a) {
      expr = op;
    }
    else {
      expr = new CastNode(a, op->getDataType());
    }
  }

  void visit(const CallNode* op) {
    std::vector<IndexExpr> args;
    std::vector<IndexExpr> rewrittenArgs;
    std::vector<int> definedArgs;
    bool rewritten = false;

    Annihilator annihilator = findProperty<Annihilator>(op->properties);

    // TODO: Check exhausted default against result default
    for(int argIdx = 0; argIdx < (int) op->args.size(); ++argIdx) {
      IndexExpr arg = op->args[argIdx];
      IndexExpr rewrittenArg = rewrite(arg);
      rewrittenArgs.push_back(rewrittenArg);

      if (rewrittenArg.defined()) {
        definedArgs.push_back(argIdx);
      } else {
        // TODO: fill value instead of 0
        rewrittenArg = Literal::zero(arg.getDataType());
      }

      args.push_back(rewrittenArg);
      if (arg != rewrittenArg) {
        rewritten = true;
      }
    }

    if(annihilator.defined()) {
      IndexExpr e = annihilator.annihilates(args);
      if(e.defined()) {
        expr = e;
        return;
      }
    }

    Identity identity = findProperty<Identity>(op->properties);
    if(identity.defined()) {
      IndexExpr e = identity.simplify(args);
      if(e.defined()) {
        expr = e;
        return;
      }
    }

    if (rewritten) {
      const std::map<IndexExpr, IndexExpr> subs = util::zipToMap(op->args, rewrittenArgs);
      IterationAlgebra newAlg = replaceAlgIndexExprs(op->iterAlg, subs);
      expr = new CallNode(op->name, args, op->defaultLowerFunc, newAlg, op->properties,
                          op->regionDefinitions, definedArgs);
    }
    else {
      expr = op;
    }

  }

  void visit(const CallIntrinsicNode* op) {
    std::vector<IndexExpr> args;
    std::vector<size_t> zeroArgs;
    bool rewritten = false;
    for (size_t i = 0; i < op->args.size(); ++i) {
      IndexExpr arg = op->args[i];
      IndexExpr rewrittenArg = rewrite(arg);
      if (!rewrittenArg.defined()) {
        rewrittenArg = Literal::zero(arg.getDataType());
        zeroArgs.push_back(i);
      }
      args.push_back(rewrittenArg);
      if (arg != rewrittenArg) {
        rewritten = true;
      }
    }
    const auto zeroPreservingArgsSets = op->func->zeroPreservingArgs(args);
    for (const auto& zeroPreservingArgs : zeroPreservingArgsSets) {
      taco_iassert(!zeroPreservingArgs.empty());
      if (std::includes(zeroArgs.begin(), zeroArgs.end(),
                        zeroPreservingArgs.begin(), zeroPreservingArgs.end())) {
        expr = IndexExpr();
        return;
      }
    }
    if (rewritten) {
      expr = new CallIntrinsicNode(op->func, args);
    }
    else {
      expr = op;
    }
  }

  void visit(const ReductionNode* op) {
    IndexExpr a = rewrite(op->a);
    if (!a.defined()) {
      expr = IndexExpr();
    }
    else if (a == op->a) {
      expr = op;
    }
    else {
      expr = new ReductionNode(op->op, op->var, a);
    }
  }

  void visit(const AssignmentNode* op) {
    IndexExpr rhs = rewrite(op->rhs);
    if (!rhs.defined()) {
      stmt = IndexStmt();
      zeroedVars.insert(op->lhs.getTensorVar());
    }
    else if (rhs == op->rhs) {
      stmt = op;
    }
    else {
      stmt = new AssignmentNode(op->lhs, rhs, op->op);
    }
  }

  void visit(const YieldNode* op) {
    IndexExpr expr = rewrite(op->expr);
    if (expr == op->expr) {
      stmt = op;
    }
    else {
      stmt = new YieldNode(op->indexVars, expr);
    }
  }

  void visit(const ForallNode* op) {
    IndexStmt body = rewrite(op->stmt);
    if (!body.defined()) {
      stmt = IndexStmt();
    }
    else if (body == op->stmt) {
      stmt = op;
    }
    else {
      stmt = new ForallNode(op->indexVar, body, op->parallel_unit, op->output_race_strategy, op->unrollFactor);
    }
  }

  void visit(const WhereNode* op) {
    IndexStmt producer = rewrite(op->producer);
    IndexStmt consumer = rewrite(op->consumer);
    if (!consumer.defined()) {
      stmt = IndexStmt();
    }
    else if (!producer.defined()) {
      stmt = consumer;
    }
    else if (producer == op->producer && consumer == op->consumer) {
      stmt = op;
    }
    else {
      stmt = new WhereNode(consumer, producer);
    }
  }

  void visit(const SequenceNode* op) {
    taco_not_supported_yet;
  }

  void visit(const AssembleNode* op) {
    taco_not_supported_yet;
  }

  void visit(const MultiNode* op) {
    taco_not_supported_yet;
  }

  void visit(const SuchThatNode* op) {
    IndexStmt body = rewrite(op->stmt);
    if (!body.defined()) {
      stmt = IndexStmt();
    }
    else if (body == op->stmt) {
      stmt = op;
    }
    else {
      stmt = new SuchThatNode(body, op->predicate);
    }
  }
};

IndexExpr zero(IndexExpr expr, const set<Access>& zeroed) {
  return Zero(zeroed).rewrite(expr);
}

IndexStmt zero(IndexStmt stmt, const std::set<Access>& zeroed) {
  return Zero(zeroed).rewrite(stmt);
}

// Attempts to infer the fill value of a given expression. If we cannot infer the value, an empty expression
// is returned
struct fillValueInferrer : IndexExprRewriterStrict {
  public:
    virtual void visit(const AccessNode* op) {
      expr = op->tensorVar.getFill();
    };

    virtual void visit(const LiteralNode* op) {
      expr = op;
    }

    virtual void visit(const NegNode* op) {
      IndexExpr a = rewrite(op->a);
      if(equals(a, Literal::zero(a.getDataType()))) {
        expr = a;
        return;
      }
      expr = IndexExpr();
    }

    virtual void visit(const AddNode* op) {
      IndexExpr a = rewrite(op->a);
      IndexExpr b = rewrite(op->b);

      if(equals(a, Literal::zero(a.getDataType())) && isa<Literal>(b)) {
        expr = b;
        return;
      }

      if(equals(b, Literal::zero(b.getDataType())) && isa<Literal>(a)) {
        expr = a;
        return;
      }

      expr = IndexExpr();
    }

    virtual void visit(const SubNode* op) {
      IndexExpr a = rewrite(op->a);
      IndexExpr b = rewrite(op->b);

      if(equals(b, Literal::zero(b.getDataType())) && isa<Literal>(a)) {
        expr = a;
        return;
      }

      expr = IndexExpr();
    }

    virtual void visit(const MulNode* op) {
      IndexExpr a = rewrite(op->a);
      IndexExpr b = rewrite(op->b);

      if(equals(a, Literal::zero(a.getDataType()))) {
        expr = a;
        return;
      }

      if(equals(b, Literal::zero(b.getDataType()))) {
        expr = b;
        return;
      }

      expr = IndexExpr();
    }

    virtual void visit(const DivNode* op) {
      IndexExpr a = rewrite(op->a);
      IndexExpr b = rewrite(op->b);

      if(equals(a, Literal::zero(a.getDataType()))) {
        expr = a;
        return;
      }

      expr = IndexExpr();
    }

    virtual void visit(const SqrtNode* op) {
      IndexExpr a = rewrite(op->a);
      if(equals(a, Literal::zero(a.getDataType()))) {
        expr = a;
        return;
      }
      expr = IndexExpr();
    }

    virtual void visit(const CastNode* op) {
      expr = IndexExpr();
    }

    virtual void visit(const CallNode* op) {
      Annihilator annihilator = findProperty<Annihilator>(op->properties);
      if(annihilator.defined()) {
        IndexExpr e = annihilator.annihilates(op->args);
        if(e.defined()) {
          expr = e;
          return;
        }
      }

      Identity identity = findProperty<Identity>(op->properties);
      if(identity.defined()) {
        IndexExpr e = identity.simplify(op->args);
        if(e.defined()) {
          expr = e;
          return;
        }
      }

      expr = IndexExpr();
    }

    virtual void visit(const CallIntrinsicNode*) {
      // TODO Implement or remove this
      taco_not_supported_yet;
    }

    virtual void visit(const ReductionNode*) {
      expr = IndexExpr();
    }

    virtual void visit(const IndexVarNode*) {
      expr = IndexExpr();
    }
  };


IndexExpr inferFill(IndexExpr expr) {
  return fillValueInferrer().rewrite(expr);
}

bool hasNoForAlls(IndexStmt stmt) {

  bool noForAlls = true;
  match(stmt,
        std::function<void(const ForallNode*)>([&](const ForallNode* op) {
          noForAlls = false;
        })
  );
  return noForAlls;
}

IndexStmt generatePackStmt(TensorVar tensor, 
                           std::string otherName, Format otherFormat, 
                           std::vector<IndexVar> indexVars, 
                           bool otherIsOnRight) { 

  const Type type = tensor.getType();
  TensorVar other(otherName, type, otherFormat);

  const Format format = tensor.getFormat();
  IndexStmt packStmt = otherIsOnRight ? 
                       (tensor(indexVars) = other(indexVars)) : 
                       (other(indexVars) = tensor(indexVars));

  for (int i = format.getOrder() - 1; i >= 0; --i) {
    int mode = format.getModeOrdering()[i];
    packStmt = forall(indexVars[mode], packStmt);
  }

  bool doAppend = true;
  const Format lhsFormat = otherIsOnRight ? format : otherFormat;
  for (int i = lhsFormat.getOrder() - 1; i >= 0; --i) {
    const auto modeFormat = lhsFormat.getModeFormats()[i];
    if (modeFormat.isBranchless() && i != 0) {
      const auto parentModeFormat = lhsFormat.getModeFormats()[i - 1];
      if (parentModeFormat.isUnique() || !parentModeFormat.hasAppend()) {
        doAppend = false;
        break;
      }
    }
  }
  if (!doAppend) {
    packStmt = packStmt.assemble(otherIsOnRight ? tensor : other, AssembleStrategy::Insert);
  }

  return packStmt; 
}

IndexStmt generatePackCOOStmt(TensorVar tensor, 
                              std::vector<IndexVar> indexVars, bool otherIsOnRight) {

  const std::string tensorName = tensor.getName();
  const Format format = tensor.getFormat();

  const Format bufferFormat = COO(format.getOrder(), false, true, false, 
                                  format.getModeOrdering());

  return generatePackStmt(tensor, tensorName + "_COO", bufferFormat, indexVars, otherIsOnRight);
}
}
