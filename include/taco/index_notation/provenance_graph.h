#ifndef TACO_PROVENANCE_GRAPH_H
#define TACO_PROVENANCE_GRAPH_H

#include "taco/lower/iterator.h"

namespace taco {
struct IndexVarRelNode;
enum IndexVarRelType {UNDEFINED, SPLIT, DIVIDE, POS, FUSE, BOUND, PRECOMPUTE};

/// A pointer class for IndexVarRelNodes provides some operations for all IndexVarRelTypes
class IndexVarRel : public util::IntrusivePtr<const IndexVarRelNode> {
public:
  IndexVarRel() : IntrusivePtr(nullptr) {}
  IndexVarRel(IndexVarRelNode* node) : IntrusivePtr(node) {}
  void print(std::ostream& stream) const;
  bool equals(const IndexVarRel &rel) const;
  IndexVarRelType getRelType() const;

  template<typename T>
  const T* getNode() const {
    return static_cast<const T*>(ptr);
  }

  const IndexVarRelNode* getNode() const {
    return ptr;
  }
};

std::ostream& operator<<(std::ostream&, const IndexVarRel&);
bool operator==(const IndexVarRel&, const IndexVarRel&);

/// Index variable relations are used to track how new index variables are derived
/// in the scheduling language
struct IndexVarRelNode : public util::Manageable<IndexVarRelNode>,
                         private util::Uncopyable {
  IndexVarRelNode() : relType(UNDEFINED) {}
  IndexVarRelNode(IndexVarRelType type) : relType(type) {}
  virtual ~IndexVarRelNode() = default;
  virtual void print(std::ostream& stream) const {
    taco_iassert(relType == UNDEFINED);
    stream << "underived";
  }

  /// returns list of index variables that are derived from
  virtual std::vector<IndexVar> getParents() const {
    taco_ierror;
    return {};
  }

  /// returns list of index variables that are newly derived
  virtual std::vector<IndexVar> getChildren() const {
    taco_ierror;
    return {};
  }

  /// returns list of index variables that are sized based on the size of the parent index variables (ie. outer for split)
  virtual std::vector<IndexVar> getIrregulars() const {
    taco_ierror;
    return {};
  }

  /// Given coordinate bounds for parents, determine the new coordinate bounds relative to this possibly fused space
  virtual std::vector<ir::Expr> computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const;

  /// Determine the iteration bounds for a newly derived index variable given the iteration bounds of the parents
  virtual std::vector<ir::Expr> deriveIterBounds(IndexVar indexVar, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                                                 std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                                 Iterators iterators, ProvenanceGraph provGraph) const;

  /// Recover a parent index variable expression as a function of the children index variables
  virtual ir::Expr recoverVariable(IndexVar indexVar, std::map<IndexVar, ir::Expr> variableNames, Iterators iterators, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, ProvenanceGraph provGraph) const;

  /// Recover a child index variable expression as a function of all other variables (parents and siblings) in the relationship
  virtual ir::Stmt recoverChild(IndexVar indexVar, std::map<IndexVar, ir::Expr> variableNames, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const;
  IndexVarRelType relType;
};

/// The split relation takes a parentVar's iteration space and stripmines into an outervar that iterates over splitFactor-sized
/// iterations over innerVar
struct SplitRelNode : public IndexVarRelNode {
  SplitRelNode(IndexVar parentVar, IndexVar outerVar, IndexVar innerVar, size_t splitFactor);

  const IndexVar& getParentVar() const;
  const IndexVar& getOuterVar() const;
  const IndexVar& getInnerVar() const;
  const size_t& getSplitFactor() const;

  void print(std::ostream& stream) const;
  bool equals(const SplitRelNode &rel) const;
  std::vector<IndexVar> getParents() const; // parentVar
  std::vector<IndexVar> getChildren() const; // outerVar, innerVar
  std::vector<IndexVar> getIrregulars() const; // outerVar

  /// if parent is in position space then bound is just the parent's bound
  /// if innerVar defined and not outerVar or if neither variables are defined then return the parent's bound
  /// if the outerVar is already defined then the inner var constrains bound to splitFactor-sized strip at outerVar * splitFactor
  /// if both variables are defined then constrain to single length 1 strip at outerVar * splitFactor + innerVar
  std::vector<ir::Expr> computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const;

  /// outerVar has parentBounds / splitFactor and innerVar has 0 -> splitFactor
  std::vector<ir::Expr> deriveIterBounds(IndexVar indexVar, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators, ProvenanceGraph provGraph) const;

  /// parentVar = outerVar * splitFactor + innerVar
  ir::Expr recoverVariable(IndexVar indexVar, std::map<IndexVar, ir::Expr> variableNames, Iterators iterators, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, ProvenanceGraph provGraph) const;

  /// outerVar = parentVar - innerVar, innerVar = parentVar - outerVar * splitFactor
  ir::Stmt recoverChild(IndexVar indexVar, std::map<IndexVar, ir::Expr> relVariables, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

bool operator==(const SplitRelNode&, const SplitRelNode&);

// DivideRelNode takes a parentVar's iteration space and divides it into divFactor
// equal pieces. outerVar iterates over the number of pieces, and innerVar iterates
// over each piece.
  struct DivideRelNode : public IndexVarRelNode {
    DivideRelNode(IndexVar parentVar, IndexVar outerVar, IndexVar innerVar, size_t divFactor);

    const IndexVar &getParentVar() const;

    const IndexVar &getOuterVar() const;

    const IndexVar &getInnerVar() const;

    const size_t &getDivFactor() const;

    void print(std::ostream &stream) const;

    bool equals(const DivideRelNode &rel) const;

    std::vector<IndexVar> getParents() const; // parentVar
    std::vector<IndexVar> getChildren() const; // outerVar, innerVar
    std::vector<IndexVar> getIrregulars() const; // innerVar

    // computeRelativeBound performs similar logic to SplitRelNode::computeRelativeBound.
    std::vector<ir::Expr>
    computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds,
                         std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators,
                         ProvenanceGraph provGraph) const;

    /// outerVar has bounds 0 -> divFactor and innerVar has parentBounds / divFactor.
    std::vector<ir::Expr>
    deriveIterBounds(IndexVar indexVar, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                     std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                     std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators,
                     ProvenanceGraph provGraph) const;

    /// parentVar = outerVar * (parentBounds / divFactor) + innerVar.
    ir::Expr recoverVariable(IndexVar indexVar, std::map<IndexVar, ir::Expr> variableNames, Iterators iterators,
                             std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds,
                             std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds,
                             ProvenanceGraph provGraph) const;

    /// outerVar = parentVar / (parentBounds / divFactor), innerVar = parentVar - outerVar * (parentBounds / divFactor).
    ir::Stmt
    recoverChild(IndexVar indexVar, std::map<IndexVar, ir::Expr> relVariables, bool emitVarDecl, Iterators iterators,
                 ProvenanceGraph provGraph) const;

  private:
    struct Content;
    std::shared_ptr<Content> content;
  };

bool operator==(const DivideRelNode&, const DivideRelNode&);

/// The Pos relation maps an index variable to the position space of a given access
struct PosRelNode : public IndexVarRelNode {
  PosRelNode(IndexVar i, IndexVar ipos, const Access& access);

  const IndexVar& getParentVar() const;
  const IndexVar& getPosVar() const;
  const Access& getAccess() const;

  void print(std::ostream& stream) const;
  bool equals(const PosRelNode &rel) const;
  std::vector<IndexVar> getParents() const; // parentVar
  std::vector<IndexVar> getChildren() const; // posVar
  std::vector<IndexVar> getIrregulars() const; // posVar

  /// Coordinate bounds remain unchanged
  std::vector<ir::Expr> computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const;

  /// get length of position array
  std::vector<ir::Expr> deriveIterBounds(IndexVar indexVar, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators, ProvenanceGraph provGraph) const;

  /// look up coord in coordinate array
  ir::Expr recoverVariable(IndexVar indexVar, std::map<IndexVar, ir::Expr> variableNames, Iterators iterators, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, ProvenanceGraph provGraph) const;

  /// Search for position based on coordinate of parentVar
  ir::Stmt recoverChild(IndexVar indexVar, std::map<IndexVar, ir::Expr> relVariables, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const;

private:
  /// Gets the coord array for the access iterator
  ir::Expr getAccessCoordArray(Iterators iterators, ProvenanceGraph provGraph) const;

  /// Finds the iterator that corresponds to the given access and indexvar
  Iterator getAccessIterator(Iterators iterators, ProvenanceGraph provGraph) const;

  /// Search for position bounds given coordinate bounds
  std::vector<ir::Expr> locateBounds(std::vector<ir::Expr> coordBounds,
                                                 Datatype boundType,
                                                 Iterators iterators,
                                                 ProvenanceGraph provGraph) const;
  struct Content;
  std::shared_ptr<Content> content;
};

bool operator==(const PosRelNode&, const PosRelNode&);

/// The fuse relation fuses the iteration space of two directly nested index variables
struct FuseRelNode : public IndexVarRelNode {
  FuseRelNode(IndexVar outerParentVar, IndexVar innerParentVar, IndexVar fusedVar);

  const IndexVar& getOuterParentVar() const;
  const IndexVar& getInnerParentVar() const;
  const IndexVar& getFusedVar() const;

  void print(std::ostream& stream) const;
  bool equals(const FuseRelNode &rel) const;
  std::vector<IndexVar> getParents() const; // outerParentVar, innerParentVar
  std::vector<IndexVar> getChildren() const; // fusedVar
  std::vector<IndexVar> getIrregulars() const; // fusedVar

  /// outerParentVar bound * innerParentVar bound
  std::vector<ir::Expr> computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const;

  /// outerParentVar bound * innerParentVar bound
  std::vector<ir::Expr> deriveIterBounds(IndexVar indexVar, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators, ProvenanceGraph provGraph) const;

  /// outerParentVar = fusedVar / innerSize, innerParentVar = fusedVar % innerSize
  ir::Expr recoverVariable(IndexVar indexVar, std::map<IndexVar, ir::Expr> variableNames, Iterators iterators, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, ProvenanceGraph provGraph) const;

  /// outerParentVar * innerSize + innerParentVar
  ir::Stmt recoverChild(IndexVar indexVar, std::map<IndexVar, ir::Expr> relVariables, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const;
private:
  /// returns combined parent var bounds taking into account both min and max bounds
  std::vector<ir::Expr> combineParentBounds(std::vector<ir::Expr> outerParentBound, std::vector<ir::Expr> innerParentBound) const;
  struct Content;
  std::shared_ptr<Content> content;
};

bool operator==(const FuseRelNode&, const FuseRelNode&);

/// The bound relation allows expressing a constraint or value known at compile-time that allows for compile-time optimizations
struct BoundRelNode : public IndexVarRelNode {
  BoundRelNode(IndexVar parentVar, IndexVar boundVar, size_t bound, BoundType boundType);

  const IndexVar& getParentVar() const;
  const IndexVar& getBoundVar() const;
  const size_t& getBound() const;
  const BoundType& getBoundType() const;

  void print(std::ostream& stream) const;
  bool equals(const BoundRelNode &rel) const;
  std::vector<IndexVar> getParents() const; // parentVar
  std::vector<IndexVar> getChildren() const; // boundVar
  std::vector<IndexVar> getIrregulars() const; // boundVar

  /// Coordinate bounds remain unchanged, only iteration bounds change
  std::vector<ir::Expr> computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const;

  /// Constrained depending on bound_type
  std::vector<ir::Expr> deriveIterBounds(IndexVar indexVar, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators, ProvenanceGraph provGraph) const;

  /// parentVar = boundVar
  ir::Expr recoverVariable(IndexVar indexVar, std::map<IndexVar, ir::Expr> variableNames, Iterators iterators, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, ProvenanceGraph provGraph) const;

  /// boundVar = parentVar
  ir::Stmt recoverChild(IndexVar indexVar, std::map<IndexVar, ir::Expr> relVariables, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const;
private:
  struct Content;
  std::shared_ptr<Content> content;
};

bool operator==(const BoundRelNode&, const BoundRelNode&);

/// The precompute relation allows creating a new precomputeVar that is iterated over for the precompute loop and shares same sizes as parentVar
/// This allows precomputeVar to be scheduled separately from the parentVar
struct PrecomputeRelNode : public IndexVarRelNode {
  PrecomputeRelNode(IndexVar parentVar, IndexVar precomputeVar);

  const IndexVar& getParentVar() const;
  const IndexVar& getPrecomputeVar() const;

  void print(std::ostream& stream) const;
  bool equals(const PrecomputeRelNode &rel) const;
  std::vector<IndexVar> getParents() const; // parentVar
  std::vector<IndexVar> getChildren() const; // precomputeVar
  std::vector<IndexVar> getIrregulars() const; // precomputeVar

  /// all bounds remain unchanged and parentVar = precomputeVar
  std::vector<ir::Expr> computeRelativeBound(std::set<IndexVar> definedVars, std::map<IndexVar, std::vector<ir::Expr>> computedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators, ProvenanceGraph provGraph) const;
  std::vector<ir::Expr> deriveIterBounds(IndexVar indexVar, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators, ProvenanceGraph provGraph) const;
  ir::Expr recoverVariable(IndexVar indexVar, std::map<IndexVar, ir::Expr> variableNames, Iterators iterators, std::map<IndexVar, std::vector<ir::Expr>> parentIterBounds, std::map<IndexVar, std::vector<ir::Expr>> parentCoordBounds, ProvenanceGraph provGraph) const;
  ir::Stmt recoverChild(IndexVar indexVar, std::map<IndexVar, ir::Expr> relVariables, bool emitVarDecl, Iterators iterators, ProvenanceGraph provGraph) const;
private:
  struct Content;
  std::shared_ptr<Content> content;
};

bool operator==(const PrecomputeRelNode&, const PrecomputeRelNode&);


/// An IndexVarprovGraph is a side IR that takes in Concrete Index Notation and supports querying
/// relationships between IndexVars. Gets relationships from SuchThat node in Concrete Index Notation
class ProvenanceGraph {
public:
  ProvenanceGraph() {}
  ProvenanceGraph(IndexStmt concreteStmt);

  /// Returns the children of a given index variable, {} if no children or if indexVar is not in graph
  std::vector<IndexVar> getChildren(IndexVar indexVar) const;

  /// Returns the parents of a given index variable, {} if no parents or if indexVar is not in graph
  std::vector<IndexVar> getParents(IndexVar indexVar) const;

  /// Retrieves descendants that are fully derived
  std::vector<IndexVar> getFullyDerivedDescendants(IndexVar indexVar) const;

  /// Retrieves ancestors that are underived
  std::vector<IndexVar> getUnderivedAncestors(IndexVar indexVar) const;

  /// Retrieves fully derived descendant that is irregular return true if one exists else false
  bool getIrregularDescendant(IndexVar indexVar, IndexVar *irregularChild) const;

  /// Returns first ancestor such that its parent is coord and it is pos
  bool getPosIteratorAncestor(IndexVar indexVar, IndexVar *irregularChild) const;

  /// Returns first descendant such that its parent is coord and it is pos
  bool getPosIteratorDescendant(IndexVar indexVar, IndexVar *irregularChild) const;

  /// Returns innermost pos variable that can be used to directly iterate over positions
  bool getPosIteratorFullyDerivedDescendant(IndexVar indexVar, IndexVar *irregularChild) const;

  /// Node is irregular if its size depends on the input (otherwise is static)
  /// A node is irregular if there exists a path to an underived ancestor that does not fix size
  bool isIrregular(IndexVar indexVar) const;

  /// Node is underived if has no parents
  bool isUnderived(IndexVar indexVar) const;

  /// is indexVar derived from ancestor
  bool isDerivedFrom(IndexVar indexVar, IndexVar ancestor) const;

  /// Node is fully derived if has no children
  bool isFullyDerived(IndexVar indexVar) const;

  /// Node is available if parents appear in defined
  bool isAvailable(IndexVar indexVar, std::set<IndexVar> defined) const;

  /// Node is recoverable if children appear in defined
  bool isRecoverable(IndexVar indexVar, std::set<IndexVar> defined) const;

  /// isRecoverable helper method to handle precompute relations and where statements in the provenance graph
  bool isRecoverablePrecompute(IndexVar indexVar, std::set<IndexVar> defined, std::vector<IndexVar> producers, std::vector<IndexVar> consumers) const;

  /// Node is recoverable if at most 1 unknown variable in relationship (parents + siblings)
  bool isChildRecoverable(taco::IndexVar indexVar, std::set<taco::IndexVar> defined) const;

  /// Return bounds with respect to underived coordinate space. Used for constructing guards and determining binary search target
  std::map<IndexVar, std::vector<ir::Expr>> deriveCoordBounds(std::vector<IndexVar> definedVarOrder, std::map<IndexVar, std::vector<ir::Expr>> underivedBounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators) const;

  /// adds relative bounds for indexVar and all ancestors to map. Used in deriveCoordBounds to simplify logic
  void addRelativeBoundsToMap(IndexVar indexVar, std::set<IndexVar> alreadyDefined, std::map<IndexVar, std::vector<ir::Expr>> &bounds, std::map<IndexVar, ir::Expr> variableExprs, Iterators iterators) const;

  /// takes relative bounds and propagates backwards to underived ancestors (note: might be more than one due to fuse)
  void computeBoundsForUnderivedAncestors(IndexVar indexVar, std::map<IndexVar, std::vector<ir::Expr>> relativeBounds, std::map<IndexVar, std::vector<ir::Expr>> &computedBounds) const;

  /// Returns iteration bounds of indexVar used for determining loop bounds.
  std::vector<ir::Expr> deriveIterBounds(IndexVar indexVar, std::vector<IndexVar> definedVarOrder, std::map<IndexVar, std::vector<ir::Expr>> underivedBounds, std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators) const;

  /// Returns true if index variable has constrained coordinate bounds because of a parent relation
  bool hasCoordBounds(IndexVar indexVar) const;

  /// whether or not the index variable is in position space
  bool isPosVariable(IndexVar indexVar) const;

  /// whether or not the index variable is in coordinate space
  bool isCoordVariable(IndexVar indexVar) const;

  /// whether or not the index variable is in the position space of a given access
  bool isPosOfAccess(IndexVar indexVar, Access access) const;

  /// does the index variable have a descendant in position space
  bool hasPosDescendant(IndexVar indexVar) const;

  /// does the index variable have an exact bound known at compile-time
  bool hasExactBound(IndexVar indexVar) const;

  /// Once indexVar is defined what new variables become recoverable
  /// returned in order of recovery (ie if parent being recovered allows its parent to also be recovered then parent comes first)
  std::vector<IndexVar> newlyRecoverableParents(IndexVar indexVar, std::set<IndexVar> previouslyDefined) const;

  /// Returns path from underived to indexvar
  std::vector<IndexVar> derivationPath(IndexVar ancestor, IndexVar indexVar) const;

  /// Recover a variable from its children
  ir::Expr recoverVariable(IndexVar indexVar, std::vector<IndexVar> definedVarOrder, std::map<IndexVar, std::vector<ir::Expr>> underivedBounds, std::map<IndexVar, ir::Expr> childVariables, Iterators iterators) const;

  /// Recover a child from other variables in relationship ex. split inner from parent and outer
  /// emitVarDecl = whether to emit new variables or just assign values to existing variables.
  ir::Stmt recoverChild(IndexVar indexVar, std::map<IndexVar, ir::Expr> relVariables, bool emitVarDecl, Iterators iterators) const;

  /// Retrieves set of all index variables
  std::set<IndexVar> getAllIndexVars() const;

  /// isDivided returns whether or not the target IndexVar was divided through
  /// a `.divide` scheduling operation.
  bool isDivided(IndexVar indexVar) const;

private:
  std::map<IndexVar, IndexVarRel> childRelMap;
  std::map<IndexVar, IndexVarRel> parentRelMap;

  std::map<IndexVar, std::vector<IndexVar>> parentsMap;
  std::map<IndexVar, std::vector<IndexVar>> childrenMap;

  std::set<IndexVar> nodes;
};

}
#endif
