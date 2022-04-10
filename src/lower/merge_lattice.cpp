#include "taco/lower/merge_lattice.h"

#include <set>
#include <vector>
#include <algorithm>

#include "taco/lower/iterator.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "tensor_path.h"
#include "mode_access.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"
#include "taco/index_notation/iteration_algebra.h"
#include "taco/util/scopedmap.h"

using namespace std;

namespace taco {

class MergeLatticeBuilder : public IndexNotationVisitorStrict, public IterationAlgebraVisitorStrict {
public:
  MergeLatticeBuilder(IndexVar i, Iterators iterators, ProvenanceGraph provGraph, std::set<IndexVar> definedIndexVars,
                      std::map<TensorVar, const AccessNode *> whereTempsToResult = {})
                      : i(i), iterators(iterators), provGraph(provGraph), definedIndexVars(definedIndexVars),
                        whereTempsToResult(whereTempsToResult) {}

  MergeLattice build(IndexStmt stmt) {
    stmt.accept(this);
    MergeLattice l = lattice;
    lattice = MergeLattice({});
    return l;
  }

  MergeLattice build(IndexExpr expr) {
    expr.accept(this);
    MergeLattice l = lattice;
    lattice = MergeLattice({});
    return l;
  }

  MergeLattice build(IterationAlgebra alg) {
    alg.accept(this);
    MergeLattice l = lattice;
    lattice = MergeLattice({});
    return l;
  }

  Iterator getIterator(Access access, IndexVar accessVar) {
    // must have matching underived ancestor
    map<IndexVar, int> accessUnderivedAncestorsToLoc;
    int locCounter = 0;
    for (IndexVar indexVar : access.getIndexVars()) {
      vector<IndexVar> underivedVars = provGraph.getUnderivedAncestors(indexVar);
      if (underivedVars.size() != 1) {
        // this is a temporary accessed by scheduled var
        Iterator levelIterator = iterators.levelIterator(ModeAccess(access, 1));
        return levelIterator;
      }
      accessUnderivedAncestorsToLoc[underivedVars[0]] = locCounter++;
    }

    vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(accessVar);
    int loc = -1;
    for (int i = 0; i < (int) underivedAncestors.size(); i++) {
      if (accessUnderivedAncestorsToLoc.count(underivedAncestors[i])) {
        loc = accessUnderivedAncestorsToLoc[underivedAncestors.back()] + 1;
        break;
      }
    }

    taco_iassert(loc != -1);
    Iterator levelIterator = iterators.levelIterator(ModeAccess(access, loc));
    return levelIterator;
  }

private:
  IndexVar i;
  Iterators iterators;
  MergeLattice lattice = MergeLattice({});
  ProvenanceGraph provGraph;
  std::set<IndexVar> definedIndexVars;
  map<TensorVar,MergeLattice> latticesOfTemporaries;
  std::map<TensorVar, const AccessNode *> whereTempsToResult;
  map<Access, MergePoint> seenMergePoints;

  MergeLattice modeIterationLattice() {
    return MergeLattice({MergePoint({iterators.modeIterator(i)}, {}, {})});
  }

  void visit(const RegionNode* node) {
    if(!node->expr().defined()) {
      // Region is empty so return empty lattice
      lattice = MergeLattice({});
      return;
    }

    lattice = build(node->expr());
  }

  void visit(const ComplementNode* node) {
    taco_iassert(isa<Region>(node->a)) << "Demorgan's rule must be applied before lowering.";
    lattice = build(node->a);

    vector<MergePoint> points = flipPoints(lattice.points());

    // Otherwise, all tensors are sparse
    points = includeMissingProducerPoints(points);

    // Add dimension point
    Iterator dimIter = iterators.modeIterator(i);
    points = includeDimensionIterator(points, dimIter);

    bool needsDimPoint = true;
    for(const auto& point: points) {
      if(point.locators().empty() && point.iterators().size() == 1 && point.iterators()[0] == dimIter) {
        needsDimPoint = false;
        break;
      }
    }

    if(needsDimPoint) {
      points.push_back(MergePoint({dimIter}, {}, {}));
    }

    lattice = MergeLattice(points);
  }

  void visit(const IntersectNode* node) {
    MergeLattice a = build(node->a);
    MergeLattice b = build(node->b);

    if (a.points().size() > 0 && b.points().size() > 0) {
      lattice = intersectLattices(a, b);
    } else {
      // If any side of an intersection is empty, the entire intersection must be empty
      lattice = MergeLattice({});
    }
  }

  void visit(const UnionNode* node) {
    MergeLattice a = build(node->a);
    MergeLattice b = build(node->b);
    if (a.points().size() > 0 && b.points().size() > 0) {
      lattice = unionLattices(a, b);
    }
      // Scalar operands
    else if (a.points().size() > 0) {
      lattice = a;
    }
    else if (b.points().size() > 0) {
      lattice = b;
    }
  }

  void visit(const IndexVarNode* varNode) {
    // There are a few cases here...

    // 1) If var in the expression is the same as the var being lowered, we need to return a lattice
    //    with one point that iterates over the universe of the current dimension.
    //    Why: TACO needs to know if it needs to generate merge loops to deal with computing the current index var.
    //      Eg. b(i) + i where b is sparse. To preserve semantics, we need to merge the sparse iteration set of b
    //      with the implied dense space of i.
    //      Question: What if the user WANTS i to be 'sparse'? Just define a func where + is an intersection =)
    // 2) The vars differ. This case actually has 2 subcases...
    //    a) The loop variable ('i' in this builder) is derived from the variable used in the expression
    //       ('var' defined below). In this case, return a mode iterator over the derived var ('i' in the builder)
    //       so taco can generate the correct merge loops for this level.
    //    b) The loop variable is not derived from the variable used in the expression. In this case, we just return
    //        an empty lattice as there is nothing that needs to be merged =)
    // TODO: Add these cases to the test suite....
    IndexVar var(varNode);
    taco_iassert(provGraph.isUnderived(var));
    if (var == i) {
      lattice = MergeLattice({MergePoint({Iterator(var)}, {}, {})});
    } else {
      if (provGraph.isDerivedFrom(i, var)) {
        lattice = MergeLattice({MergePoint({iterators.modeIterator(i)}, {}, {})});
      } else {
        lattice = MergeLattice({});
      }
    }
  }

  void visit(const AccessNode* access)
  {
    // TODO: Case where Access is used in computation but not iteration algebra
    if(seenMergePoints.find(access) != seenMergePoints.end()) {
      lattice = MergeLattice({seenMergePoints.at(access)});
      return;
    }

    if (util::contains(latticesOfTemporaries, access->tensorVar)) {
      // If the accessed tensor variable is a temporary with an associated merge
      // lattice then we return that lattice.
      lattice = latticesOfTemporaries.at(access->tensorVar);
      return;
    }

    vector<IndexVar> underivedAcestors = provGraph.getUnderivedAncestors(i);

    set<IndexVar> accessUnderivedAncestors;
    for (IndexVar indexVar : access->indexVars) {
      vector<IndexVar> underived = provGraph.getUnderivedAncestors(indexVar);
      accessUnderivedAncestors.insert(underived.begin(), underived.end());
    }

    IndexVar accessVar;
    bool foundAccessVar = false;

    // use the outermost fused underived ancestor if multiple appear in access
    for (int i = (int) underivedAcestors.size() - 1; i >= 0; i--) {
      if (util::contains(accessUnderivedAncestors, underivedAcestors[i])) {
        accessVar = underivedAcestors[i];
        foundAccessVar = true;
      }
    }
    if (!foundAccessVar) {
      // The access expression does not index i so we construct a lattice from
      // the mode iterator.  This is sufficient to support broadcast semantics!
      lattice = modeIterationLattice();
      return;
    }

    Iterator iterator = getIterator(access, i);
    taco_iassert(iterator.hasCoordIter() || iterator.hasPosIter() ||
                 iterator.hasLocate())
            << "Iterator must support at least one capability";

    vector<Iterator> pointIterators = {iterator};
    if (provGraph.hasCoordBounds(i)) { // if there are coordiante bounds then add a ranger
      pointIterators.push_back(iterators.modeIterator(i));
    }

    // If the iterator has an index set, then consider that iterator as another
    // iterator that is part of this point.
    if (iterator.hasIndexSet()) {
      pointIterators.push_back(iterator.getIndexSetIterator());
    }

    IndexVar posIteratorDescendant;
    // if this loop is actually iterating over this access then can return iterator (+ coord ranger if applicable)
    // as entire merge point
    if (provGraph.getPosIteratorDescendant(accessVar, &posIteratorDescendant) && posIteratorDescendant == i) {
      MergePoint point = MergePoint(pointIterators, {}, {});
      lattice = MergeLattice({point});
    }
    // If this is a position variable then return an iterator over the variable and locate into the access
    else if (provGraph.isPosVariable(i)) {
      MergePoint point = MergePoint({iterators.modeIterator(i)}, {iterator}, {});
      lattice = MergeLattice({point});
    }
    else {
      // If iterator does not support coordinate or position iteration then
      // iterate over the dimension and locate from it
      MergePoint point = (!iterator.hasCoordIter() && !iterator.hasPosIter())
                         ? MergePoint({iterators.modeIterator(i)}, {iterator}, {})
                         : MergePoint(pointIterators, {}, {});
      lattice = MergeLattice({point});
    }

    seenMergePoints.insert({access, lattice.points()[0]});
  }

  void visit(const LiteralNode* node) {
    // TODO: if constant is zero, then lattice should iterate over no coordinate
    //       (rather than all coordinates)
    lattice = modeIterationLattice();
  }

  void visit(const NegNode* node) {
    lattice = build(node->a);
  }

  void visit(const AddNode* node) {
    lattice = build(new UnionNode(Region(node->a), Region(node->b)));
  }

  void visit(const SubNode* expr) {
    lattice = build(new UnionNode(Region(expr->a), Region(expr->b)));
  }

  void visit(const MulNode* expr) {
    lattice = build(new IntersectNode(Region(expr->a), Region(expr->b)));
  }

  void visit(const DivNode* expr) {
    lattice = build(new IntersectNode(Region(expr->a), Region(expr->b)));
  }

  void visit(const SqrtNode* expr) {
    lattice = build(expr->a);
  }

  void visit(const CastNode* expr) {
    lattice = build(expr->a);
  }

  void visit(const CallNode* expr) {
    taco_iassert(expr->iterAlg.defined()) << "Algebra must be defined" << endl;
    lattice = build(expr->iterAlg);

    // Now we need to store regions that should be kept when applying optimizations.
    // Can't remove regions described by special regions since the lowerer must emit checks for those in
    // all cases.
    const auto regionDefs = expr->regionDefinitions;
    const vector<IndexExpr> inputs = expr->args;
    set<set<Iterator>> regionsToKeep;

    for(auto& it : regionDefs) {
      vector<int> region = it.first;
      set<Iterator> regionToKeep;
      for(auto idx : region) {
        match(inputs[idx],
              function<void(const AccessNode*)>([&](const AccessNode* n) {
                  set<Iterator> tensorRegion = seenMergePoints.at(n).tensorRegion();
                  regionToKeep.insert(tensorRegion.begin(), tensorRegion.end());
              })
        );
      }
      regionsToKeep.insert(regionToKeep);
    }

    lattice = MergeLattice(lattice.points(), regionsToKeep);
  }

  void visit(const CallIntrinsicNode* expr) {
    const auto zeroPreservingArgsSets = 
        expr->func->zeroPreservingArgs(expr->args);

    std::set<size_t> zeroPreservingArgs;
    for (const auto& zeroPreservingArgsSet : zeroPreservingArgsSets) {
      taco_iassert(!zeroPreservingArgsSet.empty());
      for (const auto zeroPreservingArg : zeroPreservingArgsSet) {
        zeroPreservingArgs.insert(zeroPreservingArg);
      }
    }

    MergeLattice l = modeIterationLattice();
    for (size_t i = 0; i < expr->args.size(); ++i) {
      if (!util::contains(zeroPreservingArgs, i)) {
        MergeLattice argLattice = build(expr->args[i]);
        l = unionLattices(l, argLattice);
      }
    }

    for (const auto& zeroPreservingArgsSet : zeroPreservingArgsSets) {
      MergeLattice zeroPreservingLattice({});
      for (const auto zeroPreservingArg : zeroPreservingArgsSet) {
        MergeLattice argLattice = build(expr->args[zeroPreservingArg]);
        zeroPreservingLattice = unionLattices(zeroPreservingLattice, 
                                              argLattice);
      }
      l = intersectLattices(l, zeroPreservingLattice);
    }

    lattice = l;
  }

  void visit(const ReductionNode* node) {
    taco_ierror << "Merge lattices must be created from concrete index "
    << "notation, which does not have reduction nodes.";
  }

  void visit(const AssignmentNode* node) {
    lattice = build(node->rhs);
    latticesOfTemporaries.insert({node->lhs.getTensorVar(), lattice});

    // This is to allow for scalar temporaries to be used (for example
    // to reduce teh number of atomic instructions). In this case, we still
    // want to coiterate the result variable and use those underived index variables
    // (whereas the scalar has no index variables)
    const AccessNode * lhs = (const AccessNode *) node->lhs.ptr;
    if (whereTempsToResult.count(lhs->tensorVar) && lhs->tensorVar.getOrder() == 0) {
      lhs = whereTempsToResult[lhs->tensorVar];
    }
    set<IndexVar> lhsUnderivedAncestors;
    for (IndexVar indexVar : lhs->indexVars) {
      vector<IndexVar> underived = provGraph.getUnderivedAncestors(indexVar);
      lhsUnderivedAncestors.insert(underived.begin(), underived.end());
    }

    // find results for all underived ancestors
    vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(i);
    set<IndexVar> underivedAncestorsSet = set<IndexVar>(underivedAncestors.begin(), underivedAncestors.end());
    set<Iterator> resultIterators;
    for (auto accessVar : underivedAncestorsSet) {
      if (lhsUnderivedAncestors.count(accessVar)) {
        resultIterators.insert(getIterator(lhs, accessVar));
      }
    }

    if (!resultIterators.empty()) {
      vector<MergePoint> points;
      for (auto &point : lattice.points()) {
        points.push_back(MergePoint(point.iterators(), point.locators(),
                                    vector<Iterator>(resultIterators.begin(), resultIterators.end()),
                                    point.isOmitter()));
      }
      lattice = MergeLattice(points, lattice.getTensorRegionsToKeep());
    }
  }

  void visit(const YieldNode* node) {
    lattice = build(node->expr);
  }

  void visit(const ForallNode* node) {
    lattice = build(node->stmt);
  }

  void visit(const WhereNode* node) {
    // Each where produces a temporary that is consumed on the left-hand side.
    // Since where nodes can be nested, it is possible to for multiple
    // temporaries to be consumed by a consumer expression.  The expression that
    // compute temporaries have an iteration space.  The merge lattice of these
    // iteration spaces must be merged with the iteration space of the
    // expression the temporary is combined with.  The merge lattice
    // construction strategy for where nodes is to keep a map of temporaries and
    // their corresponding merge lattices.
    build(node->producer);
    lattice = build(node->consumer);
  }

  void visit(const MultiNode* node) {
    lattice = unionLattices(build(node->stmt1), build(node->stmt2));
  }

  void visit(const SequenceNode* node) {
    taco_not_supported_yet;
  }

  void visit(const AssembleNode* op) {
    taco_not_supported_yet;
  }

  void visit(const SuchThatNode* node) {
    taco_not_supported_yet;
  }

  vector<MergePoint>
  enumerateChildrenPoints(const MergePoint& point, const map<set<Iterator>, MergePoint>& originalPoints,
                          set<set<Iterator>>& seen) {
    set<Iterator> pointIters(point.iterators().begin(), point.iterators().end());
    set<Iterator> pointLocs(point.locators().begin(), point.locators().end());

    set<Iterator> regions = point.tensorRegion();
    set<Iterator> currentRegion = point.tensorRegion();

    vector<MergePoint> result;
    for(const auto& tensorIt: regions) {
      currentRegion.erase(tensorIt);

      if(util::contains(seen, currentRegion)) {
        currentRegion.insert(tensorIt);
        continue;
      }

      if(util::contains(originalPoints, currentRegion)) {
        result.push_back(originalPoints.at(currentRegion));
      }
      else if(!currentRegion.empty()){
        MergePoint mp({}, {}, {});
        for(const auto& it: currentRegion) {
          mp = unionPoints(mp, seenMergePoints.at(iterators.modeAccess(it).getAccess()));
        }

        vector<Iterator> newIters;
        vector<Iterator> newLocators = mp.locators();
        for(const auto& it: mp.iterators()) {
          if(util::contains(pointLocs, it)) {
            newLocators.push_back(it);
          }
          else {
            newIters.push_back(it);
          }
        }

        result.push_back(MergePoint(newIters, newLocators, point.results()));
      }

      seen.insert(currentRegion);
      currentRegion.insert(tensorIt);
    }
    return result;
  }

  vector<MergePoint>
  includeMissingProducerPoints(const vector<MergePoint>& points) {
    if(points.empty()) return points;

    map<set<Iterator>, MergePoint> originalPoints;
    set<set<Iterator>> seen;
    for(const auto& point: points) {
      originalPoints.insert({point.tensorRegion(), point});
    }

    vector<MergePoint> frontier = {points[0]};
    vector<MergePoint> exactLattice;

    while(!frontier.empty()) {
      vector<MergePoint> nextFrontier;
      for (const auto &frontierPoint: frontier) {
        exactLattice.push_back(frontierPoint);
        util::append(nextFrontier, enumerateChildrenPoints(frontierPoint, originalPoints, seen));
      }

      frontier = nextFrontier;
    }

    return exactLattice;
  }

  static vector<MergePoint>
  includeDimensionIterator(const vector<MergePoint>& points, const Iterator& dimIter) {
    vector<MergePoint> results;
    for (auto& point : points) {
      vector<Iterator> iterators = point.iterators();
      if (!any(iterators, [](Iterator it){ return it.isDimensionIterator(); })) {
        taco_iassert(point.iterators().size() > 0);
        results.push_back(MergePoint(combine(iterators, {dimIter}),
                                     point.locators(),
                                     point.results(),
                                     point.isOmitter()));
      }
      else {
        results.push_back(point);
      }
    }
    return results;
  }

  /**
   * The intersection of two lattices is the result of merging all the
   * combinations of merge points from the two lattices.
   */
  static MergeLattice intersectLattices(MergeLattice left, MergeLattice right)
  {
    vector<MergePoint> points;

    // Choose a side to locate from.  We can only choose one side, we make this
    // decision once for all intersected lattice points, and we locate from the
    // right by default.
    bool locateLeft = locateFromLeft(left, right);

    // Append all combinations of a and b merge points
    struct pointSort {
      bool operator()(const MergePoint& a, const MergePoint& b) {
        size_t left_size  = a.iterators().size() + a.locators().size();
        size_t right_size = b.iterators().size() + b.locators().size();
        return left_size > right_size;
      }
    } pointSorter;

    // Append all combinations of the merge points of a and b
    auto sorted_apoint = left.points();
    auto sorted_bpoint = right.points();
    std::sort(sorted_apoint.begin(), sorted_apoint.end(), pointSorter);
    std::sort(sorted_bpoint.begin(), sorted_bpoint.end(), pointSorter);

    set<Iterator> apoint_root_set;
    if (!sorted_apoint.empty())
      apoint_root_set = sorted_apoint.begin()->tensorRegion();

    set<Iterator>bpoint_root_set;
    if (!sorted_bpoint.empty())
      bpoint_root_set = sorted_bpoint.begin()->tensorRegion();


    for (auto& apoint : sorted_apoint) {
      for (auto& bpoint : sorted_bpoint) {
        bool hasIntersection = true;

        auto apoint_set = apoint.tensorRegion();
        auto bpoint_set = bpoint.tensorRegion();

        for (auto& it : apoint_set) {
          if (!std::count(bpoint_set.begin(), bpoint_set.end(), it) &&
              std::count(bpoint_root_set.begin(), bpoint_root_set.end(), it)) {
            hasIntersection = false;
          }
        }
        for (auto& it : bpoint_set) {
          if (!std::count(apoint_set.begin(), apoint_set.end(), it) &&
              std::count(apoint_root_set.begin(), apoint_root_set.end(), it)) {
            hasIntersection = false;
          }
        }
        if (hasIntersection)
          points.push_back(intersectPoints(apoint, bpoint, locateLeft));
      }
    }
    std::sort(points.begin(), points.end(), pointSorter);

    // Correctness: ensures that points produced on BOTH the left and the
    //              right lattices are produced in the final intersection.
    //              Needed since some subPoints may omit leading to erroneous
    //              omit intersection points.
    points = correctPointTypesAfterIntersect(left.points(), right.points(), points);

    // Correctness: Deduplicate regions that are described by multiple lattice
    //              points and resolves conflicts arising between omitters and
    //              producers
     points = removeDuplicatedTensorRegions(points, true);
     
    // Optimization: Removed a subLattice of points if the entire subLattice is
    //               made of only omitters
    // points = removeUnnecessaryOmitterPoints(points);

    set<set<Iterator>> toKeep = left.getTensorRegionsToKeep();
    set<set<Iterator>> toKeepRight = right.getTensorRegionsToKeep();

    toKeep.insert(toKeepRight.begin(), toKeepRight.end());
    return MergeLattice(points, toKeep);
  }

  /**
   * The union of two lattices is an intersection followed by the lattice
   * points of the first lattice followed by the merge points of the second.
   */
  static MergeLattice unionLattices(MergeLattice left, MergeLattice right)
  {
    vector<MergePoint> points;

    struct pointSort {
      bool operator()(const MergePoint& a, const MergePoint& b) {
        size_t left_size  = a.iterators().size() + a.locators().size();
        size_t right_size = b.iterators().size() + b.locators().size();
        return left_size > right_size;
      }
    } pointSorter;

    // Append all combinations of the merge points of a and b
    auto sorted_apoint = left.points();
    auto sorted_bpoint = right.points();
    std::sort(sorted_apoint.begin(), sorted_apoint.end(), pointSorter);
    std::sort(sorted_bpoint.begin(), sorted_bpoint.end(), pointSorter);

    set<Iterator> apoint_root_set;
    if (!sorted_apoint.empty())
      apoint_root_set = sorted_apoint.begin()->tensorRegion();

    set<Iterator>bpoint_root_set;
    if (!sorted_bpoint.empty())
      bpoint_root_set = sorted_bpoint.begin()->tensorRegion();

    for (auto& apoint : sorted_apoint) {
      for (auto& bpoint : sorted_bpoint) {
        bool hasIntersection = true;

        auto apoint_set = apoint.tensorRegion();
        auto bpoint_set = bpoint.tensorRegion();

        for (auto& it : apoint_set) {
          if (!std::count(bpoint_set.begin(), bpoint_set.end(), it) &&
              std::count(bpoint_root_set.begin(), bpoint_root_set.end(), it)) {
            hasIntersection = false;
          }
        }
        for (auto& it : bpoint_set) {
          if (!std::count(apoint_set.begin(), apoint_set.end(), it) &&
              std::count(apoint_root_set.begin(), apoint_root_set.end(), it)) {
            hasIntersection = false;
          }
        }
        if (hasIntersection)
          points.push_back(unionPoints(apoint, bpoint));
      }
    }

    // Append the merge points of a
    util::append(points, left.points());

    // Append the merge points of b
    util::append(points, right.points());

    std::sort(points.begin(), points.end(), pointSorter);

    // Correctness: This ensures that points omitted on BOTH the left and the
    //              right lattices are omitted in the Union. Needed since some
    //              subpoints may produce leading to erroneous producer regions
    points = correctPointTypesAfterUnion(left.points(), right.points(), points);

    // Correctness: Deduplicate regions that are described by multiple lattice
    //              points and resolves conflicts arising between omitters and
    //              producers
    points = removeDuplicatedTensorRegions(points, false);

    // Optimization: insert a dimension iterator if one of the iterators in the
    //               iterate set is not ordered.
    points = insertDimensionIteratorIfNotOrdered(points);

    // Optimization: move iterators to the locate set if they support locate and
    //               are subsets of some other iterator.
    points = moveLocateSubsetIteratorsToLocateSet(points);

    // Optimization: Removes a subLattice of points if the entire subLattice is
    //               made of only omitters
    // points = removeUnnecessaryOmitterPoints(points);
    set<set<Iterator>> toKeep = left.getTensorRegionsToKeep();
    set<set<Iterator>> toKeepRight = right.getTensorRegionsToKeep();

    toKeep.insert(toKeepRight.begin(), toKeepRight.end());
    return MergeLattice(points, toKeep);
  }

  /**
   * Conjunctively merge two merge points a and b into a new point. The steps
   * of the new merge point are a union (concatenation) of the steps of a and
   * b. The expression of the new merge point is expr_a op expr_b, where op is
   * a binary expr type.  If locateLeft is true then we will locate from a,
   *  otherwise we will locate from b.
   */
  static MergePoint intersectPoints(MergePoint left, MergePoint right,
                                    bool locateLeft)
  {
    vector<Iterator> iterators;
    vector<Iterator> locators;

    tie(iterators, locators) = split((locateLeft ? left : right).iterators(),
                                     [](Iterator it){return !it.hasLocate();});
    iterators = filter(iterators, [](Iterator it) {
      return !it.isDimensionIterator();
    });

    iterators = (locateLeft) ? combine(iterators, right.iterators())
                             : combine(left.iterators(), iterators);
    locators = (locateLeft) ? combine(locators, left.locators(),
                                      right.locators())
                            : combine(left.locators(), locators,
                                      right.locators());

    // Remove duplicate iterators.
    iterators = deduplicateDimensionIterators(iterators);

    vector<Iterator> results = combine(left.results(),   right.results());

    return MergePoint(iterators, locators, results, left.isOmitter() || right.isOmitter());
  }

  /**
   * Disjunctively merge two merge points a and b into a new point. The steps
   * of the new merge point are a union (concatenation) of the steps of a and
   * b. The expression of the new merge point is expr_a op expr_b, where op is
   * a binary expr type.
   */
  static MergePoint unionPoints(MergePoint left, MergePoint right)
  {

    vector<Iterator> iterators= combine(left.iterators(),right.iterators());
    vector<Iterator> locaters = combine(left.locators(), right.locators());
    vector<Iterator> results  = combine(left.results(),  right.results());

    // Remove duplicate iterators.
    iterators = deduplicateDimensionIterators(iterators);

    return MergePoint(iterators, locaters, results, left.isOmitter() && right.isOmitter());
  }

  static bool locateFromLeft(MergeLattice left, MergeLattice right)
  {
    // Locate from the side with a dimension iterator
    if (any(right.iterators(),
            [](Iterator it){ return it.isDimensionIterator(); })) {
      return false;
    }
    if (any(left.iterators(),
            [](Iterator it){ return it.isDimensionIterator(); })) {
      return true;
    }

    // Locate from the side with a full+locate iterator
    if (any(right.iterators(),
            [](Iterator it){ return it.isFull() && it.hasLocate(); })) {
      return false;
    }
    if (any(left.iterators(),
            [](Iterator it){ return it.isFull() && it.hasLocate(); })) {
      return true;
    }

    // Locate from the side with more locate iterators
    size_t leftNumLocates  = count(left.iterators(),
                                   [](Iterator it){ return it.hasLocate(); });
    size_t rightNumLocates = count(right.iterators(),
                                   [](Iterator it){ return it.hasLocate(); });
    return (leftNumLocates > rightNumLocates);
  }

  static vector<MergePoint>
  insertDimensionIteratorIfNotOrdered(const vector<MergePoint>& points)
  {
    vector<MergePoint> results;
    for (auto& point : points) {
      vector<Iterator> iterators = point.iterators();
      if (any(iterators, [](Iterator it){ return !it.isOrdered(); }) &&
          !any(iterators, [](Iterator it){ return it.isDimensionIterator(); })) {
        taco_iassert(point.iterators().size() > 0);
        Iterator dimension(iterators[0].getIndexVar());
        results.push_back(MergePoint(combine(iterators, {dimension}),
                                     point.locators(),
                                     point.results(),
                                     point.isOmitter()));
      }
      else {
        results.push_back(point);
      }
    }
    return results;
  }

  static vector<MergePoint>
  moveLocateSubsetIteratorsToLocateSet(const vector<MergePoint>& points)
  {
    vector<Iterator> full = filter(points[0].iterators(),
                                   [](Iterator it){ return it.isFull(); });

    // We only support, for now, optimizing for subsets of full iterators.  If
    // there are no full iterators then we don't do anything.
    if (full.size() == 0) {
      return points;
    }

    // Move locate iterators to the locate set, except the first full iterator.
    Iterator firstFull = full[0];
    vector<MergePoint> result;
    for (auto& point : points) {
      vector<Iterator> locators;
      vector<Iterator> iterators;
      tie(locators, iterators) = split(point.iterators(),
                                       [&firstFull](Iterator it) {
                                         return it.hasLocate() && it != firstFull;
                                       });
      result.push_back(MergePoint(iterators,
                                  combine(point.locators(), locators),
                                  point.results(),
                                  point.isOmitter()));
    }
    return result;
  }

  static vector<Iterator>
  deduplicateDimensionIterators(const vector<Iterator>& iterators)
  {
    vector<Iterator> deduplicates;

    // Remove all but one of the dense iterators, which are all the same.
    bool dimensionIteratorFound = false;
    for (auto& iterator : iterators) {
      if (iterator.isDimensionIterator()) {
        if (!dimensionIteratorFound) {
          deduplicates.push_back(iterator);
          dimensionIteratorFound = true;
        }
      }
      else {
        deduplicates.push_back(iterator);
      }
    }
    return deduplicates;
  }

  static vector<Iterator>
  removeDimensionIterators(const vector<Iterator>& iterators)
  {
    vector<Iterator> result;

    // Remove all but one of the dense iterators, which are all the same.
    for (auto& iterator : iterators) {
      if (!iterator.isDimensionIterator()) {
        result.push_back(iterator);
      }
    }
    return result;
  }


  static vector<MergePoint>
  flipPoints(const vector<MergePoint>& points) {
    vector<MergePoint> flippedPoints;
    for(const auto& mp: points) {
      MergePoint flippedPoint(mp.iterators(), mp.locators(), mp.results(), !mp.isOmitter());
      flippedPoints.push_back(flippedPoint);
    }
    return flippedPoints;
  }

  static set<set<Iterator>>
  getProducerOrOmitterRegions(const std::vector<MergePoint>& points, bool getOmitters) {
    set<set<Iterator>> result;

    for(const auto& point: points) {
      if(point.isOmitter() == getOmitters) {
        set<Iterator> region = point.tensorRegion();
        result.insert(region);
      }
    }
    return result;
  }

  static vector<MergePoint>
  correctPointTypes(const vector<MergePoint>& left, const vector<MergePoint>& right,
                    const vector<MergePoint>& points, bool preserveOmit) {
    vector<MergePoint> result;
    set<set<Iterator>> leftSet = getProducerOrOmitterRegions(left, preserveOmit);
    set<set<Iterator>> rightSet = getProducerOrOmitterRegions(right, preserveOmit);

    for (auto& point : points) {
      set<Iterator> iteratorSet = point.tensorRegion();

      MergePoint newPoint = point;
      if(util::contains(leftSet, iteratorSet) && util::contains(rightSet, iteratorSet)) {
        // Both regions produce/omit, so we ensure that this is preserved
        newPoint = MergePoint(point.iterators(), point.locators(), point.results(), preserveOmit);
      }
      result.push_back(newPoint);
    }

    return result;
  }

  static vector<MergePoint>
  correctPointTypesAfterIntersect(const vector<MergePoint>& left, const vector<MergePoint>& right,
                                  const vector<MergePoint>& points) {
    return correctPointTypes(left, right, points, false);
  }

  static vector<MergePoint>
  correctPointTypesAfterUnion(const vector<MergePoint>& left, const vector<MergePoint>& right,
                              const vector<MergePoint>& points) {
    return correctPointTypes(left, right, points, true);
  }

  static vector<MergePoint>
  removeDuplicatedTensorRegions(const vector<MergePoint>& points, bool preserveOmitters) {

    set<set<Iterator>> producerRegions = getProducerOrOmitterRegions(points, false);
    set<set<Iterator>> omitterRegions = getProducerOrOmitterRegions(points, true);

    vector<MergePoint> result;

    set<set<Iterator>> regionSets;
    for (auto& point : points) {
      set<Iterator> region = point.tensorRegion();

      if(util::contains(regionSets, region)) {
        continue;
      }

      MergePoint p = point;
      if(util::contains(producerRegions, region) && util::contains(omitterRegions, region)) {
        // If a region is marked as both produce and omit resolve the ambiguity based on the preserve
        // omitters flag.
        p = MergePoint(point.iterators(), point.locators(), point.results(), preserveOmitters);
      }

      result.push_back(p);
      regionSets.insert(region);
    }

    return result;
  }

  static vector<MergePoint>
  removeUnnecessaryOmitterPoints(const vector<MergePoint>& points) {
    vector<MergePoint> filteredPoints;

    MergeLattice l(points);
    set<set<Iterator>> removed;

    for(const auto& point : points) {

      if(util::contains(removed, point.tensorRegion())) {
        continue;
      }

      MergeLattice subLattice = l.subLattice(point);

      if(util::all(subLattice.points(), [](const MergePoint p) {return p.isOmitter();})) {
        for(const auto& p : subLattice.points()) {
          removed.insert(p.tensorRegion());
        }
      }
    }

    for(const auto& point : points) {
      if(!util::contains(removed, point.tensorRegion())) {
        filteredPoints.push_back(point);
      }
    }

    return filteredPoints;
  }

};


// class MergeLattice
MergeLattice::MergeLattice(vector<MergePoint> points, set<set<Iterator>> regionsToKeep) : points_(points),
                                                                                          regionsToKeep(regionsToKeep)
{
}

MergeLattice MergeLattice::make(Forall forall, Iterators iterators, ProvenanceGraph provGraph, std::set<IndexVar> definedIndexVars, std::map<TensorVar, const AccessNode *> whereTempsToResult)
{
  // Can emit merge lattice once underived ancestor can be recovered
  IndexVar indexVar = forall.getIndexVar();

  MergeLatticeBuilder builder(indexVar, iterators, provGraph, definedIndexVars, whereTempsToResult);

  vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(indexVar);
  for (auto ancestor : underivedAncestors) {
    if(!provGraph.isRecoverable(ancestor, definedIndexVars)) {
      return MergeLattice({MergePoint({iterators.modeIterator(indexVar)}, {}, {})});
    }
  }

  MergeLattice lattice = builder.build(forall.getStmt());

  // Can't remove points if lattice contains omitters since we lose merge cases during lowering.
  if(lattice.anyModeIteratorIsLeaf() && lattice.needExplicitZeroChecks()) {
    return lattice;
  }

  // Loop lattice and case lattice are identical so simplify here
  return lattice.getLoopLattice();
}

std::vector<MergePoint>
MergeLattice::removePointsThatLackFullIterators(const std::vector<MergePoint>& points)
{
  vector<MergePoint> result;
  vector<Iterator> fullIterators = filter(points[0].iterators(),
                                          [](Iterator it){return it.isFull();});
  for (auto& point : points) {
    bool missingFullIterator = false;
    for (auto& fullIterator : fullIterators) {
      if (!util::contains(point.iterators(), fullIterator)) {
        missingFullIterator = true;
        break;
      }
    }
    if (!missingFullIterator) {
      result.push_back(point);
    }
  }
  return result;
}

std::vector<MergePoint>
MergeLattice::removePointsWithIdenticalIterators(const std::vector<MergePoint>& points)
{
  vector<MergePoint> result;
  set<set<Iterator>> producerIteratorSets;
  for (auto& point : points) {
    set<Iterator> iteratorSet(point.iterators().begin(),
                              point.iterators().end());
    if (util::contains(producerIteratorSets, iteratorSet)) {
      continue;
    }
    result.push_back(point);
    producerIteratorSets.insert(iteratorSet);
  }
  return result;
}

bool MergeLattice::needExplicitZeroChecks() {
  if(util::any(points(), [](const MergePoint& mp) {return mp.isOmitter();})) {
    return true;
  }
  return !getTensorRegionsToKeep().empty();
}

MergeLattice MergeLattice::subLattice(MergePoint lp) const {
  // A merge point lp dominats lq iff it contains a subset of lp's
  // tensor path steps. So we scan through the points and filter those points.
  vector<MergePoint> dominatedPoints;
  vector<Iterator> lpIterators = lp.iterators();
  sort(lpIterators.begin(), lpIterators.end());
  for (auto& lq : this->points()) {
    vector<Iterator> lqIterators = lq.iterators();
    sort(lqIterators.begin(), lqIterators.end());
    if (includes(lpIterators.begin(), lpIterators.end(),
                      lqIterators.begin(), lqIterators.end())) {
      dominatedPoints.push_back(lq);
    }
  }
  return MergeLattice(dominatedPoints);
}

const vector<MergePoint>& MergeLattice::points() const {
  return points_;
}

const vector<Iterator>& MergeLattice::iterators() const {
  // The iterators merged by a lattice are those merged by the first point
  taco_iassert(points().size() > 0) << "No merge points in the merge lattice";
  return points()[0].iterators();
}

const vector<Iterator>& MergeLattice::locators() const {
  // The iterators merged by a lattice are those merged by the first point
  taco_iassert(points().size() > 0) << "No merge points in the merge lattice";
  return points()[0].locators();
}

set<Iterator> MergeLattice::exhausted(MergePoint point) {
  set<Iterator> notExhaustedIters = point.tensorRegion();

  set<Iterator> exhausted;
  vector<Iterator> modeIterators = combine(iterators(), locators());
  modeIterators = filter(modeIterators, [](const Iterator& it) {return it.isModeIterator();});

  for (auto& iterator : modeIterators) {
    if (!util::contains(notExhaustedIters, iterator)) {
      exhausted.insert(iterator);
    }
  }

  return exhausted;
}

const std::vector<Iterator>& MergeLattice::results() const {
  taco_iassert(points().size() > 0) << "No merge points in the merge lattice";
  return points()[0].results();
}

bool MergeLattice::exact() const {
  // A lattice is full if any merge point iterates over only full iterators
  // or if each sparse iterator is uniquely iterated by some lattice point.
  set<Iterator> uniquelyMergedIterators;

  if (util::any(points(), [](const MergePoint& m) {return m.isOmitter();})) {
    return false;
  }

  for (auto& point : this->points()) {
    if (all(point.iterators(), [](Iterator it) {return it.isFull();})) {
      return true;
    }
  }

  for (auto& point : this->points()) {
    if (point.iterators().size() == 1) {
      uniquelyMergedIterators.insert(point.iterators()[0]);
    }
  }

  for (auto& it : iterators()) {
    if (!util::contains(uniquelyMergedIterators, it)) {
      return false;
    }
  }
  return true;
}

bool MergeLattice::anyModeIteratorIsLeaf() const {
  if(points().empty()) {
    return false;
  }
  vector<Iterator> latticeIters = util::combine(iterators(), locators());
  return util::any(latticeIters, [](const Iterator& it) {return it.isModeIterator() && it.isLeaf();});
}

std::vector<Iterator> MergeLattice::retrieveRegionIteratorsToOmit(const MergePoint &point) const {
  vector<Iterator> omittedIterators;
  set<Iterator> pointRegion = point.tensorRegion();
  set<Iterator> seen;
  const size_t levelOfPoint = pointRegion.size();

  if(point.isOmitter()) {
    seen = set<Iterator>(pointRegion.begin(), pointRegion.end());
    omittedIterators = vector<Iterator>(seen.begin(), seen.end());
  }

  // Look at all points above
  for(const auto& mp: points()) {
    if((mp.tensorRegion().size() > levelOfPoint) && mp.isOmitter()) {
      // Grab the omitted tensors
      set<Iterator> parentRegion = mp.tensorRegion();
      std::vector<Iterator> parentItersToOmit;
      set_difference(parentRegion.begin(), parentRegion.end(),
                     pointRegion.begin(), pointRegion.end(),
                     back_inserter(parentItersToOmit));

      // Add iterators not in present point to the iterators to omit
      for(const auto& it : parentItersToOmit) {
        if(!util::contains(seen, it)) {
          seen.insert(it);
          omittedIterators.push_back(it);
        }
      }
    }
  }

  return omittedIterators;
}

set<set<Iterator>> MergeLattice::getTensorRegionsToKeep() const {
  return regionsToKeep;
}

MergeLattice MergeLattice::getLoopLattice() const {
  std::vector<MergePoint> p = removePointsThatLackFullIterators(points());
  return removePointsWithIdenticalIterators(p);
}

ostream& operator<<(ostream& os, const MergeLattice& ml) {
  return os << util::join(ml.points(), ", ");
}

bool operator==(const MergeLattice& a, const MergeLattice& b) {
  auto apoints = a.points();
  auto bpoints = b.points();
  struct pointSort {
    bool operator()(const MergePoint& a, const MergePoint& b) {
      size_t left_size  = a.iterators().size() + a.locators().size();
      size_t right_size = b.iterators().size() + b.locators().size();
      return left_size > right_size;
    }
  } pointSorter;

  std::sort(apoints.begin(), apoints.end(), pointSorter);
  std::sort(bpoints.begin(), bpoints.end(), pointSorter);
  if (apoints.size() != bpoints.size()) {
    return false;
  }
  for (size_t i = 0; i < apoints.size(); i++) {
    if (apoints[i] != bpoints[i]) {
      return false;
    }
  }
  return true;
}

bool operator!=(const MergeLattice& a, const MergeLattice& b) {
  return !(a == b);
}


// class MergePoint
struct MergePoint::Content {
  std::vector<Iterator> iterators;
  std::vector<Iterator> locators;
  std::vector<Iterator> results;
  bool omitPoint;
};

MergePoint::MergePoint(const vector<Iterator>& iterators,
                       const vector<Iterator>& locators,
                       const vector<Iterator>& results,
                       bool omitPoint) : content_(new Content) {
  taco_uassert(iterators.size() <= 1 ||
               all(iterators,
                   [](Iterator it){ return it.hasLocate() || it.isOrdered(); }))
      << "Merge points do not support iterators that do not have locate and "
      << "that are not ordered.";

  content_->iterators = util::removeDuplicates(iterators);
  content_->locators = util::removeDuplicates(locators);
  content_->results = util::removeDuplicates(results);
  content_->omitPoint = omitPoint;
}

const vector<Iterator>& MergePoint::iterators() const {
  return content_->iterators;
}

std::vector<Iterator> MergePoint::rangers() const {
  // We can remove an iterator from the rangers iff it is guaranteed to be
  // exhausted after the other rangers (the rangers are the iterators we iterate
  // over until one is exhausted).  This holds if the largest coordinate of this
  // iterator is smaller than the largest coordinate of the other iterators.
  // We will, conservatively, say this condition holds if the iterator is full
  // and there exist another iterator that is not full, since this iterator is
  // then a superset of that iterator.  We will start with all iterators and
  // only add those for which this condition does not hold to the rangers.
  if (any(iterators(), [](Iterator iterator){return !iterator.isFull();})) {
    vector<Iterator> rangers;
    for (auto& iterator : iterators()) {
      if (!iterator.isFull()) {
        rangers.push_back(iterator);
      }
    }
    return rangers;
  }
  return iterators();
}

std::vector<Iterator> MergePoint::mergers() const {
  // We can remove an iterator from the mergers iff it is a subset of the other
  // mergers (the mergers ar the iterators that specify the points we visit
  // within the range specified by the rangers).  We will, conservatively, say
  // that this condition holds if one of the other iterators is full, since this
  // iterator is then a subset of it.  We will start with all iterators and only
  // add those for which this condition does not hold to the mergers.
  if (any(iterators(), [](Iterator iterator){return iterator.isFull();})) {
    vector<Iterator> mergers;
    for (auto& iterator : iterators()) {
      if (iterator.isFull()) {
        mergers.push_back(iterator);
      }
    }
    return mergers;
  }

  // explicitly remove dimension iterators that are not full if there is at most one other iterator
  size_t numNotFull = count(iterators(), [](Iterator iterator){return !iterator.isFull() && iterator.isDimensionIterator();});
  if (numNotFull != iterators().size() && numNotFull > 0) {
    vector<Iterator> mergers;
    for (auto& iterator : iterators()) {
      if (!iterator.isDimensionIterator()) {
        mergers.push_back(iterator);
      }
    }
    return mergers;
  }

  return iterators();
}

const std::vector<Iterator>& MergePoint::locators() const {
  return content_->locators;
}

const std::vector<Iterator>& MergePoint::results() const {
  return content_->results;
}

const std::set<Iterator> MergePoint::tensorRegion() const {
  std::vector<Iterator> iterators = filter(content_->iterators,
                                    [](Iterator it) {return !it.isDimensionIterator();});

  append(iterators, content_->locators);
  return set<Iterator>(iterators.begin(), iterators.end());
}

bool MergePoint::isOmitter() const {
  return content_->omitPoint;
}

ostream& operator<<(ostream& os, const MergePoint& mlp) {
  os << "[";
  os << util::join(mlp.iterators(), ", ");
  if (mlp.iterators().size() > 0) os << " ";
  os << "|";
  os << " ";
  os << util::join(mlp.locators(),  ", ");
  if (mlp.locators().size() > 0) os << " ";
  os << "|";
  if (mlp.results().size() > 0) os << " ";
  os << util::join(mlp.results(),   ", ");

  os << "|";
  if(mlp.isOmitter()) {
    os << " O ";
  } else {
    os << " P ";
  }

  os << "]";
  return os;
}

static bool compare(const vector<Iterator>& a, const vector<Iterator>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (auto iterators : util::zip(a,b)) {
    if (iterators.first != iterators.second) {
      return false;
    }
  }
  return true;
}

bool operator==(const MergePoint& a, const MergePoint& b) {
  if (!compare(a.iterators(), b.iterators())) return false;
  if (!compare(a.locators(), b.locators())) return false;
  if (!compare(a.results(), b.results())) return false;
  if ((a.isOmitter() != b.isOmitter())) return false;
  return true;
}

bool operator!=(const MergePoint& a, const MergePoint& b) {
  return !(a == b);
}


// Free functions
std::vector<Iterator> deduplicate(const std::vector<Iterator>& iterators) {
  vector<Iterator> deduplicates;

  // Remove all but one of the dense iterators, which are all the same.
  bool added = false;
  for (auto& iterator : iterators) {
    if (iterator.isFull() && iterator.isOrdered()) {
      if (!added) {
        deduplicates.push_back(iterator);
        added = true;
      }
    }
    else {
      deduplicates.push_back(iterator);
    }
  }
  return deduplicates;
}

vector<Iterator> simplify(const vector<Iterator>& iterators) {
  vector<Iterator> simplifiedIterators;
  vector<Iterator> fullIterators;

  for (const auto& iter : iterators) {
    if (!iter.isFull()) {
      simplifiedIterators.push_back(iter);
    } else if (fullIterators.empty()) {
      // must iterate over at least one of the full dimensions
      fullIterators.push_back(iter);
    } else if (!iter.hasLocate()) {
      // preferably iterate over only full dimensions that do not support locate
      if (fullIterators[0].hasLocate()) {
        fullIterators.clear();
      }
      fullIterators.push_back(iter);
    }
  }
  util::append(simplifiedIterators, fullIterators);

  return simplifiedIterators;
}

}
