#include "taco/index_notation/transformations.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/error/error_messages.h"
#include "taco/util/collections.h"
#include "taco/lower/iterator.h"
#include "taco/lower/merge_lattice.h"

#include <iostream>
#include <algorithm>
#include <limits>

using namespace std;

namespace taco {

// class Transformation
Transformation::Transformation(Reorder reorder)
    : transformation(new Reorder(reorder)) {
}

Transformation::Transformation(Precompute precompute)
    : transformation(new Precompute(precompute)) {
}

Transformation::Transformation(ForAllReplace forallreplace)
        : transformation(new ForAllReplace(forallreplace)) {
}

Transformation::Transformation(Parallelize parallelize)
        : transformation(new Parallelize(parallelize)) {
}

Transformation::Transformation(AddSuchThatPredicates addsuchthatpredicates)
        : transformation(new AddSuchThatPredicates(addsuchthatpredicates)) {
}

IndexStmt Transformation::apply(IndexStmt stmt, string* reason) const {
  return transformation->apply(stmt, reason);
}

std::ostream& operator<<(std::ostream& os, const Transformation& t) {
  t.transformation->print(os);
  return os;
}


// class Reorder
struct Reorder::Content {
  std::vector<IndexVar> replacePattern;
  bool pattern_ordered; // In case of Reorder(i, j) need to change replacePattern ordering to actually reorder
};

Reorder::Reorder(IndexVar i, IndexVar j) : content(new Content) {
  content->replacePattern = {i, j};
  content->pattern_ordered = false;
}

Reorder::Reorder(std::vector<taco::IndexVar> replacePattern) : content(new Content) {
  content->replacePattern = replacePattern;
  content->pattern_ordered = true;
}

IndexVar Reorder::geti() const {
  return content->replacePattern[0];
}

IndexVar Reorder::getj() const {
  if (content->replacePattern.size() == 1) {
    return geti();
  }
  return content->replacePattern[1];
}

const std::vector<IndexVar>& Reorder::getreplacepattern() const {
  return content->replacePattern;
}

IndexStmt Reorder::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  string r;
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation: " + r;
    return IndexStmt();
  }

  // collect current ordering of IndexVars
  bool startedMatch = false;
  std::vector<IndexVar> currentOrdering;
  bool matchFailed = false;

  match(stmt,
        std::function<void(const ForallNode*)>([&](const ForallNode* op) {
          bool matches = std::find (getreplacepattern().begin(), getreplacepattern().end(), op->indexVar) != getreplacepattern().end();
          if (matches) {
            currentOrdering.push_back(op->indexVar);
            startedMatch = true;
          }
          else if (startedMatch && currentOrdering.size() != getreplacepattern().size()) {
            matchFailed = true;
          }
        })
  );

  if (!content->pattern_ordered && currentOrdering == getreplacepattern()) {
    taco_iassert(getreplacepattern().size() == 2);
    content->replacePattern = {getreplacepattern()[1], getreplacepattern()[0]};
  }

  if (matchFailed || currentOrdering.size() != getreplacepattern().size()) {
    *reason = "The foralls of reorder pattern: " + util::join(getreplacepattern()) + " were not directly nested.";
    return IndexStmt();
  }
  return ForAllReplace(currentOrdering, getreplacepattern()).apply(stmt, reason);
}

void Reorder::print(std::ostream& os) const {
  os << "reorder(" << util::join(getreplacepattern()) << ")";
}

std::ostream& operator<<(std::ostream& os, const Reorder& reorder) {
  reorder.print(os);
  return os;
}


// class Precompute
struct Precompute::Content {
  IndexExpr expr;
  IndexVar i;
  IndexVar iw;
  TensorVar workspace;
};

Precompute::Precompute() : content(nullptr) {
}

Precompute::Precompute(IndexExpr expr, IndexVar i, IndexVar iw,
                     TensorVar workspace) : content(new Content) {
  content->expr = expr;
  content->i = i;
  content->iw = iw;
  content->workspace = workspace;
}

IndexExpr Precompute::getExpr() const {
  return content->expr;
}

IndexVar Precompute::geti() const {
  return content->i;
}

IndexVar Precompute::getiw() const {
  return content->iw;
}

TensorVar Precompute::getWorkspace() const {
  return content->workspace;
}

static bool containsExpr(Assignment assignment, IndexExpr expr) {
   struct ContainsVisitor : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    IndexExpr expr;
    bool contains = false;

    void visit(const UnaryExprNode* node) {
      if (equals(IndexExpr(node), expr)) {
        contains = true;
      }
      else {
        IndexNotationVisitor::visit(node);
      }
    }

    void visit(const BinaryExprNode* node) {
      if (equals(IndexExpr(node), expr)) {
        contains = true;
      }
      else {
        IndexNotationVisitor::visit(node);
      }
    }

    void visit(const ReductionNode* node) {
      taco_ierror << "Reduction node in concrete index notation.";
    }
  };

  ContainsVisitor visitor;
  visitor.expr = expr;
  visitor.visit(assignment);
  return visitor.contains;
}

static Assignment getAssignmentContainingExpr(IndexStmt stmt, IndexExpr expr) {
  Assignment assignment;
  match(stmt,
        function<void(const AssignmentNode*,Matcher*)>([&assignment, &expr](
            const AssignmentNode* node, Matcher* ctx) {
          if (containsExpr(node, expr)) {
            assignment = node;
          }
        })
  );
  return assignment;
}

IndexStmt Precompute::apply(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);

  // Precondition: The expr to precompute is not in `stmt`
  Assignment assignment = getAssignmentContainingExpr(stmt, getExpr());
  if (!assignment.defined()) {
    *reason = "The expression (" + util::toString(getExpr()) + ") " +
              "is not in " + util::toString(stmt);
    return IndexStmt();
  }

  struct PrecomputeRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    Precompute precompute;

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = precompute.geti();

      if (foralli.getIndexVar() == i) {
        IndexStmt s = foralli.getStmt();
        TensorVar ws = precompute.getWorkspace();
        IndexExpr e = precompute.getExpr();
        IndexVar iw = precompute.getiw();

        IndexStmt consumer = forall(i, replace(s, {{e, ws(i)}}));
        IndexStmt producer = forall(iw, ws(iw) = replace(e, {{i,iw}}));
        Where where(consumer, producer);

        stmt = where;
        return;
      }
      IndexNotationRewriter::visit(node);
    }

  };
  PrecomputeRewriter rewriter;
  rewriter.precompute = *this;
  return rewriter.rewrite(stmt);
}

void Precompute::print(std::ostream& os) const {
  os << "precompute(" << getExpr() << ", " << geti() << ", "
     << getiw() << ", " << getWorkspace() << ")";
}

bool Precompute::defined() const {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Precompute& precompute) {
  precompute.print(os);
  return os;
}

// class ForAllReplace
struct ForAllReplace::Content {
  std::vector<IndexVar> pattern;
  std::vector<IndexVar> replacement;
};

ForAllReplace::ForAllReplace() : content(nullptr) {
}

ForAllReplace::ForAllReplace(std::vector<IndexVar> pattern, std::vector<IndexVar> replacement) : content(new Content) {
  taco_iassert(!pattern.empty());
  content->pattern = pattern;
  content->replacement = replacement;
}

std::vector<IndexVar> ForAllReplace::getPattern() const {
  return content->pattern;
}

std::vector<IndexVar> ForAllReplace::getReplacement() const {
  return content->replacement;
}

IndexStmt ForAllReplace::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  string r;
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation: " + r;
    return IndexStmt();
  }

  /// Since all IndexVars can only appear once, assume replacement will work and error if it doesn't
  struct ForAllReplaceRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    ForAllReplace transformation;
    string* reason;
    int elementsMatched = 0;
    ForAllReplaceRewriter(ForAllReplace transformation, string* reason)
            : transformation(transformation), reason(reason) {}

    IndexStmt forallreplace(IndexStmt stmt) {
      IndexStmt replaced = rewrite(stmt);

      // Precondition: Did not find pattern
      if (replaced == stmt || elementsMatched == -1) {
        *reason = "The pattern of ForAlls: " +
                  util::join(transformation.getPattern()) +
                  " was not found while attempting to replace with: " +
                  util::join(transformation.getReplacement());
        return IndexStmt();
      }
      return replaced;
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      vector<IndexVar> pattern = transformation.getPattern();
      if (elementsMatched == -1) {
        return; // pattern did not match
      }

      if(elementsMatched >= (int) pattern.size()) {
        IndexNotationRewriter::visit(node);
        return;
      }

      if (foralli.getIndexVar() == pattern[elementsMatched]) {
        if (elementsMatched + 1 < (int) pattern.size() && !isa<Forall>(foralli.getStmt())) {
          // child is not a forallnode (not directly nested)
          elementsMatched = -1;
          return;
        }
        // assume rest of pattern matches
        vector<IndexVar> replacement = transformation.getReplacement();
        bool firstMatch = (elementsMatched == 0);
        elementsMatched++;
        stmt = rewrite(foralli.getStmt());
        if (firstMatch) {
          // add replacement nodes and cut out this node
          for (auto i = replacement.rbegin(); i != replacement.rend(); ++i ) {
            stmt = forall(*i, stmt);
          }
        }
        // else cut out this node
        return;
      }
      else if (elementsMatched > 0) {
        elementsMatched = -1; // pattern did not match
        return;
      }
      // before pattern match
      IndexNotationRewriter::visit(node);
    }
  };
  return ForAllReplaceRewriter(*this, reason).forallreplace(stmt);
}

void ForAllReplace::print(std::ostream& os) const {
  os << "forallreplace(" << util::join(getPattern()) << ", " << util::join(getReplacement()) << ")";
}

std::ostream& operator<<(std::ostream& os, const ForAllReplace& forallreplace) {
  forallreplace.print(os);
  return os;
}

// class AddSuchThatRels
struct AddSuchThatPredicates::Content {
  std::vector<IndexVarRel> predicates;
};

AddSuchThatPredicates::AddSuchThatPredicates() : content(nullptr) {
}

AddSuchThatPredicates::AddSuchThatPredicates(std::vector<IndexVarRel> predicates) : content(new Content) {
  taco_iassert(!predicates.empty());
  content->predicates = predicates;
}

std::vector<IndexVarRel> AddSuchThatPredicates::getPredicates() const {
  return content->predicates;
}

IndexStmt AddSuchThatPredicates::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  string r;
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation: " + r;
    return IndexStmt();
  }

  if (isa<SuchThat>(stmt)) {
    SuchThat suchThat = to<SuchThat>(stmt);
    vector<IndexVarRel> predicate = suchThat.getPredicate();
    vector<IndexVarRel> predicates = getPredicates();
    predicate.insert(predicate.end(), predicates.begin(), predicates.end());
    return SuchThat(suchThat.getStmt(), predicate);
  }
  else{
    return SuchThat(stmt, content->predicates);
  }
}

void AddSuchThatPredicates::print(std::ostream& os) const {
  os << "addsuchthatpredicates(" << util::join(getPredicates()) << ")";
}

std::ostream& operator<<(std::ostream& os, const AddSuchThatPredicates& addSuchThatPredicates) {
  addSuchThatPredicates.print(os);
  return os;
}

struct ReplaceReductionExpr : public IndexNotationRewriter {
  const std::map<Access,Access>& substitutions;
  ReplaceReductionExpr(const std::map<Access,Access>& substitutions)
          : substitutions(substitutions) {}
  using IndexNotationRewriter::visit;
  void visit(const AssignmentNode* node) {
    if (util::contains(substitutions, node->lhs)) {
      stmt = Assignment(substitutions.at(node->lhs), rewrite(node->rhs), node->op);
    }
    else {
      IndexNotationRewriter::visit(node);
    }
  }
};


// int tj = 0; ... tj += rhs; ... lhs = rhs (reduces atomics)
struct IntroduceScalarTemp : public IndexNotationRewriter {
  using IndexNotationRewriter::visit;
  ProvenanceGraph provGraph;

  set<const AssignmentNode*> handledAssignments;
  set<const AssignmentNode*> assignmentsIndexedByNestedLoop;

  IndexStmt introduceScalarTemp(IndexStmt stmt, ProvenanceGraph provGraph) {
    this->provGraph = provGraph;
    return rewrite(stmt);
  }

  void visit(const WhereNode *op) {
    IndexStmt producer = op->producer; // don't apply transformation to producers
    IndexStmt consumer = rewrite(op->consumer);

    if (producer == op->producer && consumer == op->consumer) {
      stmt = op;
    }
    else {
      stmt = new WhereNode(consumer, producer);
    }
  }

  void visit(const ForallNode *node) {
    Forall foralli(node);
    IndexVar i = foralli.getIndexVar();

    vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(i);
    vector<const AssignmentNode *> reducedAssignments;
    match(foralli.getStmt(),
          function<void(const AssignmentNode*)>([&](const AssignmentNode* node) {
            vector<IndexVar> reductionVars = Assignment(node).getReductionVars();
            for (auto underived : underivedAncestors) {
              bool reducedByI = find(reductionVars.begin(), reductionVars.end(), underived) != reductionVars.end();
              if (reducedByI) {
                reducedAssignments.push_back(node);
                break;
              }
            }
            bool reducedByI = find(reductionVars.begin(), reductionVars.end(), i) != reductionVars.end();
            if (reducedByI) { // can be indexed by non-underived if temporary
              reducedAssignments.push_back(node);
            }
          })
    );
    if (reducedAssignments.size() > 0) {
      IndexStmt transformed_stmt = forall(i, rewrite(foralli.getStmt()), foralli.getParallelUnit(),
                                          foralli.getOutputRaceStrategy(), foralli.getUnrollFactor());
      for (auto assignment : reducedAssignments) {
        if (handledAssignments.count(assignment) || assignmentsIndexedByNestedLoop.count(assignment)) {
          continue;
        }
        handledAssignments.insert(assignment); // TODO: apply at higher levels  than just bottom-most loop
        TensorVar t(string("t") + foralli.getIndexVar().getName(), Type(assignment->lhs.getDataType()));
        IndexStmt producer = ReplaceReductionExpr(map<Access, Access>({{assignment->lhs, t}})).rewrite(
                transformed_stmt);
        taco_iassert(isa<Forall>(producer));
        IndexStmt consumer = Assignment(assignment->lhs, t, assignment->op);
        transformed_stmt = where(consumer, producer);
      }
      stmt = transformed_stmt;
    }
    else {
      IndexNotationRewriter::visit(node);
    }
    match(foralli.getStmt(),
          function<void(const AssignmentNode*)>([&](const AssignmentNode* node) {
            vector<IndexVar> freeVars = Assignment(node).getFreeVars();
            for (auto underived : underivedAncestors) {
              bool indexedByI = find(freeVars.begin(), freeVars.end(), underived) != freeVars.end();
              if (indexedByI) {
                assignmentsIndexedByNestedLoop.insert(node);
                break;
              }
            }
            bool indexedByI = find(freeVars.begin(), freeVars.end(), i) != freeVars.end();
            if (indexedByI) {
              assignmentsIndexedByNestedLoop.insert(node);
            }
          })
    );
  }
};

// class Parallelize
struct Parallelize::Content {
  IndexVar i;
  ParallelUnit  parallel_unit;
  OutputRaceStrategy output_race_strategy;
};


Parallelize::Parallelize() : content(nullptr) {
}

Parallelize::Parallelize(IndexVar i) : Parallelize(i, ParallelUnit::DefaultUnit, OutputRaceStrategy::NoRaces) {}

Parallelize::Parallelize(IndexVar i, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy) : content(new Content) {
  content->i = i;
  content->parallel_unit = parallel_unit;
  content->output_race_strategy = output_race_strategy;
}


IndexVar Parallelize::geti() const {
  return content->i;
}

ParallelUnit Parallelize::getParallelUnit() const {
  return content->parallel_unit;
}

OutputRaceStrategy Parallelize::getOutputRaceStrategy() const {
  return content->output_race_strategy;
}

IndexStmt Parallelize::apply(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);

  struct ParallelizeRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    Parallelize parallelize;
    ProvenanceGraph provGraph;
    set<IndexVar> definedIndexVars;
    set<ParallelUnit> parentParallelUnits;
    std::string reason = "";

    IndexStmt rewriteParallel(IndexStmt stmt) {
      provGraph = ProvenanceGraph(stmt);
      return rewrite(stmt);
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = parallelize.geti();

      Iterators iterators(foralli);
      definedIndexVars.insert(foralli.getIndexVar());
      MergeLattice lattice = MergeLattice::make(foralli, iterators, provGraph, definedIndexVars);
      // Precondition 3: No parallelization of variables under a reduction
      // variable (ie MergePoint has at least 1 result iterators)
      if (parallelize.getOutputRaceStrategy() == OutputRaceStrategy::NoRaces && lattice.results().empty()
          && lattice != MergeLattice({MergePoint({iterators.modeIterator(foralli.getIndexVar())}, {}, {})})) {
        reason = "Precondition failed: Free variables cannot be dominated by reduction variables in the iteration graph, "
                 "as this causes scatter behavior and we do not yet emit parallel synchronization constructs";
        return;
      }

      if (foralli.getIndexVar() == i) {
        // Precondition 1: No coiteration of node (ie Merge Lattice has only 1 iterator)
        if (lattice.iterators().size() != 1) {
          reason = "Precondition failed: The loop must not merge tensor dimensions, that is, it must be a for loop;";
          return;
        }

        // Precondition 2: Every result iterator must have insert capability
        for (Iterator iterator : lattice.results()) {
          while (true) {
            if (!iterator.hasInsert()) {
              reason = "Precondition failed: The output tensor must allow inserts";
              return;
            }
            if (iterator.isLeaf()) {
              break;
            }
            iterator = iterator.getChild();
          }
        }

        vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(i);
        IndexVar underivedAncestor = underivedAncestors.back();

        // get lattice that corresponds to underived ancestor. This is bottom-most loop that shares underived ancestor
        Forall underivedForall = foralli;
        match(foralli.getStmt(),
              function<void(const ForallNode*)>([&](const ForallNode* node) {
                vector<IndexVar> nodeUnderivedAncestors = provGraph.getUnderivedAncestors(node->indexVar);
                if (underivedAncestor == nodeUnderivedAncestors.back()) {
                  underivedForall = Forall(node);
                }
              })
        );
        MergeLattice underivedLattice = MergeLattice::make(underivedForall, iterators, provGraph, definedIndexVars);


        if(underivedLattice.results().empty() && parallelize.getOutputRaceStrategy() == OutputRaceStrategy::Temporary) {
          // Need to precompute reduction

          // Find all occurrences of reduction in expression
          vector<const AssignmentNode *> precomputeAssignments;
          match(foralli.getStmt(),
                function<void(const AssignmentNode*)>([&](const AssignmentNode* node) {
                  for (auto underivedVar : underivedAncestors) {
                    vector<IndexVar> reductionVars = Assignment(node).getReductionVars();
                    bool reducedByI =
                            find(reductionVars.begin(), reductionVars.end(), underivedVar) != reductionVars.end();
                    if (reducedByI) {
                      precomputeAssignments.push_back(node);
                      break;
                    }
                  }
                })
          );
          taco_iassert(!precomputeAssignments.empty());

          IndexStmt precomputed_stmt = forall(i, foralli.getStmt(), parallelize.getParallelUnit(), parallelize.getOutputRaceStrategy(), foralli.getUnrollFactor());
          for (auto assignment : precomputeAssignments) {
            // Construct temporary of correct type and size of outer loop
            TensorVar w(string("w_") + ParallelUnit_NAMES[(int) parallelize.getParallelUnit()], Type(assignment->lhs.getDataType(), {Dimension(i)}), taco::dense);

            // rewrite producer to write to temporary, mark producer as parallel
            IndexStmt producer = ReplaceReductionExpr(map<Access, Access>({{assignment->lhs, w(i)}})).rewrite(precomputed_stmt);
            taco_iassert(isa<Forall>(producer));
            Forall producer_forall = to<Forall>(producer);
            producer = forall(producer_forall.getIndexVar(), producer_forall.getStmt(), parallelize.getParallelUnit(), parallelize.getOutputRaceStrategy(), foralli.getUnrollFactor());

            // build consumer that writes from temporary to output, mark consumer as parallel reduction
            ParallelUnit reductionUnit = ParallelUnit::CPUThreadGroupReduction;
            if (should_use_CUDA_codegen()) {
              if (parentParallelUnits.count(ParallelUnit::GPUWarp)) {
                reductionUnit = ParallelUnit::GPUWarpReduction;
              }
              else {
                reductionUnit = ParallelUnit::GPUBlockReduction;
              }
            }
            IndexStmt consumer = forall(i, Assignment(assignment->lhs, w(i), assignment->op), reductionUnit, OutputRaceStrategy::ParallelReduction);
            precomputed_stmt = where(consumer, producer);
          }
          stmt = precomputed_stmt;
          return;
        }

        if (parallelize.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
          // want to avoid extra atomics by accumulating variable and then reducing at end
          stmt = forall(i, IntroduceScalarTemp().introduceScalarTemp(foralli.getStmt(), provGraph), parallelize.getParallelUnit(), parallelize.getOutputRaceStrategy(), foralli.getUnrollFactor());
          return;
        }


        stmt = forall(i, foralli.getStmt(), parallelize.getParallelUnit(), parallelize.getOutputRaceStrategy(), foralli.getUnrollFactor());
        return;
      }

      if (foralli.getParallelUnit() != ParallelUnit::NotParallel) {
        parentParallelUnits.insert(foralli.getParallelUnit());
      }
      IndexNotationRewriter::visit(node);
    }

  };

  ParallelizeRewriter rewriter;
  rewriter.parallelize = *this;
  IndexStmt rewritten = rewriter.rewriteParallel(stmt);
  if (!rewriter.reason.empty()) {
    *reason = rewriter.reason;
    return IndexStmt();
  }
  return rewritten;
}


void Parallelize::print(std::ostream& os) const {
  os << "parallelize(" << geti() << ")";
}


std::ostream& operator<<(std::ostream& os, const Parallelize& parallelize) {
  parallelize.print(os);
  return os;
}


// Autoscheduling functions

IndexStmt parallelizeOuterLoop(IndexStmt stmt) {
  // get outer ForAll
  Forall forall;
  bool matched = false;
  match(stmt,
        function<void(const ForallNode*,Matcher*)>([&forall, &matched](
                const ForallNode* node, Matcher* ctx) {
          if (!matched) forall = node;
          matched = true;
        })
  );

  if (!matched) return stmt;
  string reason;

  if (should_use_CUDA_codegen()) {
    IndexVar i1, i2;
    IndexStmt parallelized256 = stmt.split(forall.getIndexVar(), i1, i2, 256);
    parallelized256 = Parallelize(i1, ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces).apply(parallelized256, &reason);
    if (parallelized256 == IndexStmt()) {
      return stmt;
    }
    parallelized256 = Parallelize(i2, ParallelUnit::GPUThread, OutputRaceStrategy::NoRaces).apply(parallelized256, &reason);
    if (parallelized256 == IndexStmt()) {
      return stmt;
    }
    return parallelized256;
  }
  else {
    IndexStmt parallelized = Parallelize(forall.getIndexVar(), ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces).apply(stmt, &reason);
    if (parallelized == IndexStmt()) {
      // can't parallelize
      return stmt;
    }
    return parallelized;
  }
}

// Takes in a set of pairs of IndexVar and level for a given tensor and orders
// the IndexVars by tensor level
static vector<pair<IndexVar, bool>> 
varOrderFromTensorLevels(set<pair<IndexVar, pair<int, bool>>> tensorLevelVars) {
  vector<pair<IndexVar, pair<int, bool>>> sortedPairs(tensorLevelVars.begin(), 
                                                      tensorLevelVars.end());
  auto comparator = [](const pair<IndexVar, pair<int, bool>> &left, 
                       const pair<IndexVar, pair<int, bool>> &right) {
    return left.second.first < right.second.first;
  };
  std::sort(sortedPairs.begin(), sortedPairs.end(), comparator);

  vector<pair<IndexVar, bool>> varOrder;
  std::transform(sortedPairs.begin(),
                sortedPairs.end(),
                std::back_inserter(varOrder),
                [](const std::pair<IndexVar, pair<int, bool>>& p) {
                  return pair<IndexVar, bool>(p.first, p.second.second);
                });
  return varOrder;
}


// Takes in varOrders from many tensors and creates a map of dependencies between IndexVars
static map<IndexVar, set<IndexVar>>
depsFromVarOrders(map<string, vector<pair<IndexVar,bool>>> varOrders) {
  map<IndexVar, set<IndexVar>> deps;
  for (const auto& varOrderPair : varOrders) {
    const auto& varOrder = varOrderPair.second;
    for (auto firstit = varOrder.begin(); firstit != varOrder.end(); ++firstit) {
      for (auto secondit = firstit + 1; secondit != varOrder.end(); ++secondit) {
        if (firstit->second || secondit->second) { // one of the dimensions must enforce constraints
          if (deps.count(secondit->first)) {
            deps[secondit->first].insert(firstit->first);
          } else {
            deps[secondit->first] = {firstit->first};
          }
        }
      }
    }
  }
  return deps;
}


static vector<IndexVar>
topologicallySort(map<IndexVar,set<IndexVar>> hardDeps,
                  map<IndexVar,multiset<IndexVar>> softDeps,
                  vector<IndexVar> originalOrder) {
  vector<IndexVar> sortedVars;
  unsigned long countVars = originalOrder.size();
  while (sortedVars.size() < countVars) {
    // Scan for variable with no dependencies
    IndexVar freeVar;
    size_t freeVarPos = std::numeric_limits<size_t>::max();
    size_t minSoftDepsRemaining = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < originalOrder.size(); ++i) {
      IndexVar var = originalOrder[i];
      if (!hardDeps.count(var) || hardDeps[var].empty()) {  
        const size_t softDepsRemaining = softDeps.count(var) ? 
                                         softDeps[var].size() : 0;
        if (softDepsRemaining < minSoftDepsRemaining) {
          freeVar = var;
          freeVarPos = i;
          minSoftDepsRemaining = softDepsRemaining;
        }
      }
    }

    // No free var found there is a cycle
    taco_iassert(freeVarPos != std::numeric_limits<size_t>::max())
        << "Cycles in iteration graphs must be resolved, through transpose, "
        << "before the expression is passed to the topological sorting "
        << "routine.";

    sortedVars.push_back(freeVar);

    // remove dependencies on variable
    for (auto& varTensorDepsPair : hardDeps) {
      auto& varTensorDeps = varTensorDepsPair.second;
      if (varTensorDeps.count(freeVar)) {
        varTensorDeps.erase(freeVar);
      }
    }
    for (auto& varTensorDepsPair : softDeps) {
      auto& varTensorDeps = varTensorDepsPair.second;
      if (varTensorDeps.count(freeVar)) {
        varTensorDeps.erase(freeVar);
      }
    }
    originalOrder.erase(originalOrder.begin() + freeVarPos);
  }
  return sortedVars;
}


IndexStmt reorderLoopsTopologically(IndexStmt stmt) {
  // Collect tensorLevelVars which stores the pairs of IndexVar and tensor
  // level that each tensor is accessed at
  struct DAGBuilder : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    // int is level, bool is if level enforces constraints (ie not dense)
    map<string, set<pair<IndexVar, pair<int, bool>>>> tensorLevelVars;
    IndexStmt innerBody;
    map <IndexVar, ParallelUnit> forallParallelUnit;
    map <IndexVar, OutputRaceStrategy> forallOutputRaceStrategy;
    vector<IndexVar> indexVarOriginalOrder;
    Iterators iterators;

    DAGBuilder(Iterators iterators) : iterators(iterators) {};

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();

      MergeLattice lattice = MergeLattice::make(foralli, iterators, ProvenanceGraph(), {}); // TODO
      indexVarOriginalOrder.push_back(i);
      forallParallelUnit[i] = foralli.getParallelUnit();
      forallOutputRaceStrategy[i] = foralli.getOutputRaceStrategy();

      // Iterator and if Iterator enforces constraints
      vector<pair<Iterator, bool>> depIterators;
      for (Iterator iterator : lattice.points()[0].iterators()) {
        if (!iterator.isDimensionIterator()) {
          depIterators.push_back({iterator, true});
        }
      }

      for (Iterator iterator : lattice.points()[0].locators()) {
        depIterators.push_back({iterator, false});
      }

      // add result iterators that append
      for (Iterator iterator : lattice.results()) {
        depIterators.push_back({iterator, !iterator.hasInsert()});
      }

      for (const auto& iteratorPair : depIterators) {
        int level = iteratorPair.first.getMode().getLevel();
        string tensor = to<ir::Var>(iteratorPair.first.getTensor())->name;
        if (tensorLevelVars.count(tensor)) {
          tensorLevelVars[tensor].insert({{i, {level, iteratorPair.second}}});
        }
        else {
          tensorLevelVars[tensor] = {{{i, {level, iteratorPair.second}}}};
        }
      }

      if (!isa<Forall>(foralli.getStmt())) {
        innerBody = foralli.getStmt();
        return; // Only reorder first contiguous section of ForAlls
      }
      IndexNotationVisitor::visit(node);
    }
  };

  Iterators iterators(stmt);
  DAGBuilder dagBuilder(iterators);
  stmt.accept(&dagBuilder);

  // Construct tensor dependencies (sorted list of IndexVars) from tensorLevelVars
  map<string, vector<pair<IndexVar, bool>>> tensorVarOrders;
  for (const auto& tensorLevelVar : dagBuilder.tensorLevelVars) {
    tensorVarOrders[tensorLevelVar.first] = 
        varOrderFromTensorLevels(tensorLevelVar.second);
  }
  const auto hardDeps = depsFromVarOrders(tensorVarOrders);

  struct CollectSoftDependencies : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    map<IndexVar, multiset<IndexVar>> softDeps;

    void visit(const AssignmentNode* op) {
      op->lhs.accept(this);
      op->rhs.accept(this);
    }

    void visit(const AccessNode* node) {
      const auto& modeOrdering = node->tensorVar.getFormat().getModeOrdering();
      for (size_t i = 1; i < (size_t)node->tensorVar.getOrder(); ++i) {
        const auto srcVar = node->indexVars[modeOrdering[i - 1]];
        const auto dstVar = node->indexVars[modeOrdering[i]];
        softDeps[dstVar].insert(srcVar);
      }
    }
  };
  CollectSoftDependencies collectSoftDeps;
  stmt.accept(&collectSoftDeps);

  const auto sortedVars = topologicallySort(hardDeps, collectSoftDeps.softDeps, 
                                            dagBuilder.indexVarOriginalOrder);

  // Reorder Foralls use a rewriter in case new nodes introduced outside of Forall
  struct TopoReorderRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const vector<IndexVar>& sortedVars;
    IndexStmt innerBody;
    const map <IndexVar, ParallelUnit> forallParallelUnit;
    const map <IndexVar, OutputRaceStrategy> forallOutputRaceStrategy;

    TopoReorderRewriter(const vector<IndexVar>& sortedVars, IndexStmt innerBody,
                        const map <IndexVar, ParallelUnit> forallParallelUnit,
                        const map <IndexVar, OutputRaceStrategy> forallOutputRaceStrategy)
        : sortedVars(sortedVars), innerBody(innerBody),
        forallParallelUnit(forallParallelUnit), forallOutputRaceStrategy(forallOutputRaceStrategy)  {
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();

      // first forall must be in collected variables
      taco_iassert(util::contains(sortedVars, i));
      stmt = innerBody;
      for (auto it = sortedVars.rbegin(); it != sortedVars.rend(); ++it) {
        stmt = forall(*it, stmt, forallParallelUnit.at(*it), forallOutputRaceStrategy.at(*it), foralli.getUnrollFactor());
      }
      return;
    }

  };
  TopoReorderRewriter rewriter(sortedVars, dagBuilder.innerBody, 
                               dagBuilder.forallParallelUnit, dagBuilder.forallOutputRaceStrategy);
  return rewriter.rewrite(stmt);
}

IndexStmt scalarPromote(IndexStmt stmt) {
  std::vector<Access> resultAccesses;
  std::tie(resultAccesses, std::ignore) = getResultAccesses(stmt);

  std::map<Access,IndexVar> hoistLevel;
  std::map<Access,IndexExpr> reduceOp;
  struct FindHoistLevel : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    const std::vector<Access>& resultAccesses;
    std::map<Access,IndexVar>& hoistLevel;
    std::map<Access,IndexExpr>& reduceOp;
    std::map<Access,std::set<IndexVar>> hoistIndices;
    std::set<IndexVar> indices;
    
    FindHoistLevel(const std::vector<Access>& resultAccesses,
                   std::map<Access,IndexVar>& hoistLevel,
                   std::map<Access,IndexExpr>& reduceOp) : 
        resultAccesses(resultAccesses), 
        hoistLevel(hoistLevel), 
        reduceOp(reduceOp) {}

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();

      indices.insert(i);
      for (const auto& resultAccess : resultAccesses) {
        std::set<IndexVar> resultIndices(resultAccess.getIndexVars().begin(),
                                         resultAccess.getIndexVars().end());
        if (std::includes(indices.begin(), indices.end(), 
                          resultIndices.begin(), resultIndices.end()) &&
            !util::contains(hoistLevel, resultAccess)) {
          hoistLevel[resultAccess] = i;
          hoistIndices[resultAccess] = indices;
          if (resultIndices != indices) {
            reduceOp[resultAccess] = IndexExpr();
          }
        }
      }
      IndexNotationVisitor::visit(node);
      indices.erase(i);
    }

    void visit(const AssignmentNode* op) {
      if (util::contains(hoistLevel, op->lhs) && 
          hoistIndices[op->lhs] == indices) {
        hoistLevel.erase(op->lhs);
      }
      if (util::contains(reduceOp, op->lhs)) {
        reduceOp[op->lhs] = op->op;
      }
    }
  };
  FindHoistLevel findHoistLevel(resultAccesses, hoistLevel, reduceOp);
  stmt.accept(&findHoistLevel);
  
  struct HoistWrites : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const std::map<Access,IndexVar>& hoistLevel;
    const std::map<Access,IndexExpr>& reduceOp;

    HoistWrites(const std::map<Access,IndexVar>& hoistLevel,
                const std::map<Access,IndexExpr>& reduceOp) : 
        hoistLevel(hoistLevel), reduceOp(reduceOp) {}

    void visit(const ForallNode* node) {
      IndexNotationRewriter::visit(node);

      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();
      IndexStmt body = foralli.getStmt();

      for (const auto& resultAccess : hoistLevel) {
        if (resultAccess.second == i) {
          // This assumes the index expression yields at most one result tensor; 
          // will not work correctly if there are multiple results.
          TensorVar resultVar = resultAccess.first.getTensorVar();
          TensorVar val(resultVar.getName(), 
                        Type(resultVar.getType().getDataType(), {}));
          IndexExpr op = util::contains(reduceOp, resultAccess.first) 
                       ? reduceOp.at(resultAccess.first) : IndexExpr();
          IndexStmt consumer = Assignment(Access(resultAccess.first), val(), op);
          IndexStmt producer = ReplaceReductionExpr(
              map<Access,Access>({{resultAccess.first, val()}})).rewrite(body);
          stmt = forall(i, where(consumer, producer), foralli.getParallelUnit(),
                        foralli.getOutputRaceStrategy(),
                        foralli.getUnrollFactor());
        }
      }
    }
  };
  HoistWrites hoistWrites(hoistLevel, reduceOp);
  return hoistWrites.rewrite(stmt);
}

static bool compare(std::vector<IndexVar> vars1, std::vector<IndexVar> vars2) {
  return vars1 == vars2;
}

// TODO Temporary function to insert workspaces into SpMM kernels
static IndexStmt optimizeSpMM(IndexStmt stmt) {
  if (!isa<Forall>(stmt)) {
    return stmt;
  }
  Forall foralli = to<Forall>(stmt);
  IndexVar i = foralli.getIndexVar();

  if (!isa<Forall>(foralli.getStmt())) {
    return stmt;
  }
  Forall forallk = to<Forall>(foralli.getStmt());
  IndexVar k = forallk.getIndexVar();

  if (!isa<Forall>(forallk.getStmt())) {
    return stmt;
  }
  Forall forallj = to<Forall>(forallk.getStmt());
  IndexVar j = forallj.getIndexVar();

  if (!isa<Assignment>(forallj.getStmt())) {
    return stmt;
  }
  Assignment assignment = to<Assignment>(forallj.getStmt());

  if (!isa<Mul>(assignment.getRhs())) {
    return stmt;
  }
  Mul mul = to<Mul>(assignment.getRhs());

  taco_iassert(isa<Access>(assignment.getLhs()));
  if (!isa<Access>(mul.getA())) {
    return stmt;
  }
  if (!isa<Access>(mul.getB())) {
    return stmt;
  }

  Access Aaccess = to<Access>(assignment.getLhs());
  Access Baccess = to<Access>(mul.getA());
  Access Caccess = to<Access>(mul.getB());

  if (Aaccess.getIndexVars().size() != 2 ||
      Baccess.getIndexVars().size() != 2 ||
      Caccess.getIndexVars().size() != 2) {
    return stmt;
  }

  if (!compare(Aaccess.getIndexVars(), {i,j}) ||
      !compare(Baccess.getIndexVars(), {i,k}) ||
      !compare(Caccess.getIndexVars(), {k,j})) {
    return stmt;
  }

  TensorVar A = Aaccess.getTensorVar();
  if (A.getFormat().getModeFormats()[0].getName() != "dense" ||
      A.getFormat().getModeFormats()[1].getName() != "compressed" ||
      A.getFormat().getModeOrdering()[0] != 0 ||
      A.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  TensorVar B = Baccess.getTensorVar();
  if (B.getFormat().getModeFormats()[0].getName() != "dense" ||
      B.getFormat().getModeFormats()[1].getName() != "compressed" ||
      B.getFormat().getModeOrdering()[0] != 0 ||
      B.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  TensorVar C = Caccess.getTensorVar();
  if (C.getFormat().getModeFormats()[0].getName() != "dense" ||
      C.getFormat().getModeFormats()[1].getName() != "compressed" ||
      C.getFormat().getModeOrdering()[0] != 0 ||
      C.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  // It's an SpMM statement so return an optimized SpMM statement
  TensorVar w("w",
              Type(Float64, {A.getType().getShape().getDimension(1)}),
              taco::dense);
  return forall(i,
                where(forall(j,
                             A(i,j) = w(j)),
                      forall(k,
                             forall(j,
                                    w(j) += B(i,k) * C(k,j)))));
}

IndexStmt insertTemporaries(IndexStmt stmt)
{
  IndexStmt spmm = optimizeSpMM(stmt);
  if (spmm != stmt) {
    return spmm;
  }

  // TODO Implement general workspacing when scattering into sparse result modes

  // Result dimensions that are indexed by free variables dominated by a
  // reduction variable are scattered into.  If any of these are compressed
  // then we introduce a dense workspace to scatter into instead.  The where
  // statement must push the reduction loop into the producer side, leaving
  // only the free variable loops on the consumer side.

  //vector<IndexVar> reductionVars = getReductionVars(stmt);
  //...

  return stmt;
}

}
