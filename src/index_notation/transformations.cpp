#include "taco/index_notation/transformations.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/error/error_messages.h"
#include "taco/util/collections.h"
#include "taco/lower/iterator.h"
#include "taco/lower/merge_lattice.h"
#include "taco/lower/mode.h"
#include "taco/lower/mode_format_impl.h"

#include <iostream>
#include <algorithm>
#include <limits>
#include <set>
#include <map>
#include <vector>

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

Transformation::Transformation(SetMergeStrategy setmergestrategy)
        : transformation(new SetMergeStrategy(setmergestrategy)) {
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

struct SetMergeStrategy::Content {
  IndexVar i_var;
  MergeStrategy strategy;
};

SetMergeStrategy::SetMergeStrategy(IndexVar i, MergeStrategy strategy) : content(new Content) {
  content->i_var = i;
  content->strategy = strategy;
}

IndexVar SetMergeStrategy::geti() const {
  return content->i_var;
}

MergeStrategy SetMergeStrategy::getMergeStrategy() const {
  return content->strategy;
}

IndexStmt SetMergeStrategy::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  string r;
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation: " + r;
    return IndexStmt();
  }

  struct SetMergeStrategyRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    ProvenanceGraph provGraph;
    map<TensorVar,ir::Expr> tensorVars;
    set<IndexVar> definedIndexVars;

    SetMergeStrategy transformation;
    string reason;
    SetMergeStrategyRewriter(SetMergeStrategy transformation)
            : transformation(transformation) {}

    IndexStmt setmergestrategy(IndexStmt stmt) {
      provGraph = ProvenanceGraph(stmt);
      tensorVars = createIRTensorVars(stmt);
      return rewrite(stmt);
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = transformation.geti();
      
      definedIndexVars.insert(foralli.getIndexVar());

      if (foralli.getIndexVar() == i) {
        Iterators iterators(foralli, tensorVars);
        MergeLattice lattice = MergeLattice::make(foralli, iterators, provGraph, 
                                                  definedIndexVars);
        for (auto iterator : lattice.iterators()) {
          if (!iterator.isOrdered()) {
            reason = "Precondition failed: Variable " 
            + i.getName() +
            " is not ordered and cannot be galloped.";
            return;
          }
        }
        if (lattice.points().size() != 1) {
          reason = "Precondition failed: The merge lattice of variable " 
                + i.getName() +
                " has more than 1 point and cannot be merged by galloping";
          return;
        }

        MergeStrategy strategy = transformation.getMergeStrategy();
        stmt = rewrite(foralli.getStmt());
        stmt = Forall(node->indexVar, stmt, strategy, node->parallel_unit, 
                      node->output_race_strategy, node->unrollFactor);
        return;
      }
      IndexNotationRewriter::visit(node);
    }
  };
  SetMergeStrategyRewriter rewriter = SetMergeStrategyRewriter(*this);
  IndexStmt rewritten = rewriter.setmergestrategy(stmt);
  if (!rewriter.reason.empty()) {
    *reason = rewriter.reason;
    return IndexStmt();
  }
  return rewritten;
}

void SetMergeStrategy::print(std::ostream& os) const {
  os << "mergeby(" << geti() << ", " 
     << MergeStrategy_NAMES[(int)getMergeStrategy()] << ")";
}

std::ostream& operator<<(std::ostream& os, const SetMergeStrategy& setmergestrategy) {
  setmergestrategy.print(os);
  return os;
}

// class Precompute
struct Precompute::Content {
  IndexExpr expr;
  std::vector<IndexVar> i_vars;
  std::vector<IndexVar> iw_vars;
  TensorVar workspace;
};

Precompute::Precompute() : content(nullptr) {
}

Precompute::Precompute(IndexExpr expr, IndexVar i, IndexVar iw,
                     TensorVar workspace) : content(new Content) {
  std::vector<IndexVar> i_vars{i};
  std::vector<IndexVar> iw_vars{iw};
  content->expr = expr;
  content->i_vars = i_vars;
  content->iw_vars = iw_vars;
  content->workspace = workspace;
}

  Precompute::Precompute(IndexExpr expr, std::vector<IndexVar> i_vars,
                         std::vector<IndexVar> iw_vars,
                         TensorVar workspace) : content(new Content) {
  content->expr = expr;
  content->i_vars = i_vars;
  content->iw_vars = iw_vars;
  content->workspace = workspace;
}
  
IndexExpr Precompute::getExpr() const {
  return content->expr;
}

std::vector<IndexVar>& Precompute::getIVars() const {
  return content->i_vars;
}

std::vector<IndexVar>& Precompute::getIWVars() const {
  return content->iw_vars;
}

TensorVar Precompute::getWorkspace() const {
  return content->workspace;
}

static bool containsExpr(Assignment assignment, IndexExpr expr) {
   struct ContainsVisitor : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    IndexExpr expr;
    bool contains = false;

    void visit(const AccessNode* node) {
      if (equals(IndexExpr(node), expr)) {
        contains = true;
      }
    }

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

static IndexStmt eliminateRedundantReductions(IndexStmt stmt, 
    const std::set<TensorVar>* const candidates = nullptr) {

  struct ReduceToAssign : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const std::set<TensorVar>* const candidates;
    std::map<TensorVar,std::set<IndexVar>> availableVars;

    ReduceToAssign(const std::set<TensorVar>* const candidates) :
        candidates(candidates) {}

    IndexStmt rewrite(IndexStmt stmt) {
      for (const auto& result : getResults(stmt)) {
        availableVars[result] = {};
      }
      return IndexNotationRewriter::rewrite(stmt);
    }

    void visit(const ForallNode* op) {
      for (auto& it : availableVars) {
        it.second.insert(op->indexVar);
      }
      IndexNotationRewriter::visit(op);
      for (auto& it : availableVars) {
        it.second.erase(op->indexVar);
      }
    }

    void visit(const WhereNode* op) {
      const auto workspaces = getResults(op->producer);
      for (const auto& workspace : workspaces) {
        availableVars[workspace] = {};
      }
      IndexNotationRewriter::visit(op);
      for (const auto& workspace : workspaces) {
        availableVars.erase(workspace);
      }
    }
    
    void visit(const AssignmentNode* op) {
      const auto result = op->lhs.getTensorVar();
      if (op->op.defined() && 
          util::toSet(op->lhs.getIndexVars()) == availableVars[result] &&
          (!candidates || util::contains(*candidates, result))) {
        stmt = Assignment(op->lhs, op->rhs);
        return;
      }
      stmt = op;
    }
  };
  return ReduceToAssign(candidates).rewrite(stmt);
}

IndexStmt Precompute::apply(IndexStmt stmt, std::string* reason) const {
    INIT_REASON(reason);

    // Precondition: The expr to precompute is in `stmt`
    Assignment assignment = getAssignmentContainingExpr(stmt, getExpr());
    if (!assignment.defined()) {
        *reason = "The expression (" + util::toString(getExpr()) + ") " +
                  "is not in " + util::toString(stmt);
        return IndexStmt();
    }
    vector<IndexVar> forallIndexVars;
    match(stmt,
          function<void(const ForallNode*)>([&](const ForallNode* op) {
              forallIndexVars.push_back(op->indexVar);
          })
    );

    ProvenanceGraph provGraph = ProvenanceGraph(stmt);



    struct PrecomputeRewriter : public IndexNotationRewriter {
        using IndexNotationRewriter::visit;
        Precompute precompute;
        ProvenanceGraph provGraph;
        vector<IndexVar> forallIndexVarList;

        Assignment getConsumerAssignment(IndexStmt stmt, TensorVar& ws) {
            Assignment a = Assignment();
            match(stmt,
                  function<void(const AssignmentNode*, Matcher*)>([&](const AssignmentNode* op, Matcher* ctx) {
                      a = Assignment(op);
                  }),
                  function<void(const WhereNode*, Matcher*)>([&](const WhereNode* op, Matcher* ctx) {
                      ctx->match(op->consumer);
                      ctx->match(op->producer);
                  }),
                  function<void(const AccessNode*, Matcher*)>([&](const AccessNode* op, Matcher* ctx) {
                      if (op->tensorVar == ws) {
                          return;
                      }
                  })
            );

            IndexSetRel rel = a.getIndexSetRel();
            /// The reduceOp depends on the relation between indexVar sets of rhs and lhs. For rcl and inter, reduceOp
            /// must be +=. For lcr, reduceOp must be =. For none and equal, reduceOp can't be decided at this stage.
            switch (rel) {
                case none: a = Assignment(a.getLhs(), a.getRhs());break;
                case rcl:  a = Assignment(a.getLhs(), a.getRhs(), Add());break;
                case lcr: a = Assignment(a.getLhs(), a.getRhs());break;
                case inter: a = Assignment(a.getLhs(), a.getRhs(), Add());break;
                case equal: a = Assignment(a.getLhs(), a.getRhs());break;
            }
            return a;
        }

        Assignment getProducerAssignment(TensorVar& ws,
                                         const std::vector<IndexVar>& i_vars,
                                         const std::vector<IndexVar>& iw_vars,
                                         const IndexExpr& e,
                                         map<IndexVar, IndexVar> substitutions) {

            auto a = ws(iw_vars) = replace(e, substitutions);
            IndexSetRel rel = a.getIndexSetRel();
            /// The reduceOp depends on the relation between indexVar sets of rhs and lhs. For rcl and inter, reduceOp
            /// must be +=. For lcr, reduceOp must be =. For none and equal, reduceOp can't be decided at this stage.
            switch (rel) {
                case none: a = Assignment(a.getLhs(), a.getRhs());break;
                case rcl:  a = Assignment(a.getLhs(), a.getRhs(), Add());break;
                case lcr: a = Assignment(a.getLhs(), a.getRhs());break;
                case inter: a = Assignment(a.getLhs(), a.getRhs(), Add());break;
                case equal: a = Assignment(a.getLhs(), a.getRhs());break;
            }
            return a;
        }

        IndexStmt generateForalls(IndexStmt stmt, vector<IndexVar> indexVars) {
            auto returnStmt = stmt;
            for (auto &i : indexVars) {
                returnStmt = forall(i, returnStmt);
            }

            return returnStmt;
        }

        bool containsIndexVarScheduled(vector<IndexVar> indexVars,
                                       IndexVar indexVar) {
            bool contains = false;
            for (auto &i : indexVars) {
                if (i == indexVar) {
                    contains = true;
                } else if (provGraph.isFullyDerived(indexVar) && !provGraph.isFullyDerived(i)) {
                    for (auto &child : provGraph.getFullyDerivedDescendants(i)) {
                        if (child == indexVar)
                            contains = true;
                    }
                }
            }
            return contains;
        }

        void visit(const ForallNode* node) {
            Forall foralli(node);
            std::vector<IndexVar> i_vars = precompute.getIVars();

            bool containsWhere = false;
            match(foralli,
                  function<void(const WhereNode*)>([&](const WhereNode* op) {
                      containsWhere = true;
                  })
            );

            if (!containsWhere) {
                vector<IndexVar> forallIndexVars;
                match(foralli,
                      function<void(const ForallNode*)>([&](const ForallNode* op) {
                          forallIndexVars.push_back(op->indexVar);
                      })
                );

                IndexStmt s = foralli.getStmt();
                TensorVar ws = precompute.getWorkspace();
                IndexExpr e = precompute.getExpr();
                std::vector<IndexVar> iw_vars = precompute.getIWVars();

                map<IndexVar, IndexVar> substitutions;
                taco_iassert(i_vars.size() == iw_vars.size()) << "i_vars and iw_vars lists must be the same size";

                for (int index = 0; index < (int)i_vars.size(); index++) {
                    substitutions[i_vars[index]] = iw_vars[index];
                }

                // Build consumer by replacing with temporary (in replacedStmt)
                IndexStmt replacedStmt = replace(s, {{e, ws(i_vars) }});
                if (replacedStmt != s) {
                    // Then modify the replacedStmt to have the correct foralls
                    // by concretizing the consumer assignment

                    auto consumerAssignment = getConsumerAssignment(replacedStmt, ws);
                    auto consumerIndexVars = consumerAssignment.getIndexVars();

                    auto producerAssignment = getProducerAssignment(ws, i_vars, iw_vars, e, substitutions);
                    auto producerIndexVars = producerAssignment.getIndexVars();

                    vector<IndexVar> producerForallIndexVars;
                    vector<IndexVar> consumerForallIndexVars;
                    vector<IndexVar> outerForallIndexVars;

                    bool stopForallDistribution = false;
                    for (auto &i : util::reverse(forallIndexVars)) {
                        if (!stopForallDistribution && containsIndexVarScheduled(i_vars, i)) {
                            producerForallIndexVars.push_back(substitutions[i]);
                            consumerForallIndexVars.push_back(i);
                        } else {
                            auto consumerContains = containsIndexVarScheduled(consumerIndexVars, i);
                            auto producerContains = containsIndexVarScheduled(producerIndexVars, i);
                            if (stopForallDistribution || (producerContains && consumerContains)) {
                                outerForallIndexVars.push_back(i);
                                stopForallDistribution = true;
                            } else if (!stopForallDistribution && consumerContains) {
                                consumerForallIndexVars.push_back(i);
                            } else if (!stopForallDistribution && producerContains) {
                                producerForallIndexVars.push_back(i);
                            }
                        }
                    }
                    IndexStmt consumer = generateForalls(consumerAssignment, consumerForallIndexVars);

                    IndexStmt producer = generateForalls(producerAssignment, producerForallIndexVars);
                    Where where(consumer, producer);

                    stmt = generateForalls(where, outerForallIndexVars);

                    return;
                }
            }
            IndexNotationRewriter::visit(node);
        }
    };

    /// RedundantVisitor uses Forall Context to determine reduceOp for none and equal.
    /// We assume += is used if a workspace is accessed multiple times, otherwise =.
    /// Forall Context describes the related indexVars of the given indexVar at a specific stage. `ctx_stack` implements such concept.
    struct RedundantVisitor: public IndexNotationVisitor {
        using IndexNotationVisitor::visit;

        std::vector<Assignment>& to_change;
        std::vector<IndexVar> ctx_stack;
        std::vector<int> num_stack;
        int ctx_num;
        const ProvenanceGraph& provGraph;

        RedundantVisitor(std::vector<Assignment>& to_change, const ProvenanceGraph& provGraph):to_change(to_change), ctx_num(0), provGraph(provGraph){}

        void visit(const ForallNode* node) {
            Forall foralli(node);
            IndexVar var = foralli.getIndexVar();
            ctx_stack.push_back(var);
            if (! num_stack.empty()) {
                num_stack.back()++;
            }
            if (num_stack.empty()) {
                num_stack.push_back(1);
            }
            IndexNotationVisitor::visit(node);
        }
        void visit(const WhereNode* node) {
            num_stack.push_back(0);
            IndexNotationVisitor::visit(node->consumer);
            ctx_num = num_stack.back();
            for (int i = 0; i < ctx_num; i++){
                ctx_stack.pop_back();
            }
            num_stack.pop_back();
            num_stack.push_back(0);
            IndexNotationVisitor::visit(node->producer);
            ctx_num = num_stack.back();
            for (int i = 0; i < ctx_num; i++){
                ctx_stack.pop_back();
            }
            num_stack.pop_back();
        }
        void visit(const AssignmentNode* node) {
            Assignment a(node->lhs, node->rhs, node->op);
            vector<IndexVar> freeVars = a.getLhs().getIndexVars();
            set<IndexVar> seen(freeVars.begin(), freeVars.end());

            /// For equal, if some indexVar in lhs has sibling in ctx stack, reduceOp will be +=.
            bool is_equal = (a.getIndexSetRel() == equal);
            bool has_sibling = false;
            match(a.getRhs(),
                  std::function<void(const AccessNode*)>([&](const AccessNode* op) {
                      for (auto& var : op->indexVars) {
                          for (auto& svar : ctx_stack) {
                              if ((provGraph.getUnderivedAncestors(var)[0] == provGraph.getUnderivedAncestors(svar)[0]) && svar != var) {
                                  has_sibling = true;
                              }
                          }
                      }
                  }));
            if (is_equal && has_sibling) {
                to_change.push_back(a);
            }

            /// For none, if ctx_stack except the top contains indexVars in lhs, reduceOp will be +=.
            bool is_none = (a.getIndexSetRel() == none);
            bool has_outside = true;
            for (auto & var : seen) {
                for (auto &svar: ctx_stack) {
                    if (svar != ctx_stack.back() && var != svar) {
                        has_outside = false;
                    }
                }
            }
            if (is_none && has_outside) {
                to_change.push_back(a);
            }
        }
    };

    struct RedundantRewriter: public IndexNotationRewriter {
        using IndexNotationRewriter::visit;
        std::set<Assignment> to_change;
        RedundantRewriter(std::vector<Assignment>& to_change):to_change(to_change.begin(),to_change.end()){}

        void visit(const AssignmentNode* node) {
            Assignment a(node->lhs, node->rhs, node->op);
            for (auto & v: to_change) {
                if ((v.getLhs() == a.getLhs()) && (v.getRhs() == a.getRhs()) ) {
                    stmt = Assignment(a.getLhs(), a.getRhs(), Add());
                    return;
                }
            }
            IndexNotationRewriter::visit(node);
        }


    };

    PrecomputeRewriter rewriter;
    rewriter.precompute = *this;
    rewriter.provGraph = provGraph;
    rewriter.forallIndexVarList = forallIndexVars;
    stmt = rewriter.rewrite(stmt);
    std::vector<Assignment> to_change;
    RedundantVisitor findVisitor(to_change, provGraph);
    stmt.accept(&findVisitor);
    RedundantRewriter ReRewriter(to_change);
    stmt = ReRewriter.rewrite(stmt);
    return stmt;
}

void Precompute::print(std::ostream& os) const {
  os << "precompute(" << getExpr() << ", " << getIVars() << ", "
     << getIWVars() << ", " << getWorkspace() << ")";
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
          elementsMatched = 0;
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


IndexStmt scalarPromote(IndexStmt stmt, ProvenanceGraph provGraph, 
                        bool isWholeStmt, bool promoteScalar);

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
    map<TensorVar,ir::Expr> tensorVars;
    vector<ir::Expr> assembledByUngroupedInsert;
    set<IndexVar> definedIndexVars;
    set<IndexVar> reductionIndexVars;
    set<ParallelUnit> parentParallelUnits;
    std::string reason = "";

    IndexStmt rewriteParallel(IndexStmt stmt) {
      provGraph = ProvenanceGraph(stmt);

      const auto reductionVars = getReductionVars(stmt);

      reductionIndexVars.clear();
      for (const auto& iv : stmt.getIndexVars()) {
        if (util::contains(reductionVars, iv)) {
          for (const auto& rv : provGraph.getFullyDerivedDescendants(iv)) {
            reductionIndexVars.insert(rv);
          }
        }
      }

      tensorVars = createIRTensorVars(stmt);

      assembledByUngroupedInsert.clear();
      for (const auto& result : getAssembledByUngroupedInsertion(stmt)) {
        assembledByUngroupedInsert.push_back(tensorVars[result]);
      }

      return rewrite(stmt);
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = parallelize.geti();

      definedIndexVars.insert(foralli.getIndexVar());

      if (foralli.getIndexVar() == i) {
        // Precondition 1: No parallelization of reduction variables
        if (parallelize.getOutputRaceStrategy() == OutputRaceStrategy::NoRaces &&
            util::contains(reductionIndexVars, i)) {
          reason = "Precondition failed: Cannot parallelize reduction loops "
                   "without synchronization";
          return;
        }

        Iterators iterators(foralli, tensorVars);
        MergeLattice lattice = MergeLattice::make(foralli, iterators, provGraph, 
                                                  definedIndexVars);

        // Precondition 2: No coiteration of modes (i.e., merge lattice has 
        //                 only one iterator)
        if (lattice.iterators().size() != 1) {
          reason = "Precondition failed: The loop must not merge tensor "
                   "dimensions, that is, it must be a for loop;";
          return;
        }

        vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(i);
        IndexVar underivedAncestor = underivedAncestors.back();

        // Get lattice that corresponds to underived ancestor. This is 
        // bottom-most loop that shares underived ancestor
        Forall underivedForall = foralli;
        match(foralli.getStmt(),
              function<void(const ForallNode*)>([&](const ForallNode* node) {
                const auto nodeUnderivedAncestors = 
                    provGraph.getUnderivedAncestors(node->indexVar);
                definedIndexVars.insert(node->indexVar);
                if (underivedAncestor == nodeUnderivedAncestors.back()) {
                  underivedForall = Forall(node);
                }
              })
        );
        MergeLattice underivedLattice = MergeLattice::make(underivedForall, 
                                                           iterators, provGraph, 
                                                           definedIndexVars);

        // Precondition 3: Every result iterator must have insert capability
        for (Iterator iterator : underivedLattice.results()) {
          if (util::contains(assembledByUngroupedInsert, iterator.getTensor())) {
            for (Iterator it = iterator; !it.isRoot(); it = it.getParent()) {
              if (it.hasInsertCoord() || !it.isYieldPosPure()) {
                reason = "Precondition failed: The output tensor does not "
                         "support parallelized inserts";
                return;
              }
            }
          } else {
            while (true) {
              if (!iterator.hasInsert()) {
                reason = "Precondition failed: The output tensor must support " 
                         "inserts";
                return;
              }
              if (iterator.isLeaf()) {
                break;
              }
              iterator = iterator.getChild();
            }
          }
        }

        if (parallelize.getOutputRaceStrategy() == OutputRaceStrategy::Temporary &&
            util::contains(reductionIndexVars, underivedForall.getIndexVar())) {
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

          IndexStmt precomputed_stmt = forall(i, foralli.getStmt(), foralli.getMergeStrategy(), parallelize.getParallelUnit(), parallelize.getOutputRaceStrategy(), foralli.getUnrollFactor());
          for (auto assignment : precomputeAssignments) {
            // Construct temporary of correct type and size of outer loop
            TensorVar w(string("w_") + ParallelUnit_NAMES[(int) parallelize.getParallelUnit()], Type(assignment->lhs.getDataType(), {Dimension(i)}), taco::dense);

            // rewrite producer to write to temporary, mark producer as parallel
            IndexStmt producer = ReplaceReductionExpr(map<Access, Access>({{assignment->lhs, w(i)}})).rewrite(precomputed_stmt);
            taco_iassert(isa<Forall>(producer));
            Forall producer_forall = to<Forall>(producer);
            producer = forall(producer_forall.getIndexVar(), producer_forall.getStmt(), foralli.getMergeStrategy(), parallelize.getParallelUnit(), parallelize.getOutputRaceStrategy(), foralli.getUnrollFactor());

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
            IndexStmt consumer = forall(i, Assignment(assignment->lhs, w(i), assignment->op), foralli.getMergeStrategy(), reductionUnit, OutputRaceStrategy::ParallelReduction);
            precomputed_stmt = where(consumer, producer);
          }
          stmt = precomputed_stmt;
          return;
        }

        if (parallelize.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
          // want to avoid extra atomics by accumulating variable and then 
          // reducing at end
          IndexStmt body = scalarPromote(foralli.getStmt(), provGraph, 
                                         false, true);
          stmt = forall(i, body, foralli.getMergeStrategy(), parallelize.getParallelUnit(), 
                        parallelize.getOutputRaceStrategy(), 
                        foralli.getUnrollFactor());
          return;
        }


        stmt = forall(i, foralli.getStmt(), foralli.getMergeStrategy(), parallelize.getParallelUnit(), parallelize.getOutputRaceStrategy(), foralli.getUnrollFactor());
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


// class SetAssembleStrategy 

struct SetAssembleStrategy::Content {
  TensorVar result;
  AssembleStrategy strategy;
  bool separatelySchedulable;
};

SetAssembleStrategy::SetAssembleStrategy(TensorVar result, 
                                         AssembleStrategy strategy,
                                         bool separatelySchedulable) : 
    content(new Content) {
  content->result = result;
  content->strategy = strategy;
  content->separatelySchedulable = separatelySchedulable;
}

TensorVar SetAssembleStrategy::getResult() const {
  return content->result;
}

AssembleStrategy SetAssembleStrategy::getAssembleStrategy() const {
  return content->strategy;
}

bool SetAssembleStrategy::getSeparatelySchedulable() const {
  return content->separatelySchedulable;
}

IndexStmt SetAssembleStrategy::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  if (getAssembleStrategy() == AssembleStrategy::Append) {
    return stmt;
  }

  bool hasSeqInsertEdge = false;
  bool hasInsertCoord = false;
  bool hasNonpureYieldPos = false;
  for (const auto& modeFormat : getResult().getFormat().getModeFormats()) {
    if (hasSeqInsertEdge) {
      if (modeFormat.hasSeqInsertEdge()) {
        *reason = "Precondition failed: The output tensor does not support "
                  "ungrouped insertion (cannot have multiple modes requiring "
                  "non-trivial edge insertion)";
        return IndexStmt();
      }
    } else {
      hasSeqInsertEdge = (hasSeqInsertEdge || modeFormat.hasSeqInsertEdge());
      if (modeFormat.hasSeqInsertEdge()) {
        if (hasInsertCoord) {
          *reason = "Precondition failed: The output tensor does not support "
                    "ungrouped insertion (cannot have mode requiring "
                    "non-trivial coordinate insertion above mode requiring "
                    "non-trivial edge insertion)";
          return IndexStmt();
        }
        hasSeqInsertEdge = true;
      }
      hasInsertCoord = (hasInsertCoord || modeFormat.hasInsertCoord());
    }
    if (hasNonpureYieldPos && !modeFormat.isBranchless()) {
      *reason = "Precondition failed: The output tensor does not support "
                "ungrouped insertion (a mode that has a non-pure "
                "implementation of yield_pos cannot be followed by a "
                "non-branchless mode)";
      return IndexStmt();
    } else if (!modeFormat.isYieldPosPure()) {
      hasNonpureYieldPos = true;
    }
  }

  IndexStmt loweredQueries = stmt;

  // If attribute query computation should be independently schedulable, then 
  // need to use fresh index variables
  if (getSeparatelySchedulable()) {
    std::map<IndexVar,IndexVar> ivReplacements;
    for (const auto& indexVar : getIndexVars(stmt)) {
      ivReplacements[indexVar] = IndexVar("q" + indexVar.getName());
    }
    loweredQueries = replace(loweredQueries, ivReplacements);
  }

  // FIXME: Unneeded if scalar promotion is made default when concretizing
  loweredQueries = scalarPromote(loweredQueries);

  // Tracks all tensors that correspond to attribute query results or that are 
  // used to compute attribute queries
  std::set<TensorVar> insertedResults;  

  // Lower attribute queries to canonical forms using same schedule as 
  // actual computation
  Assemble::AttrQueryResults queryResults;
  struct LowerAttrQuery : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    TensorVar result;
    Assemble::AttrQueryResults& queryResults;
    std::set<TensorVar>& insertedResults;
    std::vector<TensorVar> arguments;
    std::vector<TensorVar> temps;
    std::map<TensorVar,TensorVar> tempReplacements;
    IndexStmt epilog;
    std::string reason = "";

    LowerAttrQuery(TensorVar result, Assemble::AttrQueryResults& queryResults, 
                   std::set<TensorVar>& insertedResults) : 
        result(result), queryResults(queryResults), 
        insertedResults(insertedResults) {}

    IndexStmt lower(IndexStmt stmt) {
      arguments = getArguments(stmt);
      temps = getTemporaries(stmt);
      for (const auto& tmp : temps) {
        tempReplacements[tmp] = TensorVar("q" + tmp.getName(), 
                                          Type(Bool, tmp.getType().getShape()), 
                                          tmp.getFormat());
      }

      queryResults = Assemble::AttrQueryResults();
      epilog = IndexStmt();
      stmt = IndexNotationRewriter::rewrite(stmt);
      if (epilog.defined()) {
        stmt = Where(epilog, stmt);
      }
      return stmt;
    }

    void visit(const ForallNode* op) {
      IndexStmt s = rewrite(op->stmt);
      if (s == op->stmt) {
        stmt = op;
      } else if (s.defined()) {
        stmt = Forall(op->indexVar, s, op->merge_strategy, op->parallel_unit, 
                      op->output_race_strategy, op->unrollFactor);
      } else {
        stmt = IndexStmt();
      }
    }

    void visit(const WhereNode* op) {
      IndexStmt producer = rewrite(op->producer);
      IndexStmt consumer = rewrite(op->consumer);
      if (producer == op->producer && consumer == op->consumer) {
        stmt = op;
      } else if (consumer.defined()) {
        stmt = producer.defined() ? Where(consumer, producer) : consumer;
      } else {
        stmt = IndexStmt();
      }
    }

    void visit(const AssignmentNode* op) {
      IndexExpr rhs = rewrite(op->rhs);

      const auto resultAccess = op->lhs;
      const auto resultTensor = resultAccess.getTensorVar();
      
      if (resultTensor != result) {
        // TODO: Should check that annihilator of original reduction op equals
        // fill value of original result
        Access lhs = to<Access>(rewrite(op->lhs));
        IndexExpr reduceOp = op->op.defined() ? Add() : IndexExpr();
        stmt = (rhs != op->rhs) ? Assignment(lhs, rhs, reduceOp) : op;
        return;
      }

      if (op->op.defined()) {
        reason = "Precondition failed: Ungrouped insertion not support for "
                 "output tensors that are scattered into";
        return;
      }

      queryResults[resultTensor] =
          std::vector<std::vector<TensorVar>>(resultTensor.getOrder());

      const auto indices = resultAccess.getIndexVars();
      const auto modeFormats = resultTensor.getFormat().getModeFormats();
      const auto modeOrdering = resultTensor.getFormat().getModeOrdering();

      std::vector<IndexVar> parentCoords;
      std::vector<IndexVar> childCoords;
      for (size_t i = 0; i < indices.size(); ++i) {
        childCoords.push_back(indices[modeOrdering[i]]);
      }

      for (size_t i = 0; i < indices.size(); ++i) {
        const auto modeName = resultTensor.getName() + std::to_string(i + 1);

        parentCoords.push_back(indices[modeOrdering[i]]);
        childCoords.erase(childCoords.begin());

        for (const auto& query:
            modeFormats[i].getAttrQueries(parentCoords, childCoords)) {
          const auto& groupBy = query.getGroupBy();

          // TODO: support multiple aggregations in single query
          taco_iassert(query.getAttrs().size() == 1);

          std::vector<Dimension> queryDims;
          for (const auto& coord : groupBy) {
            const auto pos = std::find(groupBy.begin(), groupBy.end(), coord)
                           - groupBy.begin();
            const auto dim = resultTensor.getType().getShape().getDimension(pos);
            queryDims.push_back(dim);
          }

          for (const auto& attr : query.getAttrs()) {
            switch (attr.aggr) {
              case AttrQuery::COUNT:
              {
                std::vector<IndexVar> dedupCoords = groupBy;
                dedupCoords.insert(dedupCoords.end(), attr.params.begin(),
                                   attr.params.end());
                std::vector<Dimension> dedupDims(dedupCoords.size());
                TensorVar dedupTmp(modeName + "_dedup", Type(Bool, dedupDims));
                stmt = Assignment(dedupTmp(dedupCoords), rhs, Add());
                insertedResults.insert(dedupTmp);

                const auto resultName = modeName + "_" + attr.label;
                TensorVar queryResult(resultName, Type(Int32, queryDims));
                epilog = Assignment(queryResult(groupBy),
                                    Cast(dedupTmp(dedupCoords), Int()), Add());
                for (const auto& coord : util::reverse(dedupCoords)) {
                  epilog = forall(coord, epilog);
                }
                insertedResults.insert(queryResult);

                queryResults[resultTensor][i] = {queryResult};
                return;
              }
              case AttrQuery::IDENTITY:
              case AttrQuery::MIN:
              case AttrQuery::MAX:
              default:
                taco_not_supported_yet;
                break;
            }
          }
        }
      }

      stmt = IndexStmt();
    }

    void visit(const AccessNode* op) {
      if (util::contains(arguments, op->tensorVar)) {
        expr = Access(op->tensorVar, op->indexVars, op->packageModifiers(),
                      true);
        return;
      } else if (util::contains(temps, op->tensorVar)) {
        expr = Access(tempReplacements[op->tensorVar], op->indexVars,
                      op->packageModifiers());
        return;
      }

      expr = op;
    }

    void visit(const CallNode* op) {
      std::vector<IndexExpr> args;
      bool rewritten = false;
      for(auto& arg : op->args) {
        IndexExpr rewrittenArg = rewrite(arg);
        args.push_back(rewrittenArg);
        if (arg != rewrittenArg) {
          rewritten = true;
        }
      }

      if (rewritten) {
        const std::map<IndexExpr, IndexExpr> subs = util::zipToMap(op->args, args);
        IterationAlgebra newAlg = replaceAlgIndexExprs(op->iterAlg, subs);

        struct InferSymbolic : public IterationAlgebraVisitorStrict {
          IndexExpr ret;

          IndexExpr infer(IterationAlgebra alg) {
            ret = IndexExpr();
            alg.accept(this);
            return ret;
          }
          virtual void visit(const RegionNode* op) {
            ret = op->expr();
          }

          virtual void visit(const ComplementNode* op) {
            taco_not_supported_yet;
          }

          virtual void visit(const IntersectNode* op) {
            IndexExpr lhs = infer(op->a);
            IndexExpr rhs = infer(op->b);
            ret = lhs * rhs;
          }

          virtual void visit(const UnionNode* op) {
            IndexExpr lhs = infer(op->a);
            IndexExpr rhs = infer(op->b);
            ret = lhs + rhs;
          }
        };
        expr = InferSymbolic().infer(newAlg);
      }
      else {
        expr = op;
      }
    }
  };
  LowerAttrQuery queryLowerer(getResult(), queryResults, insertedResults);
  loweredQueries = queryLowerer.lower(loweredQueries);

  if (!queryLowerer.reason.empty()) {
    *reason = queryLowerer.reason;
    return IndexStmt();
  }

  // Convert redundant reductions to assignments
  loweredQueries = eliminateRedundantReductions(loweredQueries, 
                                                &insertedResults);

  // Inline definitions of temporaries into their corresponding uses, as long 
  // as the temporaries are not the results of reductions
  std::set<TensorVar> inlinedResults;
  struct InlineTemporaries : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const std::set<TensorVar>& insertedResults;
    std::set<TensorVar>& inlinedResults;
    std::map<TensorVar,std::pair<IndexExpr,Assignment>> tmpUse;

    InlineTemporaries(const std::set<TensorVar>& insertedResults,
                      std::set<TensorVar>& inlinedResults) :
        insertedResults(insertedResults), inlinedResults(inlinedResults) {}

    void visit(const WhereNode* op) {
      IndexStmt consumer = rewrite(op->consumer);
      IndexStmt producer = rewrite(op->producer);
      if (producer == op->producer && consumer == op->consumer) {
        stmt = op;
      } else {
        stmt = new WhereNode(consumer, producer);
      }
    }

    void visit(const AssignmentNode* op) {
      const auto lhsTensor = op->lhs.getTensorVar();
      if (util::contains(tmpUse, lhsTensor) && !op->op.defined()) {
        std::map<IndexVar,IndexVar> indexMap;
        const auto& oldIndices = 
            to<Access>(tmpUse[lhsTensor].first).getIndexVars();
        const auto& newIndices = op->lhs.getIndexVars();
        for (const auto& mapping : util::zip(oldIndices, newIndices)) {
          indexMap[mapping.first] = mapping.second;
        }

        std::vector<IndexVar> newCoords;
        const auto& oldCoords = 
            tmpUse[lhsTensor].second.getLhs().getIndexVars();
        for (const auto& oldCoord : oldCoords) {
          newCoords.push_back(indexMap.at(oldCoord));
        }

        IndexExpr reduceOp = tmpUse[lhsTensor].second.getOperator();
        TensorVar queryResult = 
            tmpUse[lhsTensor].second.getLhs().getTensorVar();
        IndexExpr rhs = op->rhs;
        if (rhs.getDataType() != queryResult.getType().getDataType()) {
          rhs = Cast(rhs, queryResult.getType().getDataType());
        }
        stmt = Assignment(queryResult(newCoords), rhs, reduceOp);
        inlinedResults.insert(queryResult);
        return;
      } 

      const Access rhsAccess = isa<Access>(op->rhs) ? to<Access>(op->rhs)
          : (isa<Cast>(op->rhs) && isa<Access>(to<Cast>(op->rhs).getA()))
            ? to<Access>(to<Cast>(op->rhs).getA()) : Access();
      if (rhsAccess.defined()) {
        const auto rhsTensor = rhsAccess.getTensorVar();
        if (util::contains(insertedResults, rhsTensor)) {
          tmpUse[rhsTensor] = std::make_pair(rhsAccess, Assignment(op));
        }
      }
      stmt = op;
    }
  };
  loweredQueries = InlineTemporaries(insertedResults, 
                                     inlinedResults).rewrite(loweredQueries);

  // Eliminate computation of redundant temporaries
  struct EliminateRedundantTemps : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const std::set<TensorVar>& inlinedResults;

    EliminateRedundantTemps(const std::set<TensorVar>& inlinedResults) :
        inlinedResults(inlinedResults) {}

    void visit(const ForallNode* op) {
      IndexStmt s = rewrite(op->stmt);
      if (s == op->stmt) {
        stmt = op;
      } else if (s.defined()) {
        stmt = new ForallNode(op->indexVar, s, op->merge_strategy, op->parallel_unit, 
                              op->output_race_strategy, op->unrollFactor);
      } else {
        stmt = IndexStmt();
      }
    }

    void visit(const WhereNode* op) {
      IndexStmt consumer = rewrite(op->consumer);
      if (consumer == op->consumer) {
        stmt = op;
      } else if (consumer.defined()) {
        stmt = new WhereNode(consumer, op->producer);
      } else {
        stmt = op->producer;
      }
    }

    void visit(const AssignmentNode* op) {
      const auto lhsTensor = op->lhs.getTensorVar();
      if (util::contains(inlinedResults, lhsTensor)) {
        stmt = IndexStmt();
      } else {
        stmt = op;
      }
    }
  };
  loweredQueries = 
      EliminateRedundantTemps(inlinedResults).rewrite(loweredQueries);

  return Assemble(loweredQueries, stmt, queryResults);
}

void SetAssembleStrategy::print(std::ostream& os) const {
  os << "assemble(" << getResult() << ", " 
     << AssembleStrategy_NAMES[(int)getAssembleStrategy()] << ")";
}

std::ostream& operator<<(std::ostream& os, 
                         const SetAssembleStrategy& assemble) {
  assemble.print(os);
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
    for (const auto& temp : getTemporaries(stmt)) {
      // Don't parallelize computations that use non-scalar temporaries. 
      if (temp.getOrder() > 0) {
        return stmt;
      }
    }

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
        stmt = forall(*it, stmt, foralli.getMergeStrategy(), forallParallelUnit.at(*it), forallOutputRaceStrategy.at(*it), foralli.getUnrollFactor());
      }
      return;
    }

  };
  TopoReorderRewriter rewriter(sortedVars, dagBuilder.innerBody, 
                               dagBuilder.forallParallelUnit, dagBuilder.forallOutputRaceStrategy);
  return rewriter.rewrite(stmt);
}

IndexStmt scalarPromote(IndexStmt stmt, ProvenanceGraph provGraph, 
                        bool isWholeStmt, bool promoteScalar) {
  std::map<Access,const ForallNode*> hoistLevel;
  std::map<Access,IndexExpr> reduceOp;
  struct FindHoistLevel : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    std::map<Access,const ForallNode*>& hoistLevel;
    std::map<Access,IndexExpr>& reduceOp;
    std::map<Access,std::set<IndexVar>> hoistIndices;
    std::set<IndexVar> derivedIndices;
    std::set<IndexVar> indices;
    const ProvenanceGraph& provGraph;
    const bool isWholeStmt;
    const bool promoteScalar;
    
    FindHoistLevel(std::map<Access,const ForallNode*>& hoistLevel,
                   std::map<Access,IndexExpr>& reduceOp,
                   const ProvenanceGraph& provGraph,
                   bool isWholeStmt, bool promoteScalar) : 
        hoistLevel(hoistLevel), reduceOp(reduceOp), provGraph(provGraph),
        isWholeStmt(isWholeStmt), promoteScalar(promoteScalar) {}

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();

      // Don't allow hoisting out of forall's for GPU warp and block reduction
      if (foralli.getParallelUnit() == ParallelUnit::GPUWarpReduction || 
          foralli.getParallelUnit() == ParallelUnit::GPUBlockReduction) {
        FindHoistLevel findHoistLevel(hoistLevel, reduceOp, provGraph, false, 
                                      promoteScalar);
        foralli.getStmt().accept(&findHoistLevel);
        return;
      }

      std::vector<Access> resultAccesses;
      std::tie(resultAccesses, std::ignore) = getResultAccesses(foralli);
      for (const auto& resultAccess : resultAccesses) {
        if (!promoteScalar && resultAccess.getIndexVars().empty()) {
          continue;
        }

        std::set<IndexVar> resultIndices(resultAccess.getIndexVars().begin(),
                                         resultAccess.getIndexVars().end());
        if (std::includes(indices.begin(), indices.end(), 
                          resultIndices.begin(), resultIndices.end()) &&
            !util::contains(hoistLevel, resultAccess)) {
          hoistLevel[resultAccess] = node;
          hoistIndices[resultAccess] = indices;

          auto resultDerivedIndices = resultIndices;
          for (const auto& iv : resultIndices) {
            for (const auto& div : provGraph.getFullyDerivedDescendants(iv)) {
              resultDerivedIndices.insert(div);
            }
          }
          if (!isWholeStmt || resultDerivedIndices != derivedIndices) {
            reduceOp[resultAccess] = IndexExpr();
          }
        }
      }

      auto newIndices = provGraph.newlyRecoverableParents(i, derivedIndices);
      newIndices.push_back(i);
      derivedIndices.insert(newIndices.begin(), newIndices.end());

      const auto underivedIndices = getIndexVars(foralli);
      for (const auto& newIndex : newIndices) {
        if (util::contains(underivedIndices, newIndex)) {
          indices.insert(newIndex);
        }
      }

      IndexNotationVisitor::visit(node);

      for (const auto& newIndex : newIndices) {
        indices.erase(newIndex);
        derivedIndices.erase(newIndex);
      }
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
  FindHoistLevel findHoistLevel(hoistLevel, reduceOp, provGraph, isWholeStmt, 
                                promoteScalar);
  stmt.accept(&findHoistLevel);
  
  struct HoistWrites : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const std::map<Access,const ForallNode*>& hoistLevel;
    const std::map<Access,IndexExpr>& reduceOp;

    HoistWrites(const std::map<Access,const ForallNode*>& hoistLevel,
                const std::map<Access,IndexExpr>& reduceOp) : 
        hoistLevel(hoistLevel), reduceOp(reduceOp) {}

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();
      IndexStmt body = rewrite(foralli.getStmt());

      std::vector<IndexStmt> consumers;
      for (const auto& resultAccess : hoistLevel) {
        if (resultAccess.second == node) {
          // This assumes the index expression yields at most one result tensor; 
          // will not work correctly if there are multiple results.
          TensorVar resultVar = resultAccess.first.getTensorVar();
          TensorVar val("t" + i.getName() + resultVar.getName(), 
                        Type(resultVar.getType().getDataType(), {}));
          body = ReplaceReductionExpr(
              map<Access,Access>({{resultAccess.first, val()}})).rewrite(body);

          IndexExpr op = util::contains(reduceOp, resultAccess.first) 
                       ? reduceOp.at(resultAccess.first) : IndexExpr();
          IndexStmt consumer = Assignment(Access(resultAccess.first), val(), op);
          consumers.push_back(consumer);
        }
      }

      if (body == foralli.getStmt()) {
        taco_iassert(consumers.empty());
        stmt = node;
        return;
      }

      stmt = forall(i, body, foralli.getMergeStrategy(), foralli.getParallelUnit(),
                    foralli.getOutputRaceStrategy(), foralli.getUnrollFactor());
      for (const auto& consumer : consumers) {
        stmt = where(consumer, stmt);
      }
    }
  };
  HoistWrites hoistWrites(hoistLevel, reduceOp);
  return hoistWrites.rewrite(stmt);
}

IndexStmt scalarPromote(IndexStmt stmt) {
  return scalarPromote(stmt, ProvenanceGraph(stmt), true, false);
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

  // I think we can to linear combination of rows as long as there are no permutations in the format and the
  // level formats are ordered. The i -> k -> j loops should iterate over the data structures without issue.
  TensorVar B = Baccess.getTensorVar();
  if (!B.getFormat().getModeFormats()[0].isOrdered() ||
      !B.getFormat().getModeFormats()[1].isOrdered() ||
      B.getFormat().getModeOrdering()[0] != 0 ||
      B.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  TensorVar C = Caccess.getTensorVar();
  if (!C.getFormat().getModeFormats()[0].isOrdered() ||
      !C.getFormat().getModeFormats()[1].isOrdered() ||
      C.getFormat().getModeOrdering()[0] != 0 ||
      C.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  // It's an SpMM statement so return an optimized SpMM statement
  TensorVar w("w",
              Type(A.getType().getDataType(), 
              {A.getType().getShape().getDimension(1)}),
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