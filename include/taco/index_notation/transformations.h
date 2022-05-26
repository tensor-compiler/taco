#ifndef TACO_TRANSFORMATIONS_H
#define TACO_TRANSFORMATIONS_H

#include <memory>
#include <string>
#include <ostream>
#include <vector>
#include "index_notation.h"

namespace taco {

class TensorVar;
class IndexVar;
class IndexExpr;
class IndexStmt;

class TransformationInterface;
class Reorder;
class Precompute;
class ForAllReplace;
class AddSuchThatPredicates;
class Parallelize;
class TopoReorder;
class SetAssembleStrategy;
class SetMergeStrategy;

/// A transformation is an optimization that transforms a statement in the
/// concrete index notation into a new statement that computes the same result
/// in a different way.  Transformations affect the order things are computed
/// in as well as where temporary results are stored.
class Transformation {
public:
  Transformation(Reorder);
  Transformation(Precompute);
  Transformation(ForAllReplace);
  Transformation(Parallelize);
  Transformation(TopoReorder);
  Transformation(AddSuchThatPredicates);
  Transformation(SetAssembleStrategy);
  Transformation(SetMergeStrategy);

  IndexStmt apply(IndexStmt stmt, std::string *reason = nullptr) const;

  friend std::ostream &operator<<(std::ostream &, const Transformation &);

private:
  std::shared_ptr<const TransformationInterface> transformation;
};


/// Transformation abstract class
class TransformationInterface {
public:
  virtual ~TransformationInterface() = default;
  virtual IndexStmt apply(IndexStmt stmt, std::string *reason = nullptr) const = 0;
  virtual void print(std::ostream &os) const = 0;
};


/// The reorder optimization rewrites an index statement to swap the order of
/// the `i` and `j` loops.
/// Can also supply replacePattern and will find nested foralls with this set of indexvar
/// and reorder them to new ordering
class Reorder : public TransformationInterface {
public:
  Reorder(IndexVar i, IndexVar j);
  Reorder(std::vector<IndexVar> replacePattern);

  IndexVar geti() const;
  IndexVar getj() const;
  const std::vector<IndexVar>& getreplacepattern() const;

  /// Apply the reorder optimization to a concrete index statement.  Returns
  /// an undefined statement and a reason if the statement cannot be lowered.
  IndexStmt apply(IndexStmt stmt, std::string *reason = nullptr) const;

  void print(std::ostream &os) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a reorder command.
std::ostream &operator<<(std::ostream &, const Reorder &);


/// The precompute optimizaton rewrites an index expression to precompute `expr`
/// and store it to the given workspace.
class Precompute : public TransformationInterface {
public:
  Precompute();
  Precompute(IndexExpr expr, IndexVar i, IndexVar iw, TensorVar workspace);
  Precompute(IndexExpr expr, std::vector<IndexVar> i_vars,
             std::vector<IndexVar> iw_vars, TensorVar workspace);
  
  IndexExpr getExpr() const;
  std::vector<IndexVar>& getIVars() const;
  std::vector<IndexVar>& getIWVars() const;
  TensorVar getWorkspace() const;

  /// Apply the precompute optimization to a concrete index statement.
  IndexStmt apply(IndexStmt stmt, std::string *reason = nullptr) const;

  void print(std::ostream &os) const;

  bool defined() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a precompute command.
std::ostream &operator<<(std::ostream &, const Precompute &);


/// Replaces all occurrences of directly nested forall nodes of pattern with
/// directly nested loops of replacement
class ForAllReplace : public TransformationInterface {
public:
  ForAllReplace();

  ForAllReplace(std::vector<IndexVar> pattern, std::vector<IndexVar> replacement);

  std::vector<IndexVar> getPattern() const;

  std::vector<IndexVar> getReplacement() const;

  IndexStmt apply(IndexStmt stmt, std::string *reason = nullptr) const;

  void print(std::ostream &os) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a ForAllReplace command.
std::ostream &operator<<(std::ostream &, const ForAllReplace &);


/// Adds a SuchThat node if it does not exist and adds the given IndexVarRels
class AddSuchThatPredicates : public TransformationInterface {
public:
  AddSuchThatPredicates();

  AddSuchThatPredicates(std::vector<IndexVarRel> predicates);

  std::vector<IndexVarRel> getPredicates() const;

  IndexStmt apply(IndexStmt stmt, std::string *reason = nullptr) const;

  void print(std::ostream &os) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const AddSuchThatPredicates&);


/// The parallelize optimization tags a Forall as parallelized
/// after checking for preconditions
class Parallelize : public TransformationInterface {
public:
  Parallelize();
  Parallelize(IndexVar i);
  Parallelize(IndexVar i, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy);

  IndexVar geti() const;
  ParallelUnit getParallelUnit() const;
  OutputRaceStrategy getOutputRaceStrategy() const;

  /// Apply the parallelize optimization to a concrete index statement.
  IndexStmt apply(IndexStmt stmt, std::string* reason=nullptr) const;

  void print(std::ostream& os) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a parallelize command.
std::ostream& operator<<(std::ostream&, const Parallelize&);


class SetAssembleStrategy : public TransformationInterface {
public:
  SetAssembleStrategy(TensorVar result, AssembleStrategy strategy, 
                      bool separatelySchedulable);

  TensorVar getResult() const;
  AssembleStrategy getAssembleStrategy() const;
  bool getSeparatelySchedulable() const;

  IndexStmt apply(IndexStmt stmt, std::string *reason = nullptr) const;

  void print(std::ostream &os) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a SetAssembleStrategy command.
std::ostream &operator<<(std::ostream &, const SetAssembleStrategy&);

class SetMergeStrategy : public TransformationInterface {
public:
  SetMergeStrategy(IndexVar i, MergeStrategy strategy);

  IndexVar geti() const;
  MergeStrategy getMergeStrategy() const;

  IndexStmt apply(IndexStmt stmt, std::string *reason = nullptr) const;

  void print(std::ostream &os) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a SetMergeStrategy command.
std::ostream &operator<<(std::ostream &, const SetMergeStrategy&);

// Autoscheduling functions

/**
 * Parallelize the outer forallall loop if it passes preconditions.
 * The preconditions are:
 * 1. The loop iterates over only one data structure,
 * 2. Every result iterator has the insert capability, and
 * 3. No cross-thread reductions.
 */
IndexStmt parallelizeOuterLoop(IndexStmt stmt);

/**
 * Topologically reorder ForAlls so that all tensors are iterated in order.
 * Only reorders first contiguous section of ForAlls iterators form constraints
 * on other dimensions. For example, a {dense, dense, sparse, dense, dense}
 * tensor has constraints i -> k, j -> k, k -> l, k -> m.
 */
IndexStmt reorderLoopsTopologically(IndexStmt stmt);

/**
 * Performs scalar promotion so that reductions are done by accumulating into 
 * scalar temporaries whenever possible.
 */
IndexStmt scalarPromote(IndexStmt stmt);

/**
 * Insert where statements with temporaries into the following statements kinds:
 * 1. The result is a is scattered into but does not support random insert.
 */
IndexStmt insertTemporaries(IndexStmt stmt);

}
#endif
