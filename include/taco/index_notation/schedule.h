#ifndef TACO_SCHEDULE_H
#define TACO_SCHEDULE_H

#include <memory>
#include <vector>

namespace taco {

class TensorVar;
class IndexVar;
class IndexExpr;
class IndexStmt;


/// A transformation is an optimization that transforms a statement in the
/// concrete index notation into a new statement that computes the same result
/// in a different way.  Schedule commands affect the order things are computed
/// in as well as where temporary results are stored.
class Transformation {
public:
  virtual ~Transformation() = default;
  virtual bool isValid(IndexStmt stmt, std::string* reason=nullptr) = 0;
  virtual IndexStmt apply(IndexStmt stmt) = 0;
};


/// The reorder optimization rewrites an index statement to swap the order of
/// the `i` and `j` loops.
class Reorder : public Transformation {
public:
  Reorder();
  Reorder(IndexVar i, IndexVar j);

  IndexVar geti() const;
  IndexVar getj() const;

  /// Checks whether it is valid to reorder the given index variables in `stmt`.
  bool isValid(IndexStmt stmt, std::string* reason=nullptr);

  /// Apply the reorder optimization to a concrete index statement.
  IndexStmt apply(IndexStmt stmt);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a reorder command.
std::ostream& operator<<(std::ostream&, const Reorder&);


/// The precompute optimizaton rewrites an index expression to precompute `expr`
/// and store it to the given workspace.
class Precompute : public Transformation {
public:
  Precompute();
  Precompute(IndexExpr expr, IndexVar i, IndexVar iw, TensorVar workspace);

  IndexExpr getExpr() const;
  IndexVar geti() const;
  IndexVar getiw() const;
  TensorVar getWorkspace() const;

  /// Checks whether it is valid to precompute the given expression in `stmt`.
  bool isValid(IndexStmt stmt, std::string* reason=nullptr);

  /// Apply the precompute optimization to a concrete index statement.
  IndexStmt apply(IndexStmt stmt);

  bool defined() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a precompute command.
std::ostream& operator<<(std::ostream&, const Precompute&);


/// A schedule controls code generation and determines how index expression
/// should be computed.
class Schedule {
public:
  Schedule();

  /// Returns the workspace commands in the schedule.
  std::vector<Precompute> getPrecomputes() const;

  /// Returns the workspace of `expr`.  The result is undefined if `expr` is not
  /// stored to a workspace.
  Precompute getPrecompute(IndexExpr expr) const;

  /// Add a workspace command to the schedule.
  void addPrecompute(Precompute precompute);

  /// Removes workspace commands from the schedule.
  void clearPrecomputes();

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a schedule.
std::ostream& operator<<(std::ostream&, const Schedule&);

}
#endif
