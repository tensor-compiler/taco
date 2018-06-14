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

  /// Checks whether it is valid to apply the reorder to stmt, and that all the
  /// preconditions pass.
  bool isValid(IndexStmt stmt, std::string* reason=nullptr);

  /// Apply the reorder optimization to a concrete index statement.
  IndexStmt apply(IndexStmt stmt);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a workspace command.
std::ostream& operator<<(std::ostream&, const Reorder&);


/// The workspace optimizaton rewrites an index expression to precompute `expr`
/// and store it to a workspace.
class Workspace : public Transformation {
public:
  Workspace();
  Workspace(IndexExpr expr, IndexVar i, IndexVar iw, TensorVar workspace);

  IndexExpr getExpr() const;
  IndexVar geti() const;
  IndexVar getiw() const;
  TensorVar getWorkspace() const;

  /// Checks whether it is valid to apply the workspace to stmt, and that all
  /// the preconditions pass.
  bool isValid(IndexStmt stmt, std::string* reason=nullptr);

  /// Apply the workspace optimization to a concrete index statement.
  IndexStmt apply(IndexStmt stmt);

  bool defined() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a workspace command.
std::ostream& operator<<(std::ostream&, const Workspace&);


/// A schedule controls code generation and determines how index expression
/// should be computed.
class Schedule {
public:
  Schedule();

  /// Returns the workspace commands in the schedule.
  std::vector<Workspace> getWorkspaces() const;

  /// Returns the workspace of `expr`.  The result is undefined if `expr` is not
  /// stored to a workspace.
  Workspace getWorkspace(IndexExpr expr) const;

  /// Add a workspace command to the schedule.
  void addWorkspace(Workspace workspace);

  /// Removes workspace commands from the schedule.
  void clearWorkspaces();

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a schedule.
std::ostream& operator<<(std::ostream&, const Schedule&);

}
#endif
