#ifndef TACO_SCHEDULE_H
#define TACO_SCHEDULE_H

#include <memory>
#include <vector>

namespace taco {

class TensorVar;
class IndexVar;
class IndexExpr;
class IndexStmt;


/// The reorder optimization rewrites an index statement to swap the order of
/// the `i` and `j` loops.
class Reorder {
public:
  Reorder();
  Reorder(IndexVar i, IndexVar j);

  IndexVar geti() const;
  IndexVar getj() const;

  /// Apply the reorder optimization to a concrete index statement.
  IndexStmt apply(IndexStmt stmt);

private:
  struct Content;
  std::shared_ptr<Content> content;
};


/// The workspace optimizaton rewrites an index expression to precompute `expr`
/// and store it to a workspace.
class Workspace {
public:
  Workspace();
  Workspace(IndexExpr expr, IndexVar i, IndexVar iw, TensorVar workspace);

  IndexExpr getExpr() const;
  IndexVar geti() const;
  IndexVar getiw() const;
  TensorVar getWorkspace() const;

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
