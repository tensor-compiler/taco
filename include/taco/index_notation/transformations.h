#ifndef TACO_TRANSFORMATIONS_H
#define TACO_TRANSFORMATIONS_H

#include <memory>
#include <string>
#include <ostream>

namespace taco {

class TensorVar;
class IndexVar;
class IndexExpr;
class IndexStmt;

class TransformationInterface;
class Reorder;
class Precompute;

/// A transformation is an optimization that transforms a statement in the
/// concrete index notation into a new statement that computes the same result
/// in a different way.  Transformations affect the order things are computed
/// in as well as where temporary results are stored.
class Transformation {
public:
  Transformation(Reorder);
  Transformation(Precompute);

  IndexStmt apply(IndexStmt stmt, std::string* reason=nullptr) const;

  friend std::ostream& operator<<(std::ostream&, const Transformation&);

private:
  std::shared_ptr<const TransformationInterface> transformation;
};


/// Transformation abstract class
class TransformationInterface {
public:
  virtual ~TransformationInterface() = default;
  virtual IndexStmt apply(IndexStmt stmt, std::string* reason=nullptr) const =0;
  virtual void print(std::ostream& os) const = 0;
};


/// The reorder optimization rewrites an index statement to swap the order of
/// the `i` and `j` loops.
class Reorder : public TransformationInterface {
public:
  Reorder(IndexVar i, IndexVar j);

  IndexVar geti() const;
  IndexVar getj() const;

  /// Apply the reorder optimization to a concrete index statement.  Returns
  /// an undefined statement and a reason if the statement cannot be lowered.
  IndexStmt apply(IndexStmt stmt, std::string* reason=nullptr) const;

  void print(std::ostream& os) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a reorder command.
std::ostream& operator<<(std::ostream&, const Reorder&);


/// The precompute optimizaton rewrites an index expression to precompute `expr`
/// and store it to the given workspace.
class Precompute : public TransformationInterface {
public:
  Precompute();
  Precompute(IndexExpr expr, IndexVar i, IndexVar iw, TensorVar workspace);

  IndexExpr getExpr() const;
  IndexVar geti() const;
  IndexVar getiw() const;
  TensorVar getWorkspace() const;

  /// Apply the precompute optimization to a concrete index statement.
  IndexStmt apply(IndexStmt stmt, std::string* reason=nullptr) const;

  void print(std::ostream& os) const;

  bool defined() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a precompute command.
std::ostream& operator<<(std::ostream&, const Precompute&);

}
#endif
