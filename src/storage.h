#ifndef TAC_STORAGE_H
#define TAC_STORAGE_H

#include <string>
#include <memory>
#include <ostream>

#include "tree_visitor.h"

namespace tac {

class TreeLevel;

typedef std::shared_ptr<TreeLevel> TreeLevelPtr;

class TensorStorage {
public:
  TensorStorage() : forest{nullptr} {}
  
  TensorStorage(std::string format);

  friend std::ostream &operator<<(std::ostream&, const TensorStorage&);

private:
  TreeLevelPtr forest;
};


class TreeLevel {
public:
  static TreeLevelPtr make(std::string format);

  TreeLevel(const TreeLevelPtr& subLevel) : subLevel{subLevel} {}

  const TreeLevelPtr& getChildren() const;

  virtual void accept(TreeVisitorStrict* v) const = 0;

private:
  TreeLevelPtr subLevel;
};


/**
 * Tree level storing actual values. Value tree levels terminate a tree format.
 */
class Values : public TreeLevel {
public:
  Values() : TreeLevel(nullptr) {}

  void accept(TreeVisitorStrict *v) const {v->visit(this);}
};


class Dense : public TreeLevel {
public:
  Dense(const TreeLevelPtr& subLevel) : TreeLevel(subLevel) {}

  void accept(TreeVisitorStrict *v) const {v->visit(this);}
};


class Sparse : public TreeLevel {
public:
  Sparse(const TreeLevelPtr& subLevel) : TreeLevel(subLevel) {}

  void accept(TreeVisitorStrict *v) const {v->visit(this);}
};


class Fixed : public TreeLevel {
public:
  Fixed(const TreeLevelPtr& subLevel) : TreeLevel(subLevel) {}

  void accept(TreeVisitorStrict *v) const {v->visit(this);}
};


class Replicated : public TreeLevel {
public:
  Replicated(const TreeLevelPtr& subLevel) : TreeLevel(subLevel) {}

  void accept(TreeVisitorStrict *v) const {v->visit(this);}
};


// Factory functions
TreeLevelPtr values();
TreeLevelPtr dense(const TreeLevelPtr& subLevel);
TreeLevelPtr sparse(const TreeLevelPtr& subLevel);
TreeLevelPtr fixed(const TreeLevelPtr& subLevel);
TreeLevelPtr replicated(const TreeLevelPtr& subLevel);

}
#endif
