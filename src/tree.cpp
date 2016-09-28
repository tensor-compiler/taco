#include "tree.h"

#include <iostream>

#include "error.h"

using namespace std;

namespace tac {

// class TreeLevel
TreeLevelPtr TreeLevel::make(std::string format) {
  TreeLevelPtr level = values();
  for (size_t i=0; i < format.size(); ++i) {
    switch (format[i]) {
      case 'd':
        level = dense(level);
        break;
      case 's':
        level = sparse(level);
        break;
      case 'f':
        level = fixed(level);
        break;
      case 'r':
        level = replicated(level);
        break;
      default:
        uerror << "Format character not recognized: " << format[i];
    }
  }
  return level;
}

const TreeLevelPtr& TreeLevel::getChildren() const {
  return this->subLevel;
}

// class Values
void Values::accept(TreeVisitorStrict *v) const {
  v->visit(this);
}

// class Dense
void Dense::accept(TreeVisitorStrict *v) const {
  v->visit(this);
}

// class Sparse
void Sparse::accept(TreeVisitorStrict *v) const {
  v->visit(this);
}

// class Fixed
void Fixed::accept(TreeVisitorStrict *v) const {
  v->visit(this);
}

// class Replicated
void Replicated::accept(TreeVisitorStrict *v) const {
  v->visit(this);
}

// Factory functions
TreeLevelPtr values() {
  return shared_ptr<TreeLevel>(new Values());
}

TreeLevelPtr dense(const TreeLevelPtr& subLevel) {
  return shared_ptr<TreeLevel>(new Dense(subLevel));
}

TreeLevelPtr sparse(const TreeLevelPtr& subLevel) {
  return shared_ptr<TreeLevel>(new Sparse(subLevel));
}

TreeLevelPtr fixed(const TreeLevelPtr& subLevel) {
  return shared_ptr<TreeLevel>(new Fixed(subLevel));
}

TreeLevelPtr replicated(const TreeLevelPtr& subLevel) {
  return shared_ptr<TreeLevel>(new Replicated(subLevel));
}


// class TreeVisitor
void TreeVisitor::visit(const Values* tl) {
}

void TreeVisitor::visit(const Dense* tl) {
  tl->getChildren()->accept(this);
}

void TreeVisitor::visit(const Sparse* tl) {
  tl->getChildren()->accept(this);
}

void TreeVisitor::visit(const Fixed* tl) {
  tl->getChildren()->accept(this);
}

void TreeVisitor::visit(const Replicated* tl) {
  tl->getChildren()->accept(this);
}

std::ostream &operator<<(std::ostream& os, const shared_ptr<TreeLevel>& tl) {
  class TreePrinter : public TreeVisitorStrict {
  public:
    TreePrinter(ostream& os) : os{os} {}

    void print(const shared_ptr<TreeLevel>& tl) {
      tl->accept(this);
    }

  private:
    void visit(const Values* tl) {
    }

    void visit(const Dense* tl) {
      os << "d";
      print(tl->getChildren());
    }

    void visit(const Sparse* tl) {
      os << "s";
      print(tl->getChildren());
    }

    void visit(const Fixed* tl) {
      os << "f";
      print(tl->getChildren());
    }

    void visit(const Replicated* tl) {
      os << "r";
      print(tl->getChildren());
    }

  private:
    ostream& os;
  };

  TreePrinter(os).print(tl);
  return os;
}

}
