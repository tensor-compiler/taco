#include "tree_visitor.h"

#include <string>
#include <iostream>

#include "storage.h"

using namespace std;

namespace tac {

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
