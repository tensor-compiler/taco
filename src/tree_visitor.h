#ifndef TAC_TREE_PRINTER_H
#define TAC_TREE_PRINTER_H

#include <ostream>
#include <memory>

namespace tac {

class TreeLevel;
class Values;
class Dense;
class Sparse;
class Fixed;
class Replicated;

class TreeVisitorStrict {
public:
  virtual void visit(const Values* tl)     = 0;
  virtual void visit(const Dense* tl)      = 0;
  virtual void visit(const Sparse* tl)     = 0;
  virtual void visit(const Fixed* tl)      = 0;
  virtual void visit(const Replicated* tl) = 0;
};

class TreeVisitor : public TreeVisitorStrict {
public:
  virtual void visit(const Values* tl);
  virtual void visit(const Dense* tl);
  virtual void visit(const Sparse* tl);
  virtual void visit(const Fixed* tl);
  virtual void visit(const Replicated* tl);
};

std::ostream &operator<<(std::ostream&, const std::shared_ptr<TreeLevel>&);

}
#endif
