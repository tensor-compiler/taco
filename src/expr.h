#ifndef TAC_EXPR_H
#define TAC_EXPR_H

#include <string>

#include "util/intrusive_ptr.h"

namespace taco {

namespace internal {
class IRNode {

};

class IRHandle : public util::IntrusivePtr<const IRNode> {

};
}

class Var {
public:
  Var(const std::string& name);
  Var();

  std::string getName() const {return this->name;}

private:
  std::string name;
};

class Expr : public internal::IRHandle {
public:

private:

};

}

#endif
