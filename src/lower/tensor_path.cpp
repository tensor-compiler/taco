#include "tensor_path.h"

#include <vector>
#include <iostream>

#include "taco/tensor.h"
#include "taco/expr.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace lower {

// class TensorPath
struct TensorPath::Content {
  TensorBase  tensor;
  vector<IndexVar> vars;
};

TensorPath::TensorPath() : content(nullptr) {
}

TensorPath::TensorPath(const TensorBase& tensor, const vector<IndexVar>& vars)
    : content(new TensorPath::Content) {
  content->tensor = tensor;
  content->vars   = vars;
}

const TensorBase& TensorPath::getTensor() const {
  return content->tensor;
}

const std::vector<IndexVar>& TensorPath::getVariables() const {
  return content->vars;
}

size_t TensorPath::getSize() const {
  return getVariables().size();
}

TensorPathStep TensorPath::getStep(size_t i) const {
  taco_iassert(i < getVariables().size());
  return TensorPathStep(*this, (int)i);
}

TensorPathStep TensorPath::getLastStep() const {
  return getStep(getSize()-1);
}

TensorPathStep TensorPath::getStep(const IndexVar& var) const {
  auto& vars = getVariables();
  if (!defined() || !util::contains(vars, var)) {
    return TensorPathStep();
  }
  auto i = util::locate(vars, var);
  taco_iassert(i < vars.size());
  return getStep(i);
}

bool TensorPath::defined() const {
  return content != nullptr;
}

bool operator==(const TensorPath& l, const TensorPath& r) {
  return l.content == r.content;
}

bool operator<(const TensorPath& l, const TensorPath& r) {
  return l.content < r.content;
}

std::ostream& operator<<(std::ostream& os, const TensorPath& path) {
  if (!path.defined()) return os << "Path()";
  return os << path.getTensor().getName() << "["
            << "->" << util::join(path.getVariables(), "->") << "]";
}


// class TensorPathStep
TensorPathStep::TensorPathStep() : step(-1) {
}

TensorPathStep::TensorPathStep(const TensorPath& path, int step)
    : path(path), step(step) {
  taco_iassert(step >= -1);
  taco_iassert(step < (int)path.getVariables().size())
      << "step: " << step << std::endl << "path: " << path;
}

const TensorPath& TensorPathStep::getPath() const {
  return path;
}

int TensorPathStep::getStep() const {
  return step;
}

bool operator==(const TensorPathStep& l, const TensorPathStep& r) {
  return l.getPath() == r.getPath() && l.getStep() == r.getStep();
}

bool operator<(const TensorPathStep& l, const TensorPathStep& r) {
  return (l.getPath() != r.getPath()) ? l.getPath() < r.getPath()
                                      : l.getStep() < r.getStep();
}

std::ostream& operator<<(std::ostream& os, const TensorPathStep& step) {
  if (!step.getPath().defined()) return os << "Step()";
  return os << step.getPath().getTensor().getName()
            << (step.getStep() >= 0
                ? to_string(step.getStep())
                : "root");
}

}}
