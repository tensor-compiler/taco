#include "tensor_path.h"

#include <vector>

#include "var.h"
#include "internal_tensor.h"
#include "error.h"

using namespace std;
using namespace taco::internal;

namespace taco {
namespace lower {

// class TensorPath
struct TensorPath::Content {
  Content(Tensor tensor, vector<Var> variables)
      : tensor(tensor), variables(variables) {
  }
  Tensor tensor;
  vector<Var> variables;

  vector<TensorPathStep> steps;
};

TensorPath::TensorPath() : content(nullptr) {
}

TensorPath::TensorPath(Tensor tensor, vector<Var> path)
    : content(new TensorPath::Content(tensor, path)) {
  for (size_t i=0; i < path.size(); ++i) {
    content->steps.push_back(TensorPathStep(*this, i));
  }
}

const Tensor& TensorPath::getTensor() const {
  return content->tensor;
}

const std::vector<Var>& TensorPath::getVariables() const {
  return content->variables;
}

size_t TensorPath::getSize() const {
  return content->steps.size();
}

const TensorPathStep& TensorPath::getStep(size_t i) const {
  iassert(i < content->steps.size());
  return content->steps[i];
}

const TensorPathStep& TensorPath::getLastStep() const {
  return content->steps[content->steps.size()-1];
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

std::ostream& operator<<(std::ostream& os, const TensorPath& tensorPath) {
  return os << tensorPath.getTensor().getName() << "["
            << "->" << util::join(tensorPath.getVariables(), "->") << "]";
}


// class TensorPathStep
TensorPathStep::TensorPathStep() : step(-1) {
}

TensorPathStep::TensorPathStep(const TensorPath& path, int step)
    : path(path), step(step) {
  iassert(step >= -1);
  iassert(step < (int)path.getVariables().size())
      << "step: " << step << std::endl << "path: " << path;
}

const TensorPath& TensorPathStep::getPath() const {
  return path;
}

int TensorPathStep::getStep() const {
  return step;
}

bool operator==(const TensorPathStep& l, const TensorPathStep& r) {
  return l.path == r.path && l.step == r.step;
}

bool operator<(const TensorPathStep& l, const TensorPathStep& r) {
  return (l.path != r.path) ? l.path < r.path : l.step < r.step;
}

std::ostream& operator<<(std::ostream& os, const TensorPathStep& step) {
  if (!step.getPath().defined()) return os << "Step()";
  return os << step.getPath().getTensor().getName()
            << (step.getStep() >= 0
                ? util::toString(step.getPath().getVariables()[step.getStep()])
                : "root");
}

}}
