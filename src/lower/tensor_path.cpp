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
};

TensorPath::TensorPath() : content(nullptr) {
}

TensorPath::TensorPath(Tensor tensor, vector<Var> path)
    : content(new TensorPath::Content(tensor, path)) {
}

size_t TensorPath::getSize() const {
  return getVariables().size();
}

const Tensor& TensorPath::getTensor() const {
  return content->tensor;
}

const std::vector<Var>& TensorPath::getVariables() const {
  return content->variables;
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
TensorPathStep::TensorPathStep() {
}

TensorPathStep::TensorPathStep(const TensorPath& path, int step)
    : path(path), step(step) {
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
  return os << step.getPath().getTensor().getName()
            << step.getPath().getVariables()[step.getStep()];
}

}}
