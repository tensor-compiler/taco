#include "tensor_path.h"

#include <vector>

#include "var.h"
#include "internal_tensor.h"
#include "error.h"

using namespace std;
using namespace taco::internal;

namespace taco {
namespace is {

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
struct TensorPathStep::Content {
  TensorPath path;
  size_t step;
};

TensorPathStep::TensorPathStep() : content(nullptr) {
}

TensorPathStep::TensorPathStep(const TensorPath& path, size_t step)
    : content(new TensorPathStep::Content) {
  iassert(step < path.getVariables().size());
  content->path = path;
  content->step = step;
}

const TensorPath& TensorPathStep::getPath() const {
  return content->path;
}

size_t TensorPathStep::getStep() const {
  return content->step;
}

bool operator==(const TensorPathStep& l, const TensorPathStep& r) {
  return l.content == r.content;
}

bool operator<(const TensorPathStep& l, const TensorPathStep& r) {
  return l.content < r.content;
}

std::ostream& operator<<(std::ostream& os, const TensorPathStep& step) {
  return os << step.getPath().getTensor().getName()
            << step.getPath().getVariables()[step.getStep()];
}

}}
