#include "tensor_path.h"

#include <vector>
#include <iostream>

#include "var.h"
#include "internal_tensor.h"
#include "error.h"
#include "util/collections.h"

using namespace std;
using namespace taco::internal;

namespace taco {
namespace lower {

// class TensorPath
struct TensorPath::Content {
  Content(Tensor tensor, vector<Var> variables)
      : tensor(tensor), variables(variables) {}
  Tensor                 tensor;
  vector<Var>            variables;
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

size_t TensorPath::getSize() const {
  return content->variables.size();
}

TensorPathStep TensorPath::getStep(size_t i) const {
  iassert(i < content->variables.size());
  return TensorPathStep(*this, (int)i);
}

TensorPathStep TensorPath::getStep(const Var& var) const {
  if (!defined() || !util::contains(content->variables, var)) {
    return TensorPathStep();
  }
  auto i = util::locate(content->variables, var);
  iassert(i < content->variables.size());
  return getStep(i);
}

TensorPathStep TensorPath::getLastStep() const {
  return getStep(getSize()-1);
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
                ? to_string(step.getStep())
                : "root");
}

// convenience functions
vector<TensorPathStep> getRandomAccessSteps(vector<TensorPathStep> steps) {
  vector<TensorPathStep> randomAccessSteps;
  for (TensorPathStep& step : steps) {
    Format format = step.getPath().getTensor().getFormat();
    if (format.getLevels()[step.getStep()].getType() == LevelType::Dense) {
      randomAccessSteps.push_back(step);
    }
  }
  return randomAccessSteps;
}

vector<TensorPathStep> getSequentialAccessSteps(vector<TensorPathStep> steps) {
  vector<TensorPathStep> sequentialAccessSteps;
  for (TensorPathStep& step : steps) {
    Format format = step.getPath().getTensor().getFormat();
    if (format.getLevels()[step.getStep()].getType() == LevelType::Sparse) {
      sequentialAccessSteps.push_back(step);
    }
  }
  return sequentialAccessSteps;
}

}}
