#include "tensor_path.h"

#include <vector>

#include "var.h"
#include "internal_tensor.h"

using namespace std;
using namespace taco::internal;

namespace taco {
namespace is {

// class TensorPath
struct TensorPath::Content {
  Content(Tensor tensor, vector<Var> path) : tensor(tensor), path(path) {
  }
  Tensor tensor;
  vector<Var> path;
};

TensorPath::TensorPath(Tensor tensor, vector<Var> path)
    : content(new TensorPath::Content(tensor, path)) {
}

const Tensor& TensorPath::getTensor() const {
  return content->tensor;
}

const std::vector<Var>& TensorPath::getPath() const {
  return content->path;
}

std::ostream& operator<<(std::ostream& os, const TensorPath& tensorPath) {
  return os << tensorPath.getTensor().getName() << " ("
            << "->" << util::join(tensorPath.getPath(), "->") << ")";
}

}}
