#ifndef TACO_INTERNAL_TENSOR_H
#define TACO_INTERNAL_TENSOR_H

#include <memory>
#include <string>
#include <vector>

#include "expr.h"
#include "ir.h"
#include "ir_printer.h"
#include "format.h"
#include "component_types.h"
#include "packed_tensor.h"
#include "util/intrusive_ptr.h"
#include "util/strings.h"

namespace taco {
template <typename T> struct Read;
template <typename T> class Tensor;

namespace internal {

class Tensor : public util::Manageable<Tensor> {
  friend class  taco::Tensor<double>;
  friend struct Read<double>;

  Tensor(std::string name, std::vector<size_t> dimensions, Format format)
      : name(name), dimensions(dimensions), format(format) {
  }

  std::string getName() const {
    return name;
  }

  Format getFormat() const {
    return format;
  }

  const std::vector<size_t>& getDimensions() const {
    return dimensions;
  }

  size_t getOrder() const {
    return dimensions.size();
  }

  const std::vector<taco::Var>& getIndexVars() const {
    return indexVars;
  }

  taco::Expr getExpr() const {
    return expr;
  }

  void pack(const std::vector<std::vector<int>>& coords,
            internal::ComponentType ctype, const void* values);

  void compile();
  void assemble();
  void evaluate();

  std::shared_ptr<PackedTensor> getPackedTensor() {
    return packedTensor;
  }

  const std::shared_ptr<PackedTensor> getPackedTensor() const {
    return packedTensor;
  }

  friend std::ostream& operator<<(std::ostream& os, const internal::Tensor& t) {
    std::vector<std::string> dimStrings;
    for (int dim : t.getDimensions()) {
      dimStrings.push_back(std::to_string(dim));
    }
    os << t.getName()
       << " (" << util::join(dimStrings, "x") << ", " << t.format << ")";

    // Print packed data
    if (t.getPackedTensor() != nullptr) {
      os << std::endl << *t.getPackedTensor();
    }
    return os;
  }

  std::string                     name;
  std::vector<size_t>             dimensions;
  Format                          format;

  std::shared_ptr<PackedTensor>   packedTensor;

  std::vector<taco::Var>          indexVars;
  taco::Expr                      expr;

  std::shared_ptr<internal::Stmt> code;
};

}}
#endif
