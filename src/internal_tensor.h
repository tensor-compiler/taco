#ifndef TACO_INTERNAL_TENSOR_H
#define TACO_INTERNAL_TENSOR_H

#include <memory>
#include <string>
#include <vector>

#include "expr.h"
#include "ir.h"
#include "ir_printer.h"
#include "format.h"
#include "iteration_schedule.h"
#include "component_types.h"
#include "packed_tensor.h"
#include "util/intrusive_ptr.h"
#include "util/strings.h"

namespace taco {
struct Read;
template <typename T> class Tensor;

namespace internal {

struct TensorContent {
  std::string                     name;
  std::vector<size_t>             dimensions;
  Format                          format;

  std::shared_ptr<PackedTensor>   packedTensor;

  std::vector<taco::Var>          indexVars;
  taco::Expr                      expr;

  IterationSchedule               schedule;
  std::shared_ptr<internal::Stmt> code;
};

class Tensor : public util::Manageable<Tensor> {
  friend class  taco::Tensor<double>;

public:
  Tensor(std::string name, std::vector<size_t> dimensions, Format format)
      : content(new TensorContent) {
    content->name = name;
    content->dimensions = dimensions;
    content->format = format;
  }

  void setExpr(taco::Expr expr) {
    content->expr = expr;
  }

  void setIndexVars(std::vector<taco::Var> indexVars) {
    content->indexVars = indexVars;
  }

  std::string getName() const {
    return content->name;
  }

  const Format& getFormat() const {
    return content->format;
  }

  size_t getOrder() const {
    return content->dimensions.size();
  }

  const std::vector<size_t>& getDimensions() const {
    return content->dimensions;
  }

  const std::vector<taco::Var>& getIndexVars() const {
    return content->indexVars;
  }

  taco::Expr getExpr() const {
    return content->expr;
  }

  void pack(const std::vector<std::vector<int>>& coords,
            internal::ComponentType ctype, const void* values);

  void compile();
  void assemble();
  void evaluate();

  std::shared_ptr<PackedTensor> getPackedTensor() {
    return content->packedTensor;
  }

  const std::shared_ptr<PackedTensor> getPackedTensor() const {
    return content->packedTensor;
  }

  friend std::ostream& operator<<(std::ostream& os, const internal::Tensor& t) {
    std::vector<std::string> dimStrings;
    for (int dim : t.getDimensions()) {
      dimStrings.push_back(std::to_string(dim));
    }
    os << t.getName()
       << " (" << util::join(dimStrings, "x") << ", " << t.getFormat() << ")";

    // Print packed data
    if (t.getPackedTensor() != nullptr) {
      os << std::endl << *t.getPackedTensor();
    }
    return os;
  }

private:
  std::shared_ptr<TensorContent> content;
};

}}
#endif
