#ifndef TACO_INTERNAL_TENSOR_H
#define TACO_INTERNAL_TENSOR_H

#include <memory>
#include <string>
#include <vector>

#include "ir.h"
#include "ir_printer.h"
#include "format.h"
#include "iteration_schedule.h"
#include "component_types.h"
#include "packed_tensor.h"
#include "util/intrusive_ptr.h"
#include "util/strings.h"

namespace taco {
struct Var;
struct Expr;

namespace internal {

class Tensor {
public:
  Tensor(std::string name, std::vector<size_t> dimensions, Format format);

  std::string getName() const;
  size_t getOrder() const;
  const std::vector<size_t>& getDimensions() const;
  const Format& getFormat() const;
  const std::vector<taco::Var>& getIndexVars() const;
  const taco::Expr& getExpr() const;
  const std::shared_ptr<PackedTensor> getPackedTensor() const;

  void pack(const std::vector<std::vector<int>>& coords,
            internal::ComponentType ctype, const void* values);

  void compile();
  void assemble();
  void evaluate();

  void setExpr(taco::Expr expr);
  void setIndexVars(std::vector<taco::Var> indexVars);

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
  struct Content;
  std::shared_ptr<Content> content;
};

}}
#endif
