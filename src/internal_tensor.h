#ifndef TACO_INTERNAL_TENSOR_H
#define TACO_INTERNAL_TENSOR_H

#include <memory>
#include <string>
#include <vector>

#include "format.h"
#include "component_types.h"
#include "util/strings.h"

namespace taco {
class Var;
class Expr;
class PackedTensor;

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

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream& os, const internal::Tensor& t);

}}
#endif
