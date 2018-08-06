#ifndef TACO_KERNEL_H
#define TACO_KERNEL_H

#include <vector>
#include <memory>

namespace taco {

class Function;
class IndexStmt;
class TensorStorage;
namespace ir {
class Module;
}

/// A tensor compute kernel is a runnable object that executes a concrete index
/// notation statement.  Kernels allocate memory, assemble indices, and compute
/// component values of the result tensors in the concrete index statement.
/// They can be called to do all these things at once (`evaluate`), to only
/// allocate memory and assemble indices (`assemble`), or to only compute
/// component values (`compute`).
class Kernel {
public:
  /// Construct an undefined kernel.
  Kernel();

  /// Construct a kernel from relevant function pointers and a module.
  Kernel(IndexStmt stmt, std::shared_ptr<ir::Module> module,
         void* evaluate, void* assemble, void* compute);

  /// Evaluate the kernel on the given tensor storage arguments, which includes
  /// allocating memory, assembling indices, and computing component values.
  /// @{
  bool operator()(const std::vector<TensorStorage>& args) const;
  template <typename... Args> bool operator()(const Args&... args) const {
    return operator()({args...});
  }
  /// @}

  /// Execute the kernel to assemble the indices of the results.
  /// @{
  bool assemble(const std::vector<TensorStorage>& args) const;
  template <typename... Args> bool assemble(const Args&... args) const {
    return assemble({args...});
  }
  /// @}

  /// Execute the kernel to compute the component values of the results, but
  /// do not allocate result memory or assemble result indices.
  /// @{
  bool compute(const std::vector<TensorStorage>& args) const;
  template <typename... Args> bool compute(const Args&... args) const {
    return compute({args...});
  }
  /// @}

  /// Check whether the kernel is defined.
  bool defined();

  /// Print the tensor compute kernel.
  friend std::ostream& operator<<(std::ostream&, const Kernel&);

private:
  struct Content;
  std::shared_ptr<Content> content;
  size_t numResults;
  void* evaluateFunction;
  void* assembleFunction;
  void* computeFunction;
};

/// Compile a concrete index notation statement to a runnable kernel.
Kernel compile(IndexStmt stmt);

}
#endif
