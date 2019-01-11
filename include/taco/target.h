#ifndef TACO_TARGET_H
#define TACO_TARGET_H

#include "taco/error.h"

namespace taco {

/// This struct represents the machine & OS to generate code for, both for
/// JIT and AOT code generation.
struct Target {
  /// Architectures.  If C99, we generate C code, and if it is a specific
  /// machine arch (e.g. x86 or arm) we use LLVM.
  enum Arch {C99=0, X86} arch;
  
  /// Operating System.  Used when deciding which OS-specific calls to use.
  enum OS {OSUnknown=0, Linux, MacOS, Windows} os;

  std::string compiler_env = "TACO_CC";

  std::string compiler = "cc";
  
  // As we support them, we'll stick in optional features into the target as
  // well, including things like parallelism model (e.g. openmp, cilk) for
  // C code generation, and hardware features (e.g. AVX) for LLVM code gen.
  
  /// Given a string of the form arch-os-features, construct the corresponding
  /// Target object.
  Target(const std::string &s);

  Target(Arch a, OS o) : arch(a), os(o) { 
    taco_tassert(a == C99 && o != Windows && o != OSUnknown)
        << "Unsupported target.";
  }
  
  /// Validate a target string
  static bool validateTargetString(const std::string &s);
  
};

  /// Gets the target from the environment.  If this is not set in the
  /// environment, it uses the default C99 backend with the current OS
  Target getTargetFromEnvironment();

} // namespace taco

#endif
