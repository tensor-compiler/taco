#ifndef TACO_BACKEND_LLVM_H
#define TACO_BACKEND_LLVM_H

#include <llvm/Support/raw_ostream.h>

#include "taco/ir/ir_visitor.h"

namespace taco {
namespace ir {

class Codegen_LLVM : IRVisitorStrict {
// public:
    // Codegen_LLVM(const Target &target, llvm::LLVMContext &context);
};

}
}

#endif