#include "codegen_llvm.h"

using namespace std;

namespace taco{
namespace ir{


void CodeGen_LLVM::compile(Stmt stmt, bool isFirst) {
  throw logic_error("Not Implemented.");
  stmt.accept(this);
}

} // namespace ir
} // namespace taco