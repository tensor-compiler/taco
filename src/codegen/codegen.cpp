#include "codegen.h"
#include "taco/cuda.h"
#include "codegen_cuda.h"
#include "codegen_c.h"

using namespace std;

namespace taco {
namespace ir {

shared_ptr<CodeGen> CodeGen::init_default(std::ostream &dest, OutputKind outputKind) {
  if (should_use_CUDA_codegen()) {
    return make_shared<CodeGen_CUDA>(dest, outputKind);
  }
  else {
    return make_shared<CodeGen_C>(dest, outputKind);
  }
}

}}
