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

int CodeGen::countYields(const Function *func) {
  struct CountYields : public IRVisitor {
    int yields = 0;

    using IRVisitor::visit;

    void visit(const Yield* op) {
      yields++;
    }
  };

  CountYields counter;
  Stmt(func).accept(&counter);
  return counter.yields;
}

// Check if a function has an Allocate node.
// Used to decide if we should print the repack code
class CheckForAlloc : public IRVisitor {
public:
  bool hasAlloc;
  CheckForAlloc() : hasAlloc(false) { }
protected:
  using IRVisitor::visit;
  void visit(const Allocate *op) {
    hasAlloc = true;
  }
};

bool CodeGen::checkForAlloc(const Function *func) {
  CheckForAlloc checker;
  func->accept(&checker);
  return checker.hasAlloc;
}

// helper to translate from taco type to C type
string CodeGen::toCType(Datatype type, bool is_ptr) {
  stringstream ret;
  ret << type;

  if (is_ptr) {
    ret << "*";
  }
  return ret.str();
}


}}
