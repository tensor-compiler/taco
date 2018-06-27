#include "taco/index_notation/kernel.h"

#include <iostream>

#include "taco/index_notation/index_notation.h"
#include "taco/storage/storage.h"
#include "taco/lower/lower.h"
#include "taco/codegen/module.h"

using namespace std;

namespace taco {

struct Kernel::Content {
  IndexStmt stmt;
  shared_ptr<ir::Module> module;
};

Kernel::Kernel(IndexStmt stmt, shared_ptr<ir::Module> module, void* evaluate,
               void* assemble, void* compute) : content(new Content) {
  content->stmt = stmt;
  content->module = module;
  this->evaluateFunction = evaluate;
  this->assembleFunction = assemble;
  this->computeFunction = compute;
}

static vector<void*> packArgs(const vector<storage::TensorStorage>& args) {
  vector<void*> arguments;
  arguments.reserve(args.size());
  for (auto& arg : args) {
    arguments.push_back(arg);
  }
  return arguments;
}

bool Kernel::operator()(const std::vector<storage::TensorStorage>& args) const {
  auto arguments = packArgs(args).data();
  int result = content->module->callFuncPacked("evaluate", arguments);
  return (result == 0);
}

std::ostream& operator<<(std::ostream& os, const Kernel& kernel) {
  return os << "// Kernel for: " << kernel.content->stmt << endl << endl
            << kernel.content->module->getSource();
}

Kernel compile(IndexStmt stmt) {
  string reason;
  taco_uassert(isConcreteNotation(stmt, &reason))
      << "Statement not valid concrete index notation and cannot be compiled. "
      << reason << endl << stmt;

  shared_ptr<ir::Module> module(new ir::Module);

  module->addFunction(lower::lower(stmt, "evaluate", true, true));
  module->addFunction(lower::lower(stmt, "assemble", true, false));
  module->addFunction(lower::lower(stmt, "compute",  false, true));
  module->compile();

  void* evaluate = module->getFuncPtr("evaluate");
  void* assemble = module->getFuncPtr("assemble");
  void* compute  = module->getFuncPtr("compute");
  return Kernel(stmt, module, evaluate, assemble, compute);
}

}
