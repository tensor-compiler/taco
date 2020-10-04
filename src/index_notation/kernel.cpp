#include "taco/index_notation/kernel.h"

#include <iostream>

#include "taco/index_notation/index_notation.h"
#include "taco/lower/lower.h"
#include "taco/codegen/module.h"
#include "taco/storage/storage.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/taco_tensor_t.h"
#include <taco/index_notation/transformations.h>
#include "taco/index_notation/index_notation_nodes.h"


using namespace std;

namespace taco {

struct Kernel::Content {
  shared_ptr<ir::Module> module;
};

Kernel::Kernel() : content(nullptr) {
  this->numResults = 0;
  this->evaluateFunction = nullptr;
  this->assembleFunction = nullptr;
  this->computeFunction = nullptr;
}

Kernel::Kernel(IndexStmt stmt, shared_ptr<ir::Module> module, void* evaluate,
               void* assemble, void* compute) : content(new Content) {
  content->module = module;
  this->numResults = getResults(stmt).size();
  this->evaluateFunction = evaluate;
  this->assembleFunction = assemble;
  this->computeFunction = compute;
}

static inline
vector<void*> packArguments(const vector<TensorStorage>& args) {
  vector<void*> arguments;
  arguments.reserve(args.size());
  for (auto& arg : args) {
    arguments.push_back(static_cast<taco_tensor_t*>(arg));
  }
  return arguments;
}

static inline
void unpackResults(size_t numResults, const vector<void*> arguments,
                   const vector<TensorStorage>& args) {
  for (size_t i = 0; i < numResults; i++) {
    taco_tensor_t* tensorData = ((taco_tensor_t*)arguments[i]);
    TensorStorage storage = args[i];
    Format format = storage.getFormat();

    vector<ModeIndex> modeIndices;
    size_t num = 1;
    for (int i = 0; i < storage.getOrder(); i++) {
      ModeFormat modeType = format.getModeFormats()[i];
      if (modeType.getName() == Dense.getName()) {
        Array size = makeArray({*(int*)tensorData->indices[i][0]});
        modeIndices.push_back(ModeIndex({size}));
        num *= ((int*)tensorData->indices[i][0])[0];
      } else if (modeType.getName() == Sparse.getName()) {
        auto size = ((int*)tensorData->indices[i][0])[num];
        Array pos = Array(type<int>(), tensorData->indices[i][0],
                          num+1, Array::UserOwns);
        Array idx = Array(type<int>(), tensorData->indices[i][1],
                          size, Array::UserOwns);
        modeIndices.push_back(ModeIndex({pos, idx}));
        num = size;
      } else {
        taco_not_supported_yet;
      }
    }
    storage.setIndex(Index(format, modeIndices));
    storage.setValues(Array(storage.getComponentType(), tensorData->vals, num));
  }
}

bool Kernel::operator()(const vector<TensorStorage>& args) const {
  vector<void*> arguments = packArguments(args);
  int result = content->module->callFuncPacked("evaluate", arguments.data());
  unpackResults(this->numResults, arguments, args);
  return (result == 0);
}

bool Kernel::assemble(const vector<TensorStorage>& args) const {
  vector<void*> arguments = packArguments(args);
  int result = content->module->callFuncPacked("assemble", arguments.data());
  unpackResults(this->numResults, arguments, args);
  return (result == 0);
}

bool Kernel::compute(const vector<TensorStorage>& args) const {
  vector<void*> arguments = packArguments(args);
  int result = content->module->callFuncPacked("compute", arguments.data());
  return (result == 0);
}

bool Kernel::defined() {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Kernel& kernel) {
  return os << kernel.content->module->getSource();
}

Kernel compile(IndexStmt stmt) {
  string reason;
  taco_uassert(isConcreteNotation(stmt, &reason))
      << "Statement not valid concrete index notation and cannot be compiled. "
      << reason << endl << stmt;

  shared_ptr<ir::Module> module(new ir::Module);
  IndexStmt parallelStmt = parallelizeOuterLoop(stmt);
  module->addFunction(lower(parallelStmt, "compute",  false, true));
  module->addFunction(lower(stmt, "assemble", true, false));
  module->addFunction(lower(stmt, "evaluate", true, true));
  module->compile();

  void* evaluate = module->getFuncPtr("evaluate");
  void* assemble = module->getFuncPtr("assemble");
  void* compute  = module->getFuncPtr("compute");
  return Kernel(stmt, module, evaluate, assemble, compute);
}

}
