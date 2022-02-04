#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <algorithm>
#include <unordered_set>

#include "taco/ir/ir_visitor.h"
#include "taco/ir/ir_rewriter.h"
#include "codegen_cuda.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"
#include "taco/ir/simplify.h"

#define GEN_TIMING_CODE false

using namespace std;

namespace taco {
  namespace ir {

// Some helper functions
    namespace {

// Include stdio.h for printf
// stdlib.h for malloc/realloc
// math.h for sqrt
// MIN preprocessor macro
// This *must* be kept in sync with taco_tensor_t.h
const string cHeaders =
  "#ifndef TACO_C_HEADERS\n"
  "#define TACO_C_HEADERS\n"
  "#include <stdio.h>\n"
  "#include <stdlib.h>\n"
  "#include <stdint.h>\n"
  "#include <math.h>\n"
  "#include <thrust/complex.h>\n"
  "#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))\n"
  "#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))\n"
  "#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)\n"
  "#ifndef TACO_TENSOR_T_DEFINED\n"
  "#define TACO_TENSOR_T_DEFINED\n"
  "typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;\n"
  "typedef struct {\n"
  "  int32_t      order;         // tensor order (number of modes)\n"
  "  int32_t*     dimensions;    // tensor dimensions\n"
  "  int32_t      csize;         // component size\n"
  "  int32_t*     mode_ordering; // mode storage ordering\n"
  "  taco_mode_t* mode_types;    // mode storage types\n"
  "  uint8_t***   indices;       // tensor index data (per mode)\n"
  "  uint8_t*     vals;          // tensor values\n"
  "  uint8_t*     fill_value;    // tensor fill value\n"
  "  int32_t      vals_size;     // values array size\n"
  "} taco_tensor_t;\n"
  "#endif\n"
  "#endif\n\n"; // // https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

const string gpuAssertMacro =
  "#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }\n"
  "inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)\n"
  "{\n"
  "  if (code != cudaSuccess)\n"
  "  {\n"
  "    fprintf(stderr,\"GPUassert: %s %s %d\\n\", cudaGetErrorString(code), file, line);\n"
  "    if (abort) exit(code);\n"
  "  }\n"
  "}\n"
  "__device__ __host__ int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {\n"
  "  if (array[arrayStart] >= target) {\n"
  "    return arrayStart;\n"
  "  }\n"
  "  int lowerBound = arrayStart; // always < target\n"
  "  int upperBound = arrayEnd; // always >= target\n"
  "  while (upperBound - lowerBound > 1) {\n"
  "    int mid = (upperBound + lowerBound) / 2;\n"
  "    int midValue = array[mid];\n"
  "    if (midValue < target) {\n"
  "      lowerBound = mid;\n"
  "    }\n"
  "    else if (midValue > target) {\n"
  "      upperBound = mid;\n"
  "    }\n"
  "    else {\n"
  "      return mid;\n"
  "    }\n"
  "  }\n"
  "  return upperBound;\n"
  "}\n"
  "__device__ __host__ int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {\n"
  "  if (array[arrayEnd] <= target) {\n"
  "    return arrayEnd;\n"
  "  }\n"
  "  int lowerBound = arrayStart; // always <= target\n"
  "  int upperBound = arrayEnd; // always > target\n"
  "  while (upperBound - lowerBound > 1) {\n"
  "    int mid = (upperBound + lowerBound) / 2;\n"
  "    int midValue = array[mid];\n"
  "    if (midValue < target) {\n"
  "      lowerBound = mid;\n"
  "    }\n"
  "    else if (midValue > target) {\n"
  "      upperBound = mid;\n"
  "    }\n"
  "    else {\n"
  "      return mid;\n"
  "    }\n"
  "  }\n"
  "  return lowerBound;\n"
  "}\n"
  "__global__ void taco_binarySearchBeforeBlock(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int values_per_block, int num_blocks) {\n"
  "  int thread = threadIdx.x;\n"
  "  int block = blockIdx.x;\n"
  "  int idx = block * blockDim.x + thread;\n"
  "  if (idx >= num_blocks+1) {\n"
  "    return;\n"
  "  }\n"
  "\n"
  "  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, idx * values_per_block);\n"
  "}\n"
  "\n"
  "__host__ int * taco_binarySearchBeforeBlockLaunch(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int values_per_block, int block_size, int num_blocks){\n"
  "  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;\n"
  "  taco_binarySearchBeforeBlock<<<num_search_blocks, block_size>>>(array, results, arrayStart, arrayEnd, values_per_block, num_blocks);\n"
  "  return results;\n"
  "}\n"
  "__global__ void taco_binarySearchIndirectBeforeBlock(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int * __restrict__ targets, int num_blocks) {\n"
  "  int thread = threadIdx.x;\n"
  "  int block = blockIdx.x;\n"
  "  int idx = block * blockDim.x + thread;\n"
  "  if (idx >= num_blocks+1) {\n"
  "    return;\n"
  "  }\n"
  "\n"
  "  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, targets[idx]);\n"
  "}\n"
  "\n"
  "__host__ int * taco_binarySearchIndirectBeforeBlockLaunch(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int * __restrict__ targets, int block_size, int num_blocks){\n"
  "  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;\n"
  "  taco_binarySearchIndirectBeforeBlock<<<num_search_blocks, block_size>>>(array, results, arrayStart, arrayEnd, targets, num_blocks);\n"
  "  return results;\n"
  "}\n"
  "template<typename T>\n"
  "__device__ inline void atomicAddWarp(T *array, int index, T val)\n"
  "{\n"
  "  int leader_index = __shfl_sync(-1, index, 0);\n"
  "  int mask = __ballot_sync(-1, leader_index == index);\n"
  "  if(mask == -1) {\n"
  "    val += __shfl_down_sync(-1, val, 16);\n"
  "    val += __shfl_down_sync(-1, val, 8);\n"
  "    val += __shfl_down_sync(-1, val, 4);\n"
  "    val += __shfl_down_sync(-1, val, 2);\n"
  "    val += __shfl_down_sync(-1, val, 1);\n"
  "    if(threadIdx.x % 32 == 0) {\n"
  "      atomicAdd(&array[index], val);\n"
  "    }\n"
  "  } else {\n"
  "    atomicAdd(&array[index], val);\n"
  "  }\n"
  "}\n";

const std::string blue="\033[38;5;67m";
const std::string nc="\033[0m";
} // anonymous namespace

// find variables for generating declarations
// also only generates a single var for each GetProperty
class CodeGen_CUDA::FindVars : public IRVisitor {
public:
  map<Expr, string, ExprCompare> varMap;

  // the variables for which we need to add declarations
  map<Expr, string, ExprCompare> varDecls;

  vector<Expr> localVars;

  // this maps from tensor, property, mode, index to the unique var
  map<tuple<Expr, TensorProperty, int, int>, string> canonicalPropertyVar;

  // this is for convenience, recording just the properties unpacked
  // from the output tensor so we can re-save them at the end
  map<tuple<Expr, TensorProperty, int, int>, string> outputProperties;

  // TODO: should replace this with an unordered set
  vector<Expr> outputTensors;

  // Stop searching for variables at device functions (used to generate kernel launches)
  bool stopAtDeviceFunction;

  bool inBlock;

  CodeGen_CUDA *codeGen;

  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs, CodeGen_CUDA *codeGen,
           bool stopAtDeviceFunction=false)
  : codeGen(codeGen) {
    for (auto v: inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(varMap.count(var) == 0) <<
                                           "Duplicate input found in codegen: " << var->name;
      varMap[var] = var->name;
    }
    for (auto v: outputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(varMap.count(var) == 0) <<
                                           "Duplicate output found in codegen";

      outputTensors.push_back(v);
      varMap[var] = var->name;
    }
    FindVars::stopAtDeviceFunction = stopAtDeviceFunction;
    inBlock = false;
  }

protected:
  using IRVisitor::visit;

  virtual void visit(const For *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    if (op->parallel_unit == ParallelUnit::GPUThread && stopAtDeviceFunction) {
      // Want to collect the start, end, increment for the thread loop, but no other variables
      taco_iassert(inBlock);
      inBlock = false;
    }
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    if (op->parallel_unit == ParallelUnit::GPUBlock && stopAtDeviceFunction) {
      inBlock = true;
    }
    if (op->parallel_unit == ParallelUnit::GPUThread && stopAtDeviceFunction) {
      return;
    }
    op->contents.accept(this);
  }

  virtual void visit(const Var *op) {
    if (varMap.count(op) == 0 && !inBlock) {
      varMap[op] = op->is_ptr? op->name : codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl *op) {
    if (!util::contains(localVars, op->var) && !inBlock) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const GetProperty *op) {
    if (varMap.count(op) == 0 && !inBlock) {
      auto key =
              tuple<Expr,TensorProperty,int,int>(op->tensor,op->property,
                                                 (size_t)op->mode,
                                                 (size_t)op->index);
      if (canonicalPropertyVar.count(key) > 0) {
        varMap[op] = canonicalPropertyVar[key];
      } else {
        auto unique_name = codeGen->genUniqueName(op->name);
        canonicalPropertyVar[key] = unique_name;
        varMap[op] = unique_name;
        varDecls[op] = unique_name;
        if (util::contains(outputTensors, op->tensor)) {
          outputProperties[key] = unique_name;
        }
      }
    }
  }
};

// Finds all for loops tagged with accelerator and adds statements to deviceFunctions
// Also tracks scope of when device function is called and
// tracks which variables must be passed to function.
class CodeGen_CUDA::DeviceFunctionCollector : public IRVisitor {
public:
  vector<Stmt> blockFors;
  vector<Stmt> threadFors; // contents is device function
  vector<Stmt> warpFors;
  map<Expr, string, ExprCompare> scopeMap;

  // the variables to pass to each device function
  vector<vector<pair<string, Expr>>> functionParameters;
  vector<pair<string, Expr>> currentParameters; // keep as vector so code generation is deterministic
  set<Expr> currentParameterSet;

  set<Expr> variablesDeclaredInKernel;

  vector<pair<string, Expr>> threadIDVars;
  vector<pair<string, Expr>> blockIDVars;
  vector<pair<string, Expr>> warpIDVars;
  vector<Expr> numThreads;
  vector<Expr> numWarps;

  CodeGen_CUDA *codeGen;
  // copy inputs and outputs into the map
  DeviceFunctionCollector(vector<Expr> inputs, vector<Expr> outputs, CodeGen_CUDA *codeGen) : codeGen(codeGen)  {
    inDeviceFunction = false;
    for (auto v: inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(scopeMap.count(var) == 0) <<
                                             "Duplicate input found in codegen";
      scopeMap[var] = var->name;
    }
    for (auto v: outputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(scopeMap.count(var) == 0) <<
                                             "Duplicate output found in codegen";

      scopeMap[var] = var->name;
    }
  }

protected:
  bool inDeviceFunction;
  using IRVisitor::visit;

  virtual void visit(const For *op) {
    // Don't need to find/initialize loop bounds
    if (op->parallel_unit == ParallelUnit::GPUBlock) {
      op->var.accept(this);
      taco_iassert(!inDeviceFunction) << "Nested Device functions not supported";
      blockFors.push_back(op);
      blockIDVars.push_back(pair<string, Expr>(scopeMap[op->var], op->var));
      currentParameters.clear();
      currentParameterSet.clear();
      variablesDeclaredInKernel.clear();
      inDeviceFunction = true;
    }
    else if (op->parallel_unit == ParallelUnit::GPUWarp) {
      taco_iassert(inDeviceFunction) << "Nested Device functions not supported";
      taco_iassert(blockIDVars.size() == warpIDVars.size() + 1) << "No matching GPUBlock parallelize for GPUWarp";
      inDeviceFunction = false;
      op->var.accept(this);
      inDeviceFunction = true;

      warpFors.push_back(op);
      warpIDVars.push_back(pair<string, Expr>(scopeMap[op->var], op->var));
      Expr warpsInBlock = ir::simplify(ir::Div::make(ir::Sub::make(op->end, op->start), op->increment));
      numWarps.push_back(warpsInBlock);
    }
    else if (op->parallel_unit == ParallelUnit::GPUThread) {
      taco_iassert(inDeviceFunction) << "Nested Device functions not supported";
      taco_iassert(blockIDVars.size() == threadIDVars.size() + 1) << "No matching GPUBlock parallelize for GPUThread";
      if (blockIDVars.size() > warpIDVars.size()) {
        warpFors.push_back(Stmt());
        warpIDVars.push_back({});
        numWarps.push_back(0);
      }
      inDeviceFunction = false;
      op->var.accept(this);
      inDeviceFunction = true;

      threadFors.push_back(op);
      threadIDVars.push_back(pair<string, Expr>(scopeMap[op->var], op->var));
      Expr blockSize = ir::simplify(ir::Div::make(ir::Sub::make(op->end, op->start), op->increment));
      numThreads.push_back(blockSize);
    }
    else{
      op->var.accept(this);
    }
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
    if (op->parallel_unit == ParallelUnit::GPUBlock) {
      taco_iassert(blockIDVars.size() == threadIDVars.size()) << "No matching GPUThread parallelize for GPUBlock";
      inDeviceFunction = false;
      sort(currentParameters.begin(), currentParameters.end());
      functionParameters.push_back(currentParameters);
    }
  }

  virtual void visit(const Var *op) {
    if (scopeMap.count(op) == 0) {
      string name = codeGen->genUniqueName(op->name);
      if (!inDeviceFunction) {
        scopeMap[op] = name;
      }
    }
    else if (scopeMap.count(op) == 1 && inDeviceFunction && currentParameterSet.count(op) == 0
            && (threadIDVars.empty() || op != threadIDVars.back().second)
            && (blockIDVars.empty() || op != blockIDVars.back().second)
            && (warpIDVars.empty() || op != warpIDVars.back().second)
            && !variablesDeclaredInKernel.count(op)) {
      currentParameters.push_back(pair<string, Expr>(scopeMap[op], op));
      currentParameterSet.insert(op);
    }
  }

  virtual void visit(const VarDecl *op) {
    if (inDeviceFunction) {
      variablesDeclaredInKernel.insert(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const GetProperty *op) {
    if (scopeMap.count(op->tensor) == 0 && !inDeviceFunction) {
      auto key =
              tuple<Expr,TensorProperty,int,int>(op->tensor,op->property,
                                                 (size_t)op->mode,
                                                 (size_t)op->index);
      auto unique_name = codeGen->genUniqueName(op->name);
      scopeMap[op->tensor] = unique_name;
    }
    else if (scopeMap.count(op->tensor) == 1 && inDeviceFunction && currentParameterSet.count(op->tensor) == 0) {
      currentParameters.push_back(pair<string, Expr>(op->tensor.as<Var>()->name, op->tensor));
      currentParameterSet.insert(op->tensor);
    }
  }
};

Stmt CodeGen_CUDA::simplifyFunctionBodies(Stmt stmt) {
  struct FunctionBodySimplifier : IRRewriter {
    using IRRewriter::visit;

    void visit(const Function* func) {
      int numYields = countYields(func); // temporary fix as simplifying function with yields will break printContextDeclAndInit
      if (numYields == 0) {
        Stmt body = ir::simplify(func->body);
        stmt = Function::make(func->name, func->outputs, func->inputs, body);
      }
      else {
        stmt = func;
      }
    }
  };
  return FunctionBodySimplifier().rewrite(stmt);
}

string CodeGen_CUDA::printDeviceFuncName(const vector<pair<string, Expr>> currentParameters, int index) {
  stringstream ret;
  ret << "__global__" << endl;
  ret << "void " << funcName << "DeviceKernel" << index << "(";

  string delimiter = "";
  for (size_t i=0; i<currentParameters.size(); i++) {
    auto var = currentParameters[i].second.as<Var>();
    taco_iassert(var) << "Unable to convert output " << currentParameters[i].second
                      << " to Var";
    string varName = currentParameters[i].first;

    if (var->is_tensor) {
      ret << delimiter << "taco_tensor_t * __restrict__ " << varName;
    }
    else {
      auto tp = printCUDAType(var->type, var->is_ptr);
      ret << delimiter << tp << " ";
      if (!var->is_ptr) {
        ret << "&";
      }
      ret << var->name;
    }
    // No non-tensor parameters
    delimiter = ", ";
  }
  ret << ")";
  return ret.str();
}

void CodeGen_CUDA::printThreadIDVariable(pair<string, Expr> threadIDVar, Expr start, Expr increment, Expr numThreads) {
  auto var = threadIDVar.second.as<Var>();
  taco_iassert(var) << "Unable to convert output " << threadIDVar.second
                    << " to Var";
  string varName = threadIDVar.first;
  auto tp = printCUDAType(var->type, var->is_ptr);
  stream << tp << " " << varName << " = ";
  increment = ir::simplify(increment);
  if (!isa<Literal>(increment) || !to<Literal>(increment)->equalsScalar(1)) {
    stream << "(threadIdx.x";
    stream << " % (";
    numThreads.accept(this);
    stream << ")) * ";
    increment.accept(this);
  }
  else {
    stream << "(threadIdx.x";
    stream << " % (";
    numThreads.accept(this);
    stream << "))";
  }
  Expr expr = ir::simplify(start);
  if (!isa<Literal>(expr) || !to<Literal>(expr)->equalsScalar(0)) {
    stream << " + ";
    expr.accept(this);
  }
  stream << ";\n";
}

void CodeGen_CUDA::printWarpIDVariable(pair<string, Expr> warpIDVar, Expr start, Expr increment, Expr warpSize) {
  auto var = warpIDVar.second.as<Var>();
  taco_iassert(var) << "Unable to convert output " << warpIDVar.second
                    << " to Var";
  string varName = warpIDVar.first;
  auto tp = printCUDAType(var->type, var->is_ptr);
  stream << tp << " " << varName << " = ";
  increment = ir::simplify(increment);
  if (!isa<Literal>(increment) || !to<Literal>(increment)->equalsScalar(1)) {
    stream << "(threadIdx.x / ";
    stream << warpSize << ") * ";
    increment.accept(this);
  }
  else {
    stream << "(threadIdx.x / ";
    stream << warpSize;
    stream << ")";
  }
  Expr expr = ir::simplify(start);
  if (!isa<Literal>(expr) || !to<Literal>(expr)->equalsScalar(0)) {
    stream << " + ";
    expr.accept(this);
  }
  stream << ";\n";
}

void CodeGen_CUDA::printBlockIDVariable(pair<string, Expr> blockIDVar, Expr start, Expr increment) {
  auto var = blockIDVar.second.as<Var>();
  taco_iassert(var) << "Unable to convert output " << blockIDVar.second
                    << " to Var";
  string varName = blockIDVar.first;
  auto tp = printCUDAType(var->type, var->is_ptr);
  stream << tp << " " << varName << " = ";
  increment = ir::simplify(increment);
  if (!isa<Literal>(increment) || !to<Literal>(increment)->equalsScalar(1)) {
    stream << "blockIdx.x * ";
    increment.accept(this);
  }
  else {
    stream << "blockIdx.x";
  }
  Expr expr = ir::simplify(start);
  if (!isa<Literal>(expr) || !to<Literal>(expr)->equalsScalar(0)) {
    stream << " + ";
    expr.accept(this);
  }
  stream << ";\n";
}

void CodeGen_CUDA::printThreadBoundCheck(Expr end) {
  end = ir::simplify(end);
  stream << "if (threadIdx.x >= ";
  end.accept(this);
  stream << ") {" << "\n";
  indent++;
  doIndent();
  stream << "return;\n";
  indent--;
  doIndent();
  stream << "}" << "\n" << "\n";
}

void CodeGen_CUDA::printDeviceFuncCall(const vector<pair<string, Expr>> currentParameters, Expr blockSize, int index, Expr gridSize) {
  if (GEN_TIMING_CODE && !emittedTimerStartCode) {
    doIndent();
    stream << "float tot_ms;" << endl;
    doIndent();
    stream << "cudaEvent_t event1, event2;" << endl;
    doIndent();
    stream << "cudaEventCreate(&event1);" << endl;
    doIndent();
    stream << "cudaEventCreate(&event2);" << endl;
    doIndent();
    stream << "cudaDeviceSynchronize();" << endl;
    doIndent();
    stream << "cudaEventRecord(event1,0);" << endl;
    emittedTimerStartCode = true;
  }


  stream << funcName << "DeviceKernel" << index << "<<<";
  gridSize = ir::simplify(gridSize);
  gridSize.accept(this);
  stream << ", ";
  blockSize.accept(this);
  stream << ">>>";
  stream << "(";

  string delimiter = "";
  for (size_t i=0; i<currentParameters.size(); i++) {
    taco_iassert(currentParameters[i].second.as<Var>()) << "Unable to convert output " << currentParameters[i].second
                                                        << " to Var";
    string varName = currentParameters[i].first;
    stream << delimiter << varName;

    delimiter = ", ";
  }
  stream << ");\n";

  if (GEN_TIMING_CODE) {
    doIndent();
    stream << "cudaEventRecord(event2,0);\n";
    doIndent();
    stream << "cudaEventSynchronize(event1);\n";
    doIndent();
    stream << "cudaEventSynchronize(event2);\n";
    doIndent();
    stream << "cudaEventElapsedTime(&tot_ms, event1, event2);\n";
  }
  doIndent();
  stream << "cudaDeviceSynchronize();\n";

}


CodeGen_CUDA::CodeGen_CUDA(std::ostream &dest, OutputKind outputKind)
      : CodeGen(dest, false, false, CUDA), out(dest), outputKind(outputKind) {}

CodeGen_CUDA::~CodeGen_CUDA() {}

void CodeGen_CUDA::compile(Stmt stmt, bool isFirst) {
  deviceFunctionParameters = {};
  varMap = {};
  localVars = {};
  deviceFunctionBlockSizes = {};
  deviceFunctionGridSizes = {};
  deviceFunctions = {};
  scalarVarsPassedToDeviceFunction = {};
  deviceFunctionLoopDepth = 0;
  parentParallelUnits = {};
  parallelUnitSizes = {};
  parallelUnitIDVars = {};
  emittedTimerStartCode = false;
  isHostFunction = true;
  if (isFirst) {
    // output the headers
    out << cHeaders;
    if (outputKind == ImplementationGen) {
      out << endl << gpuAssertMacro;
    }
  }
  out << endl;
  // simplify all function bodies before so can find device functions
  stmt = simplifyFunctionBodies(stmt);
  stmt.accept(this);
}

void CodeGen_CUDA::printDeviceFunctions(const Function* func) {
  // Collect device functions
  resetUniqueNameCounters();
  deviceFunctionLoopDepth = 0;
  DeviceFunctionCollector deviceFunctionCollector(func->inputs, func->outputs, this);
  func->body.accept(&deviceFunctionCollector);
  deviceFunctions = deviceFunctionCollector.blockFors;
  deviceFunctionParameters = deviceFunctionCollector.functionParameters;
  for (int i = 0; i < (int) deviceFunctionCollector.numThreads.size(); i++) {
    Expr blockSize = deviceFunctionCollector.numThreads[i];
    if (deviceFunctionCollector.warpFors[i].defined()) {
      blockSize = Mul::make(blockSize, deviceFunctionCollector.numWarps[i]);
    }
    deviceFunctionBlockSizes.push_back(blockSize);

    const For *blockloop = to<For>(deviceFunctions[i]);
    Expr gridSize = Div::make(Add::make(Sub::make(blockloop->end, blockloop->start), Sub::make(blockloop->increment, Literal::make(1, Int()))), blockloop->increment);
    deviceFunctionGridSizes.push_back(gridSize);
  }

  resetUniqueNameCounters();
  for (size_t i = 0; i < deviceFunctions.size(); i++) {
    const For *blockloop = to<For>(deviceFunctions[i]);
    taco_iassert(blockloop->parallel_unit == ParallelUnit::GPUBlock);
    const For *threadloop = to<For>(deviceFunctionCollector.threadFors[i]);
    taco_iassert(threadloop->parallel_unit == ParallelUnit::GPUThread);
    Stmt function = blockloop->contents;
    vector<pair<string, Expr>> parameters = deviceFunctionParameters[i];

    // add scalar parameters to set
    for (auto parameter : parameters) {
      auto var = parameter.second.as<Var>();
      if (!var->is_tensor && !var->is_ptr) {
        scalarVarsPassedToDeviceFunction.insert(parameter.second);
      }
    }

    // Generate device function header
    doIndent();
    out << printDeviceFuncName(parameters, i);
    out << "{\n";
    indent++;

    // Generate device function code
    resetUniqueNameCounters();
    vector<Expr> inputs;
    for (size_t i = 0; i < parameters.size(); i++) {
      inputs.push_back(parameters[i].second);
    }

    parallelUnitIDVars = {{ParallelUnit::GPUBlock, deviceFunctionCollector.blockIDVars[i].second},
                          {ParallelUnit::GPUThread, deviceFunctionCollector.threadIDVars[i].second}};

    parallelUnitSizes = {{ParallelUnit::GPUBlock, deviceFunctionBlockSizes[i]}};

    if (deviceFunctionCollector.warpFors[i].defined()) {
      parallelUnitIDVars[ParallelUnit::GPUWarp] = deviceFunctionCollector.warpIDVars[i].second;
      parallelUnitSizes[ParallelUnit::GPUWarp] = deviceFunctionCollector.numThreads[i];
    }

    for (auto idVar : parallelUnitIDVars) {
      inputs.push_back(idVar.second);
    }

    FindVars varFinder(inputs, {}, this);
    blockloop->accept(&varFinder);
    varMap = varFinder.varMap;

    // Print variable declarations
    out << printDecls(varFinder.varDecls, inputs, {}) << endl;
    doIndent();
    printBlockIDVariable(deviceFunctionCollector.blockIDVars[i], blockloop->start, blockloop->increment);
    doIndent();
    printThreadIDVariable(deviceFunctionCollector.threadIDVars[i], threadloop->start, threadloop->increment, deviceFunctionCollector.numThreads[i]);
    if (deviceFunctionCollector.warpFors[i].defined()) {
      doIndent();
      const For *warploop = to<For>(deviceFunctionCollector.warpFors[i]);
      printWarpIDVariable(deviceFunctionCollector.warpIDVars[i], warploop->start, warploop->increment, deviceFunctionCollector.numThreads[i]);
    }
    doIndent();
    printThreadBoundCheck(deviceFunctionBlockSizes[i]);

    // output body
    print(function);

    // output repack only if we allocated memory
    if (checkForAlloc(func))
      out << endl << printPack(varFinder.outputProperties, func->outputs);
    indent--;
    doIndent();
    out << "}\n\n";
  }
}

void CodeGen_CUDA::visit(const Function* func) {
  funcName = func->name;
  // if generating a header, protect the function declaration with a guard
  if (outputKind == HeaderGen) {
    out << "#ifndef TACO_GENERATED_" << func->name << "\n";
    out << "#define TACO_GENERATED_" << func->name << "\n";
  }
  else {
    emittingCoroutine = false;
    isHostFunction = false;
    printDeviceFunctions(func);
    isHostFunction = true;
  }

  int numYields = countYields(func);
  emittingCoroutine = (numYields > 0);
  labelCount = 0;

  // Generate rest of code + calls to device functions

  // output function declaration
  doIndent();
  out << printFuncName(func);

  // if we're just generating a header, this is all we need to do
  if (outputKind == HeaderGen) {
    out << ";\n";
    out << "#endif\n";
    return;
  }

  out << " {\n";

  indent++;

  // find all the vars that are not inputs or outputs and declare them
  resetUniqueNameCounters();
  FindVars varFinder(func->inputs, func->outputs, this, true);
  func->body.accept(&varFinder);
  varMap = varFinder.varMap;
  localVars = varFinder.localVars;

  // Print variable declarations
  out << printDecls(varFinder.varDecls, func->inputs, func->outputs) << endl;

  if (emittingCoroutine) {
    out << printContextDeclAndInit(varMap, localVars, numYields, func->name)
        << endl;
  }

  // output body
  print(func->body);

  // output repack only if we allocated memory
  if (checkForAlloc(func))
    out << endl << printPack(varFinder.outputProperties, func->outputs);

  if (emittingCoroutine) {
    out << printCoroutineFinish(numYields, funcName);
  }

  doIndent();
  if (GEN_TIMING_CODE && emittedTimerStartCode && func->name.rfind("compute", 0) == 0) {
    out << "return (int) (tot_ms * 100000); // returns 10^-8 seconds (assumes tot_ms < 20 seconds)\n";
  }
  else {
    out << "return 0;\n";
  }
  indent--;

  doIndent();
  out << "}\n";
}

// For Vars, we replace their names with the generated name,
// since we match by reference (not name)
void CodeGen_CUDA::visit(const Var* op) {
  taco_iassert(varMap.count(op) > 0) <<
     "Var " << op->name << " not found in varMap";
  if (emittingCoroutine) {
//    out << "TACO_DEREF(";
  }
  out << varMap[op];
  if (emittingCoroutine) {
//    out << ")";
  }
}

static string genVectorizePragma(int width) {
  stringstream ret;
  ret << "#pragma clang loop interleave(enable) ";
  if (!width)
    ret << "vectorize(enable)";
  else
    ret << "vectorize_width(" << width << ")";

  return ret.str();
}

static string getParallelizePragma(LoopKind kind) {
  stringstream ret;
  ret << "#pragma omp parallel for";
  if (kind == LoopKind::Dynamic) {
    ret << " schedule(dynamic, 16)";
  }
  return ret.str();
}

static string getAtomicPragma() {
  return "#pragma omp atomic";
}

static string getUnrollPragma(size_t unrollFactor) {
  return "#pragma unroll " + std::to_string(unrollFactor);
}

// The next two need to output the correct pragmas depending
// on the loop kind (Serial, Static, Dynamic, Vectorized)
//
// Docs for vectorization pragmas:
// http://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-loop-hint-optimizations
void CodeGen_CUDA::visit(const For* op) {
  if (op->parallel_unit != ParallelUnit::NotParallel) {
    parentParallelUnits.insert(op->parallel_unit);
  }

  if (!isHostFunction && (op->parallel_unit == ParallelUnit::GPUThread || op->parallel_unit == ParallelUnit::GPUWarp)) {
    // Don't emit thread loop
    indent--;
    op->contents.accept(this);
    indent++;
    return;
  }

  // only first thread
  if (!isHostFunction && (op->parallel_unit == ParallelUnit::GPUWarpReduction || op->parallel_unit == ParallelUnit::GPUBlockReduction)) {
    doIndent();
    if (op->parallel_unit == ParallelUnit::GPUWarpReduction) {
      stream << "__syncwarp();" << endl;
    }
    else if (op->parallel_unit == ParallelUnit::GPUBlockReduction) {
      stream << "__syncthreads();" << endl;
    }
    doIndent();
    stream << keywordString("if") << " (";
    op->var.accept(this);
    stream << " == ";
    op->start.accept(this);
    stream << ") {" << endl;
    indent++;
  }

  for (size_t i = 0; i < deviceFunctions.size(); i++) {
    auto dFunction = deviceFunctions[i].as<For>();
    assert(dFunction);
    if (op == dFunction) {
      // Generate kernel launch
      doIndent();
      printDeviceFuncCall(deviceFunctionParameters[i], deviceFunctionBlockSizes[i], i, deviceFunctionGridSizes[i]);
      return;
    }
  }

  switch (op->kind) {
    case LoopKind::Vectorized:
      doIndent();
      out << genVectorizePragma(op->vec_width);
      out << "\n";
      break;
    case LoopKind::Static:
    case LoopKind::Dynamic:
      doIndent();
      out << getParallelizePragma(op->kind);
      out << "\n";
      break;
    default:
      if (op->unrollFactor > 0) {
        doIndent();
        out << getUnrollPragma(op->unrollFactor) << endl;
      }
      break;
  }

  doIndent();
  stream << keywordString("for") << " (";
  if (!emittingCoroutine) {
    stream << keywordString(printCUDAType(op->var.type(), false)) << " ";
  }
  op->var.accept(this);
  stream << " = ";
  op->start.accept(this);
  stream << keywordString("; ");
  op->var.accept(this);
  stream << " < ";
  parentPrecedence = BOTTOM;
  op->end.accept(this);
  stream << keywordString("; ");
  op->var.accept(this);

  auto lit = op->increment.as<Literal>();
  if (lit != nullptr && ((lit->type.isInt()  && lit->equalsScalar(1)) ||
                         (lit->type.isUInt() && lit->equalsScalar(1)))) {
    stream << "++";
  }
  else {
    stream << " += ";
    op->increment.accept(this);
  }
  stream << ") {\n";

  if(!isHostFunction) {
    deviceFunctionLoopDepth++;
  }
  op->contents.accept(this);
  if(!isHostFunction) {
    deviceFunctionLoopDepth--;
  }
  doIndent();
  stream << "}";
  stream << endl;

  if (!isHostFunction && (op->parallel_unit == ParallelUnit::GPUWarpReduction || op->parallel_unit == ParallelUnit::GPUBlockReduction)) {
    indent--;
    doIndent();
    stream << "}" << endl;
  }

  if (op->parallel_unit != ParallelUnit::NotParallel) {
    parentParallelUnits.erase(op->parallel_unit);
  }
}

void CodeGen_CUDA::visit(const While* op) {
  // it's not clear from documentation that clang will vectorize
  // while loops
  // however, we'll output the pragmas anyway
  if (op->kind == LoopKind::Vectorized) {
    doIndent();
    out << genVectorizePragma(op->vec_width);
    out << "\n";
  }

  IRPrinter::visit(op);
}

void CodeGen_CUDA::visit(const GetProperty* op) {
  taco_iassert(varMap.count(op) > 0) <<
                                     "Property of " << op->tensor << " not found in varMap";
  out << varMap[op];
}

void CodeGen_CUDA::visit(const Min* op) {
  if (op->operands.size() == 1) {
    op->operands[0].accept(this);
    return;
  }
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << "TACO_MIN(";
    op->operands[i].accept(this);
    stream << ",";
  }
  op->operands.back().accept(this);
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << ")";
  }
}

void CodeGen_CUDA::visit(const Max* op) {
  if (op->operands.size() == 1) {
    op->operands[0].accept(this);
    return;
  }
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << "TACO_MAX(";
    op->operands[i].accept(this);
    stream << ",";
  }
  op->operands.back().accept(this);
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << ")";
  }
}

void CodeGen_CUDA::visit(const Allocate* op) {
  string elementType = printCUDAType(op->var.type(), false);
  if (!isHostFunction) {
    if (parentParallelUnits.count(ParallelUnit::GPUThread)) {
      // double w_GPUThread[num];
      // for threads allocate thread local memory
      doIndent();
      stream << elementType << " ";
      op->var.accept(this);
      stream << "[";
      op->num_elements.accept(this);
      stream << "];" << endl;
      return;
    }
    // __shared__ double w_GPUThread[32]; if no warps
    // __shared__ double w_GPUThread_ALL[32 * # num warps]; if warps
    // double * w_GPUThread = w_GPUThread_ALL + warp_id * 32;
    taco_iassert(!op->is_realloc);
    doIndent();
    stream << "__shared__ " << elementType << " ";
    op->var.accept(this);
    if (parentParallelUnits.count(ParallelUnit::GPUWarp)) {
      stream << "_ALL";
    }
    stream << "[";
    if (parentParallelUnits.count(ParallelUnit::GPUWarp)) {
      Expr numElements = Mul::make(op->num_elements, Div::make(parallelUnitSizes[ParallelUnit::GPUBlock], parallelUnitSizes[ParallelUnit::GPUWarp]));
      ir::simplify(numElements).accept(this);
    }
    else {
      op->num_elements.accept(this);
    }
    stream << "];" << endl;
    if (parentParallelUnits.count(ParallelUnit::GPUWarp)) {
      doIndent();
      stream << elementType << " * ";
      op->var.accept(this);

      stream << " = ";
      op->var.accept(this);
      stream << "_ALL + ";
      parallelUnitIDVars[ParallelUnit::GPUWarp].accept(this);
      stream << " * ";
      parallelUnitSizes[ParallelUnit::GPUWarp].accept(this);
      stream << ";" << endl;
    }
    return;
  }
  string variable_name;
  if (op->is_realloc) {
    // cuda doesn't have realloc
    // void * tmp_realloc_ptr;
    // gpuErrchk(cudaMallocManaged((void**)& tmp_realloc_ptr, new_size);
    // memcpy(tmp_realloc_ptr, var, TACO_MIN(old_size, new_size));
    // cudaFree(var);
    // var = tmp_realloc_ptr;

    doIndent();
    variable_name = genUniqueName("tmp_realloc_ptr");
    stream << "void * " << variable_name << ";" << endl;
  }

  doIndent();
  stream << "gpuErrchk(cudaMallocManaged((void**)&";
  if (op->is_realloc) {
    stream << variable_name;
  }
  else {
    op->var.accept(this);
  }
  stream << ", ";
  stream << "sizeof(" << elementType << ")";
  stream << " * ";
  parentPrecedence = MUL;
  op->num_elements.accept(this);
  parentPrecedence = TOP;
  stream << "));" << endl;
  // If the operation wants the input cleared, then memset it to zero.
  if (op->clear) {
    doIndent();
    stream << "gpuErrchk(cudaMemset(";
    op->var.accept(this);
    stream << variable_name;
    stream << ", 0, ";
    stream << "sizeof(" << elementType << ")";
    stream << " * ";
    parentPrecedence = MUL;
    op->num_elements.accept(this);
    parentPrecedence = TOP;
    stream << "));" << endl;
  }

  if(op->is_realloc) {
    doIndent();
    stream << "memcpy(" << variable_name << ", ";
    op->var.accept(this);
    stream << ", ";
    stream << "TACO_MIN(";
    op->old_elements.accept(this);
    stream << ", ";
    op->num_elements.accept(this);
    stream << ")";
    stream << " * sizeof(" << elementType << "));" << endl;

    doIndent();
    stream << "cudaFree(";
    op->var.accept(this);
    stream << ");" << endl;

    doIndent();
    op->var.accept(this);
    stream << " = (" << printCUDAType(op->var.type(), true) << ") " << variable_name << ";" << endl;
  }

}

void CodeGen_CUDA::visit(const Free* op) {
  if (!isHostFunction) {
    // Don't need to free shared memory
    return;
  }
  doIndent();
  stream << "cudaFree(";
  parentPrecedence = Precedence::TOP;
  op->var.accept(this);
  stream << ");";
  stream << endl;
}

void CodeGen_CUDA::visit(const Sqrt* op) {
  taco_tassert(op->type.isFloat() && op->type.getNumBits() == 64) <<
                                                                  "Codegen doesn't currently support non-double sqrt";
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}

void CodeGen_CUDA::visit(const Continue*) {
  doIndent();
  if(!isHostFunction && deviceFunctionLoopDepth == 0) {
    // can't break out of kernel
    stream << "return;" << endl;
  }
  else {
    stream << "break;" << endl;
  }
}

void CodeGen_CUDA::visit(const VarDecl* op) {
  if (emittingCoroutine) {
    doIndent();
    op->var.accept(this);
    parentPrecedence = Precedence::TOP;
    stream << " = ";
    op->rhs.accept(this);
    stream << ";";
    stream << endl;
  }
  // f var can be passed to device function then allocated in uvm
  else if (scalarVarsPassedToDeviceFunction.count(op->var) && isHostFunction) {
    // type *x_ptr;
    // gpuErrchk(cudaMallocManaged((void**)&x_ptr, sizeof(type));
    // type &x = *x_ptr;
    // x = rhs;
    doIndent();
    stream << keywordString(printCUDAType(op->var.type(), true)) << " ";
    string varName = varNameGenerator.getUniqueName(util::toString(op->var));
    varNames.insert({op->var, varName});
    op->var.accept(this);
    stream << "_ptr;" << endl;
    parentPrecedence = Precedence ::TOP;

    doIndent();
    stream << "gpuErrchk(cudaMallocManaged((void**)&";
    op->var.accept(this);
    stream << "_ptr, sizeof(" << keywordString(printCUDAType(op->var.type(), false)) << ")));" << endl;

    doIndent();
    stream << keywordString(printCUDAType(op->var.type(), false)) << "& ";
    op->var.accept(this);
    stream << " = *";
    op->var.accept(this);
    stream << "_ptr;" << endl;

    doIndent();
    op->var.accept(this);
    stream << " = ";
    op->rhs.accept(this);
    stream << ";" << endl;
  }
  else {
    bool is_ptr = false;
    if (isa<Var>(op->var)) {
      is_ptr = to<Var>(op->var)->is_ptr;
    }
    doIndent();
    stream << keywordString(printCUDAType(op->var.type(), is_ptr)) << " ";
    string varName = varNameGenerator.getUniqueName(util::toString(op->var));
    varNames.insert({op->var, varName});
    op->var.accept(this);
    parentPrecedence = Precedence::TOP;
    stream << " = ";
    op->rhs.accept(this);
    stream << ";";
    stream << endl;
  }
}

void CodeGen_CUDA::visit(const Yield* op) {
  printYield(op, localVars, varMap, labelCount, funcName);
}

// Need to handle some binary ops so that we can add the necessary casts if complex
// Because c++ does not properly handle double * std::complex<float> or std::complex<float> * std::complex<double>
// Based on IRPrinter::printBinOp
void CodeGen_CUDA::printBinCastedOp(Expr a, Expr b, string op, Precedence precedence) {
  bool parenthesize = needsParentheses(precedence);
  if (parenthesize) {
    stream << "(";
  }
  parentPrecedence = precedence;
  Datatype mType = max_type(a.type(), b.type());
  if (mType.isComplex() && mType != a.type()) {
    stream << "(" << printCUDAType(mType, false) << ") ";
  }
  a.accept(this);
  stream << " " << op << " ";
  parentPrecedence = precedence;
  if (mType.isComplex() && mType != b.type()) {
    stream << "(" << printCUDAType(mType, false) << ") ";
  }
  b.accept(this);
  if (parenthesize) {
    stream << ")";
  }
}

void CodeGen_CUDA::visit(const Add* op) {
  printBinCastedOp(op->a, op->b, "+", Precedence::ADD);
}

void CodeGen_CUDA::visit(const Sub* op) {
  printBinCastedOp(op->a, op->b, "-", Precedence::SUB);
}

void CodeGen_CUDA::visit(const Mul* op) {
  printBinCastedOp(op->a, op->b, "*", Precedence::MUL);
}

void CodeGen_CUDA::visit(const Div* op) {
  printBinCastedOp(op->a, op->b, "/", Precedence::DIV);
}

void CodeGen_CUDA::visit(const Literal* op) {
  if (op->type.isComplex()) {
    if (color) {
      stream << blue ;
    }

    if(op->type.getKind() == Complex64) {
      std::complex<float> val = op->getValue<std::complex<float>>();
      stream << "thrust::complex<float>(" << val.real() << ", " << val.imag() << ")";
    }
    else if(op->type.getKind() == Complex128) {
      std::complex<double> val = op->getValue<std::complex<double>>();
      stream << "thrust::complex<double>(" << val.real() << ", " << val.imag() << ")";
    }
    else {
      taco_ierror << "Undefined type in IR";
    }

    if (color) {
      stream << nc;
    }
  }
  else {
    IRPrinter::visit(op);
  }
}

void CodeGen_CUDA::visit(const Call* op) {
  if (op->func == "cudaMemset") {
    IRPrinter::visit(op);
    return;
  }
  stream << op->func << "(";
  parentPrecedence = Precedence::CALL;

  // Need to print cast to type so that arguments match.
  if (op->args.size() > 0) {
    // However, the binary search arguments take int* as their first
    // argument. This pointer information isn't carried anywhere in
    // the argument expressions, so we need to special case and not
    // emit an invalid cast for that argument.
    auto opIsBinarySearch = op->func == "taco_binarySearchAfter" || op->func == "taco_binarySearchBefore";
    if (!opIsBinarySearch && (op->type != op->args[0].type() || isa<Literal>(op->args[0]))) {
      stream << "(" << printCUDAType(op->type, false) << ") ";
    }
    op->args[0].accept(this);
  }

  for (size_t i=1; i < op->args.size(); ++i) {
    stream << ", ";
    if (op->type != op->args[i].type() || isa<Literal>(op->args[i])) {
      stream << "(" << printCUDAType(op->type, false) << ") ";
    }
    op->args[i].accept(this);
  }

  stream << ")";
}

void CodeGen_CUDA::visit(const Assign* op) {
  if (GEN_TIMING_CODE && !emittedTimerStartCode && isa<ir::Call>(op->rhs)) {
    if (to<ir::Call>(op->rhs)->func == "taco_binarySearchBeforeBlockLaunch") {
      doIndent();
      stream << "float tot_ms;" << endl;
      doIndent();
      stream << "cudaEvent_t event1, event2;" << endl;
      doIndent();
      stream << "cudaEventCreate(&event1);" << endl;
      doIndent();
      stream << "cudaEventCreate(&event2);" << endl;
      doIndent();
      stream << "cudaDeviceSynchronize();" << endl;
      doIndent();
      stream << "cudaEventRecord(event1,0);" << endl;
      emittedTimerStartCode = true;
    }
  }

  if (op->use_atomics) {
    if (isHostFunction) {
      doIndent();
      stream << getAtomicPragma() << endl;
      IRPrinter::visit(op);
    }
    else {
      if (isa<Mul>(op->rhs)) {
        auto mul = to<Mul>(op->rhs);
        taco_iassert(mul->a == op->lhs);
        doIndent();
        // type atomicOldX = rhs;
        string oldValueName = genUniqueName("atomicOld");
        stream << printCUDAType(op->lhs.type(), false);
        stream << " " << oldValueName << " = ";
        op->lhs.accept(this);
        stream << ";";

        doIndent();
        stream << "atomicCAS(&";
        op->lhs.accept(this);
        stream << ", " << oldValueName << ", ";
        stream << oldValueName << " * ";
        mul->b.accept(this);
        stream << ");" << endl;
      } else if (isa<Add>(op->rhs)) {
        auto add = to<Add>(op->rhs);
        taco_iassert(add->a == op->lhs);
        doIndent();
        stream << "atomicAdd(&";
        op->lhs.accept(this);
        stream << ", ";
        add->b.accept(this);
        stream << ");" << endl;
      } else if (isa<BitOr>(op->rhs)) {
        auto bitOr = to<BitOr>(op->rhs);
        taco_iassert(bitOr->a == op->lhs);
        doIndent();
        stream << "atomicOr(&";
        op->lhs.accept(this);
        stream << ", ";
        bitOr->b.accept(this);
        stream << ");" << endl;
      } else {
        taco_ierror;
      }
    }
  }
  else {
    IRPrinter::visit(op);
  }
}

void CodeGen_CUDA::visit(const Store* op) {
  if (op->use_atomics) {
    if (isHostFunction) {
      doIndent();
      stream << getAtomicPragma() << endl;
      IRPrinter::visit(op);
    }
    else {
      if (isa<Mul>(op->data)) {
        auto mul = to<Mul>(op->data);
        taco_iassert(isa<Load>(mul->a));
        auto load = to<Load>(mul->a);
        taco_iassert(load->arr == op->arr && load->loc == op->loc);
        doIndent();
        // type atomicOldX = rhs;
        string oldValueName = genUniqueName("atomicOld");
        stream << printCUDAType(load->type, false);
        stream << " " << oldValueName << " = ";
        op->arr.accept(this);
        stream << "[";
        parentPrecedence = Precedence::TOP;
        op->loc.accept(this);
        stream << "];";

        doIndent();
        stream << "atomicCAS(&";

        op->arr.accept(this);
        stream << "[";
        parentPrecedence = Precedence::TOP;
        op->loc.accept(this);
        stream << "]";

        stream << ", " << oldValueName << ", ";
        stream << oldValueName << " * ";
        mul->b.accept(this);
        stream << ");" << endl;
      } else if (isa<Add>(op->data)) {
        auto add = to<Add>(op->data);
        taco_iassert(isa<Load>(add->a));
        taco_iassert(to<Load>(add->a)->arr == op->arr && to<Load>(add->a)->loc == op->loc);
        if (deviceFunctionLoopDepth == 0 || op->atomic_parallel_unit == ParallelUnit::GPUWarp) {
          // use atomicAddWarp
          doIndent();
          stream << "atomicAddWarp<" << printCUDAType(add->b.type(), false) << ">(";
          op->arr.accept(this);
          stream << ", ";
          op->loc.accept(this);
          stream << ", ";
          add->b.accept(this);
          stream << ");" << endl;
        }
        else {
          doIndent();
          stream << "atomicAdd(&";

          op->arr.accept(this);
          stream << "[";
          parentPrecedence = Precedence::TOP;
          op->loc.accept(this);
          stream << "]";

          stream << ", ";
          add->b.accept(this);
          stream << ");" << endl;
        }
      } else if (isa<BitOr>(op->data)) {
        auto bitOr = to<BitOr>(op->data);
        taco_iassert(isa<Load>(bitOr->a));
        taco_iassert(to<Load>(bitOr->a)->arr == op->arr && to<Load>(bitOr->a)->loc == op->loc);

        doIndent();
        stream << "atomicOr(&";

        op->arr.accept(this);
        stream << "[";
        parentPrecedence = Precedence::TOP;
        op->loc.accept(this);
        stream << "]";

        stream << ", ";
        bitOr->b.accept(this);
        stream << ");" << endl;
      } else {
        taco_ierror;
      }
    }
  }
  else {
    IRPrinter::visit(op);
  }
}
  
void CodeGen_CUDA::generateShim(const Stmt& func, stringstream &ret) {
  const Function *funcPtr = func.as<Function>();
  ret << "extern \"C\" {\n";
  ret << "  int _shim_" << funcPtr->name << "(void** parameterPack);\n";
  ret << "}\n\n";

  ret << "int _shim_" << funcPtr->name << "(void** parameterPack) {\n";
  ret << "  return " << funcPtr->name << "(";

  size_t i=0;
  string delimiter = "";

  const auto returnType = funcPtr->getReturnType();
  if (returnType.second != Datatype()) {
    ret << "(void**)(parameterPack[0]), ";
    ret << "(char*)(parameterPack[1]), ";
    ret << "(" << printCUDAType(returnType.second, true) << ")(parameterPack[2]), ";
    ret << "(int32_t*)(parameterPack[3])";

    i = 4;
    delimiter = ", ";
  }

  for (auto output : funcPtr->outputs) {
    auto var = output.as<Var>();
    auto cast_type = var->is_tensor ? "taco_tensor_t*"
                                    : printCUDAType(var->type, var->is_ptr);

    ret << delimiter << "(" << cast_type << ")(parameterPack[" << i++ << "])";
    delimiter = ", ";
  }
  for (auto input : funcPtr->inputs) {
    auto var = input.as<Var>();
    auto cast_type = var->is_tensor ? "taco_tensor_t*"
                                    : printCUDAType(var->type, var->is_ptr);
    ret << delimiter << "(" << cast_type << ")(parameterPack[" << i++ << "])";
    delimiter = ", ";
  }
  ret << ");\n";
  ret << "}\n";
}

}}
