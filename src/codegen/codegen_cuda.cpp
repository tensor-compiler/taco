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

using namespace std;

namespace taco {
  namespace ir {

// Some helper functions
    namespace {

#define CUDA_BLOCK_SIZE 256

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

  CodeGen_CUDA *codeGen;

  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs, CodeGen_CUDA *codeGen,
           bool stopAtDeviceFunction=false)
  : codeGen(codeGen) {
    for (auto v: inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(varMap.count(var) == 0) <<
                                           "Duplicate input found in codegen";
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
  }

protected:
  using IRVisitor::visit;

  virtual void visit(const For *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    if (op->accelerator && stopAtDeviceFunction) {
      return;
    }
    op->contents.accept(this);
  }

  virtual void visit(const Var *op) {
    if (varMap.count(op) == 0) {
      varMap[op] = codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const GetProperty *op) {
    if (varMap.count(op) == 0) {
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
  vector<Stmt> deviceFunctions;
  map<Expr, string, ExprCompare> scopeMap;

  // the variables to pass to each device function
  vector<vector<pair<string, Expr>>> functionParameters;
  vector<pair<string, Expr>> currentParameters; // keep as vector so code generation is deterministic
  set<Expr> currentParameterSet;

  vector<pair<string, Expr>> threadIDVars;

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
    op->var.accept(this);
    if (op->accelerator) {
      taco_iassert(!inDeviceFunction) << "Nested Device functions not supported";
      deviceFunctions.push_back(op);
      threadIDVars.push_back(pair<string, Expr>(scopeMap[op->var], op->var));
      currentParameters.clear();
      currentParameterSet.clear();
      inDeviceFunction = true;
    }
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
    if (op->accelerator) {
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
    else if (scopeMap.count(op) == 1 && inDeviceFunction && currentParameterSet.count(op) == 0 && op != threadIDVars.back().second) {
      currentParameters.push_back(pair<string, Expr>(scopeMap[op], op));
      currentParameterSet.insert(op);
    }
  }

  virtual void visit(const VarDecl *op) {
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
      ret << delimiter << "taco_tensor_t *" << varName;
    }
    else {
      auto tp = printCUDAType(var->type, var->is_ptr);
      ret << delimiter << tp << " " << var->name;
    }
    // No non-tensor parameters
    delimiter = ", ";
  }
  ret << ")";
  return ret.str();
}

void CodeGen_CUDA::printThreadIDVariable(pair<string, Expr> threadIDVar, Expr start, Expr increment) {
  auto var = threadIDVar.second.as<Var>();
  taco_iassert(var) << "Unable to convert output " << threadIDVar.second
                    << " to Var";
  string varName = threadIDVar.first;
  auto tp = printCUDAType(var->type, var->is_ptr);
  stream << tp << " " << varName << " = ";
  increment = ir::simplify(increment);
  if (!isa<Literal>(increment) || !to<Literal>(increment)->equalsScalar(1)) {
    stream << "(blockIdx.x * blockDim.x + threadIdx.x) * ";
    increment.accept(this);
  }
  else {
    stream << "blockIdx.x * blockDim.x + threadIdx.x";
  }
  Expr expr = ir::simplify(start);
  if (!isa<Literal>(expr) || !to<Literal>(expr)->equalsScalar(0)) {
    stream << " + ";
    expr.accept(this);
  }
  stream << ";\n";
}

void CodeGen_CUDA::printThreadBoundCheck(pair<string, Expr> threadIDVar, Expr end) {
  taco_iassert(threadIDVar.second.as<Var>()) << "Unable to convert output " << threadIDVar.second
                                             << " to Var";
  string varName = threadIDVar.first;
  end = ir::simplify(end);
  stream << "if (" << varName << " >= ";
  end.accept(this);
  stream << ") {" << "\n";
  indent++;
  doIndent();
  stream << "return;\n";
  indent--;
  doIndent();
  stream << "}" << "\n" << "\n";
}

void CodeGen_CUDA::printDeviceFuncCall(const vector<pair<string, Expr>> currentParameters, int index, Expr start, Expr end, Expr increment) {
  stream << funcName << "DeviceKernel" << index << "<<<";
  // ensure always rounds up
  Expr loopIterations = Div::make(Add::make(Sub::make(end, start), Sub::make(increment, Literal::make(1, Int()))), increment);
  loopIterations = ir::simplify(loopIterations);
  Expr blockSize = Div::make(Add::make(loopIterations, Literal::make(CUDA_BLOCK_SIZE - 1, Int())), Literal::make(CUDA_BLOCK_SIZE));
  blockSize = ir::simplify(blockSize);
  blockSize.accept(this);
  stream << ", " << CUDA_BLOCK_SIZE << ">>>";
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
  doIndent();
  stream << "cudaDeviceSynchronize();\n";
}


CodeGen_CUDA::CodeGen_CUDA(std::ostream &dest, OutputKind outputKind)
      : CodeGen(dest, false, false, CUDA), out(dest), outputKind(outputKind) {}

CodeGen_CUDA::~CodeGen_CUDA() {}

void CodeGen_CUDA::compile(Stmt stmt, bool isFirst) {
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
  DeviceFunctionCollector deviceFunctionCollector(func->inputs, func->outputs, this);
  func->body.accept(&deviceFunctionCollector);
  deviceFunctions = deviceFunctionCollector.deviceFunctions;
  deviceFunctionParameters = deviceFunctionCollector.functionParameters;

  resetUniqueNameCounters();
  for (size_t i = 0; i < deviceFunctions.size(); i++) {
    const For *forloop = to<For>(deviceFunctions[i]);
    Stmt function = forloop->contents;
    vector<pair<string, Expr>> parameters = deviceFunctionParameters[i];
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
    inputs.push_back(deviceFunctionCollector.threadIDVars[i].second);
    FindVars varFinder(inputs, {}, this);
    forloop->accept(&varFinder);
    varMap = varFinder.varMap;

    // Print variable declarations
    out << printDecls(varFinder.varDecls, inputs, {}) << endl;
    doIndent();
    printThreadIDVariable(deviceFunctionCollector.threadIDVars[i], forloop->start, forloop->increment);
    doIndent();
    printThreadBoundCheck(deviceFunctionCollector.threadIDVars[i], forloop->end);

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
    printDeviceFunctions(func);
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
  out << "return 0;\n";
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

// The next two need to output the correct pragmas depending
// on the loop kind (Serial, Static, Dynamic, Vectorized)
//
// Docs for vectorization pragmas:
// http://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-loop-hint-optimizations
void CodeGen_CUDA::visit(const For* op) {
  for (size_t i = 0; i < deviceFunctions.size(); i++) {
    auto dFunction = deviceFunctions[i].as<For>();
    assert(dFunction);
    if (op == dFunction) {
      // Generate kernel launch
      doIndent();
      printDeviceFuncCall(deviceFunctionParameters[i], i, op->start, op->end, op->increment);
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
    default:
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

  op->contents.accept(this);
  doIndent();
  stream << "}";
  stream << endl;
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
  stream << "TACO_MAX(";
  op->a.accept(this);
  stream << ", ";
  op->b.accept(this);
  stream << ")";
}

void CodeGen_CUDA::visit(const Allocate* op) {
  string elementType = printCUDAType(op->var.type(), false);
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

void CodeGen_CUDA::visit(const Sqrt* op) {
  taco_tassert(op->type.isFloat() && op->type.getNumBits() == 64) <<
                                                                  "Codegen doesn't currently support non-double sqrt";
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
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
  else {
    doIndent();
    stream << keywordString(printCUDAType(op->var.type(), false)) << " ";
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
void CodeGen_CUDA::printBinCastedOp(Expr a, Expr b, string op, Precedence precedence) {
  bool parenthesize = precedence > parentPrecedence;
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
  stream << op->func << "(";
  parentPrecedence = Precedence::CALL;

  // Need to print cast to type so that arguments match
  if (op->args.size() > 0) {
    if (op->type != op->args[0].type() || isa<Literal>(op->args[0])) {
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
