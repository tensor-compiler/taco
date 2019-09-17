#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <algorithm>
#include <unordered_set>
#include <taco.h>

#include "taco/ir/ir_visitor.h"
#include "codegen_c.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

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
  "#include <stdbool.h>\n"
  "#include <math.h>\n"
  "#include <complex.h>\n"
  "#include <string.h>\n"
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
  "int cmp(const void *a, const void *b) {\n"
  "  return *((const int*)a) - *((const int*)b);\n"
  "}\n"
  "#endif\n";
} // anonymous namespace

// find variables for generating declarations
// generates a single var for each GetProperty
class CodeGen_C::FindVars : public IRVisitor {
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
  vector<Expr> inputTensors;

  CodeGen_C *codeGen;

  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs, CodeGen_C *codeGen)
  : codeGen(codeGen) {
    for (auto v: inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(varMap.count(var)==0) << "Duplicate input found in codegen";
      inputTensors.push_back(v);
      varMap[var] = var->name;
    }
    for (auto v: outputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(varMap.count(var)==0) << "Duplicate output found in codegen";
      outputTensors.push_back(v);
      varMap[var] = var->name;
    }
  }

protected:
  using IRVisitor::visit;

  virtual void visit(const Var *op) {
    if (varMap.count(op) == 0) {
      varMap[op] = codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->rhs.accept(this);
  }

  virtual void visit(const For *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
  }

  virtual void visit(const GetProperty *op) {
    if (!util::contains(inputTensors, op->tensor) &&
        !util::contains(outputTensors, op->tensor)) {
      // Don't create header unpacking code for temporaries
      return;
    }

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

CodeGen_C::CodeGen_C(std::ostream &dest, OutputKind outputKind)
    : CodeGen(dest, false, true, C), out(dest), outputKind(outputKind) {}

CodeGen_C::~CodeGen_C() {}

void CodeGen_C::compile(Stmt stmt, bool isFirst) {
  if (isFirst) {
    // output the headers
    out << cHeaders;
  }
  out << endl;
  // generate code for the Stmt
  stmt.accept(this);
}

void CodeGen_C::visit(const Function* func) {
  // if generating a header, protect the function declaration with a guard
  if (outputKind == HeaderGen) {
    out << "#ifndef TACO_GENERATED_" << func->name << "\n";
    out << "#define TACO_GENERATED_" << func->name << "\n";
  }

  int numYields = countYields(func);
  emittingCoroutine = (numYields > 0);
  funcName = func->name;
  labelCount = 0;

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
  FindVars varFinder(func->inputs, func->outputs, this);
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

void CodeGen_C::visit(const VarDecl* op) {
  if (emittingCoroutine) {
    doIndent();
    op->var.accept(this);
    parentPrecedence = Precedence::TOP;
    stream << " = ";
    op->rhs.accept(this);
    stream << ";";
    stream << endl;
  } else {
    IRPrinter::visit(op);
  }
}

void CodeGen_C::visit(const Yield* op) {
  printYield(op, localVars, varMap, labelCount, funcName);
}

// For Vars, we replace their names with the generated name,
// since we match by reference (not name)
void CodeGen_C::visit(const Var* op) {
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
  ret << "#pragma omp parallel for schedule";
  switch (kind) {
    case LoopKind::Static:
      ret << "(static)";
      break;
    case LoopKind::Dynamic:
      ret << "(dynamic, 16)";
      break;
    case LoopKind::Runtime:
      ret << "(runtime)";
      break;
    default:
      break;
  }
  return ret.str();
}

// The next two need to output the correct pragmas depending
// on the loop kind (Serial, Static, Dynamic, Vectorized)
//
// Docs for vectorization pragmas:
// http://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-loop-hint-optimizations
void CodeGen_C::visit(const For* op) {
  switch (op->kind) {
    case LoopKind::Vectorized:
      doIndent();
      out << genVectorizePragma(op->vec_width);
      out << "\n";
      break;
    case LoopKind::Static:
    case LoopKind::Dynamic:
    case LoopKind::Runtime:
      doIndent();
      out << getParallelizePragma(op->kind);
      out << "\n";
    default:
      break;
  }
  
  doIndent();
  stream << keywordString("for") << " ("; 
  if (!emittingCoroutine) {
    stream << keywordString(util::toString(op->var.type())) << " ";
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

void CodeGen_C::visit(const While* op) {
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

void CodeGen_C::visit(const GetProperty* op) {
  taco_iassert(varMap.count(op) > 0) <<
      "Property " << Expr(op) << " of " << op->tensor << " not found in varMap";
  out << varMap[op];
}

void CodeGen_C::visit(const Min* op) {
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

void CodeGen_C::visit(const Max* op) {
  stream << "TACO_MAX(";
  op->a.accept(this);
  stream << ", ";
  op->b.accept(this);
  stream << ")";
}

void CodeGen_C::visit(const Allocate* op) {
  string elementType = printCType(op->var.type(), false);

  doIndent();
  op->var.accept(this);
  stream << " = (";
  stream << elementType << "*";
  stream << ")";
  if (op->is_realloc) {
    stream << "realloc(";
    op->var.accept(this);
    stream << ", ";
  }
  else {
    stream << "malloc(";
  }
  stream << "sizeof(" << elementType << ")";
  stream << " * ";
  op->num_elements.accept(this);
  stream << ");";
    stream << endl;
}

void CodeGen_C::visit(const Sqrt* op) {
  taco_tassert(op->type.isFloat() && op->type.getNumBits() == 64) <<
      "Codegen doesn't currently support non-double sqrt";
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}
  
void CodeGen_C::generateShim(const Stmt& func, stringstream &ret) {
  const Function *funcPtr = func.as<Function>();
  
  ret << "int _shim_" << funcPtr->name << "(void** parameterPack) {\n";
  ret << "  return " << funcPtr->name << "(";

  size_t i=0;
  string delimiter = "";

  const auto returnType = funcPtr->getReturnType();
  if (returnType.second != Datatype()) {
    ret << "(void**)(parameterPack[0]), ";
    ret << "(char*)(parameterPack[1]), ";
    ret << "(" << returnType.second << "*)(parameterPack[2]), ";
    ret << "(int32_t*)(parameterPack[3])";

    i = 4;
    delimiter = ", ";
  }
  
  for (auto output : funcPtr->outputs) {
    auto var = output.as<Var>();
    auto cast_type = var->is_tensor ? "taco_tensor_t*"
    : printCType(var->type, var->is_ptr);
    
    ret << delimiter << "(" << cast_type << ")(parameterPack[" << i++ << "])";
    delimiter = ", ";
  }
  for (auto input : funcPtr->inputs) {
    auto var = input.as<Var>();
    auto cast_type = var->is_tensor ? "taco_tensor_t*"
    : printCType(var->type, var->is_ptr);
    ret << delimiter << "(" << cast_type << ")(parameterPack[" << i++ << "])";
    delimiter = ", ";
  }
  ret << ");\n";
  ret << "}\n";
}

}}
