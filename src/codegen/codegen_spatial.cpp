#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <algorithm>
#include <unordered_set>
#include <taco.h>

#include "taco/ir/ir_visitor.h"
#include "codegen_spatial.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"
#include "taco/spatial.h"

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
  "int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {\n"
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
  "int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {\n"
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
  "taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,\n"
  "                                  int32_t* dimensions, int32_t* mode_ordering,\n"
  "                                  taco_mode_t* mode_types) {\n"
  "  taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));\n"
  "  t->order         = order;\n"
  "  t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));\n"
  "  t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));\n"
  "  t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));\n"
  "  t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));\n"
  "  t->csize         = csize;\n"
  "  for (int32_t i = 0; i < order; i++) {\n"
  "    t->dimensions[i]    = dimensions[i];\n"
  "    t->mode_ordering[i] = mode_ordering[i];\n"
  "    t->mode_types[i]    = mode_types[i];\n"
  "    switch (t->mode_types[i]) {\n"
  "      case taco_mode_dense:\n"
  "        t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));\n"
  "        break;\n"
  "      case taco_mode_sparse:\n"
  "        t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));\n"
  "        break;\n"
  "    }\n"
  "  }\n"
  "  return t;\n"
  "}\n"
  "void deinit_taco_tensor_t(taco_tensor_t* t) {\n"
  "  for (int i = 0; i < t->order; i++) {\n"
  "    free(t->indices[i]);\n"
  "  }\n"
  "  free(t->indices);\n"
  "  free(t->dimensions);\n"
  "  free(t->mode_ordering);\n"
  "  free(t->mode_types);\n"
  "  free(t);\n"
  "}\n"
  "#endif\n";
} // anonymous namespace

// find variables for generating declarations
// generates a single var for each GetProperty
class CodeGen_Spatial::FindVars : public IRVisitor {
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

  CodeGen_Spatial *codeGen;

  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs, CodeGen_Spatial *codeGen)
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

  virtual void visit(const Scope *op) {
    op->scopedStmt.accept(this);
    if (op->returnExpr.defined())
      op->returnExpr.accept(this);
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

CodeGen_Spatial::CodeGen_Spatial(std::ostream &dest, OutputKind outputKind, bool simplify)
    : CodeGen(dest, false, simplify, C), out(dest), outputKind(outputKind) {}

CodeGen_Spatial::~CodeGen_Spatial() {}

void CodeGen_Spatial::compile(Stmt stmt, bool isFirst) {
  varMap = {};
  localVars = {};

  if (isFirst) {
    // output the headers
    out << cHeaders;
  }
  out << endl;
  // generate code for the Stmt
  stmt.accept(this);
}

string CodeGen_Spatial::printFuncName(const Function *func, 
                              std::map<Expr, std::string, ExprCompare> inputMap, 
                              std::map<Expr, std::string, ExprCompare> outputMap) {
  stringstream ret;

  ret << "import spatial.dsl._" << endl;
  ret << "\n";
  ret << "class " << func->name << "_0 extends " << func->name << "()" << endl;
  ret << "\n";


  ret << "@spatial abstract class " << func->name << "(" << endl;

  // Parameters here

  ret << ") extends SpatialTest with ComputeCheck {" << endl;
  ret << "  type T = Int" << endl;  // FIXME: make type changeable
  ret << "\n";
  ret << "def main(args: Array[String]): Unit =" << endl;
  return ret.str();
}

void CodeGen_Spatial::visit(const Function* func) {
  // if generating a header, protect the function declaration with a guard
  if (outputKind == HeaderGen) {
    out << "#ifndef TACO_GENERATED_" << func->name << "\n";
    out << "#define TACO_GENERATED_" << func->name << "\n";
  }

  int numYields = countYields(func);
  emittingCoroutine = (numYields > 0);
  funcName = func->name;
  labelCount = 0;

  resetUniqueNameCounters();
  FindVars inputVarFinder(func->inputs, {}, this);
  func->body.accept(&inputVarFinder);
  FindVars outputVarFinder({}, func->outputs, this);
  func->body.accept(&outputVarFinder);

  // output function declaration
  doIndent();
  out << printFuncName(func, inputVarFinder.varDecls, outputVarFinder.varDecls);

  // if we're just generating a header, this is all we need to do
  if (outputKind == HeaderGen) {
    out << ";\n";
    out << "#endif\n";
    return;
  }

  out << "{\n";

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

  out << printInitMem(varFinder.varDecls, func->inputs, func->outputs) << endl;

  doIndent();
  out << "Accel {\n";
  indent++;
  //out << printDeclsAccel(varFinder.varDecls, func->inputs, func->outputs) << endl;

  // output body
  print(func->body);

  // output repack only if we allocated memory
//  if (checkForAlloc(func))
//    out << endl << printPack(varFinder.outputProperties, func->outputs);

  if (emittingCoroutine) {
    out << printCoroutineFinish(numYields, funcName);
  }

  // Reformat output (store back into DRAM)
  //out << printOutputStore(varFinder.outputProperties, func->outputs);

  out << "  }\n"; // end Accel
  indent--;

  out << "\n";
  
  indent++;
  out << printOutputCheck(varFinder.outputProperties, func->outputs);
  //out << "assert(true)\n";

  out << "}\n";   // end main
  out << "}\n";   // end SpatialTest
  //out << "return 0;\n";
}


void CodeGen_Spatial::visit(const VarDecl* op) {
  if (emittingCoroutine) {
    doIndent();
    op->var.accept(this);
    parentPrecedence = Precedence::TOP;
    stream << " = ";
    op->rhs.accept(this);
    stream << ";";
    stream << endl;
  } else {
    if (op->mem == MemoryLocation::SpatialReg) {
      doIndent();
      stream << "val";
      taco_iassert(isa<Var>(op->var));
      stream << " ";
      string varName = varNameGenerator.getUniqueName(util::toString(op->var));
      varNames.insert({op->var, varName});
      op->var.accept(this);
      parentPrecedence = Precedence::TOP;
      stream << " = Reg[T](";
      op->rhs.accept(this);
      stream << ".to[T])";
      //stream << ";";
      stream << endl;
    } else {
      doIndent();
      stream << "val";
      taco_iassert(isa<Var>(op->var));
      stream << " ";
      string varName = varNameGenerator.getUniqueName(util::toString(op->var));
      varNames.insert({op->var, varName});
      op->var.accept(this);
      parentPrecedence = Precedence::TOP;
      stream << " = ";
      op->rhs.accept(this);
      //stream << ";";
      stream << endl;
    }
  }
}

void CodeGen_Spatial::visit(const Yield* op) {
  printYield(op, localVars, varMap, labelCount, funcName);
}

// For Vars, we replace their names with the generated name,
// since we match by reference (not name)
void CodeGen_Spatial::visit(const Var* op) {
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

void CodeGen_Spatial::visit(const Malloc* op) {
  stream << "SRAM[T](";
  parentPrecedence = Precedence::TOP;
  op->size.accept(this);
  stream << ")";
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
      ret << "(static, 1)";
      break;
    case LoopKind::Dynamic:
      ret << "(dynamic, 1)";
      break;
    case LoopKind::Runtime:
      ret << "(runtime)";
      break;
    case LoopKind::Static_Chunked:
      ret << "(static)";
      break;
    default:
      break;
  }
  return ret.str();
}

static string getUnrollPragma(size_t unrollFactor) {
  return "#pragma unroll " + std::to_string(unrollFactor);
}

static string getAtomicPragma() {
  return "#pragma omp atomic";
}

// The next two need to output the correct pragmas depending
// on the loop kind (Serial, Static, Dynamic, Vectorized)
//
// Docs for vectorization pragmas:
// http://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-loop-hint-optimizations
void CodeGen_Spatial::visit(const For* op) {

  // FIXME: [Spatial] See if this is the correct location
  doIndent();
  stream << keywordString("Foreach") << " (";

  auto start_lit = op->start.as<Literal>();
  if (start_lit != nullptr && !((start_lit->type.isInt() && 
                                start_lit->equalsScalar(0)) ||
                               (start_lit->type.isUInt() && 
                                start_lit->equalsScalar(0)))) {
    op->start.accept(this);
    stream << " to ";
  }
  //stream << keywordString("; ");
  //op->var.accept(this);
  parentPrecedence = BOTTOM;
  op->end.accept(this);
  stream << keywordString(" by ");
  //op->var.accept(this);

  auto lit = op->increment.as<Literal>();
  if (lit != nullptr && ((lit->type.isInt()  && lit->equalsScalar(1)) ||
                         (lit->type.isUInt() && lit->equalsScalar(1)))) {
    stream << "1";
  }
  else {
    //stream << " += ";
    op->increment.accept(this);
  }
  
  // Parallelization factor in spatial 
  if (op->numChunks > 0 && op->numChunks <= 16) {
    stream << " par " << op->numChunks;
  }

  stream << ") { ";
  op->var.accept(this);
  stream << " =>\n";

  op->contents.accept(this);
  doIndent();
  stream << "}";
  stream << endl;
}

void CodeGen_Spatial::visit(const While* op) {
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

void CodeGen_Spatial::visit(const Reduce* op) {
  doIndent();
  stream << keywordString("Reduce") << "(" << op->reg << ")(";
  op->start.accept(this);
  stream << keywordString(" until ");
  parentPrecedence = BOTTOM;
  op->end.accept(this);
  stream << keywordString(" by ");
  op->increment.accept(this);
  if (op->par > 0 && op->par <= 16) {
    stream << " par " << op->par;
  }
  stream << ") {";
  op->var.accept(this);
  stream << " => \n";

  op->contents.accept(this);
  stream << endl;
  doIndent();

  stream << "} { _ ";
  if (op->add)
    stream << "+";
  else
    stream << "-";
  stream << " _ }";
  stream << endl;
}

void CodeGen_Spatial::visit(const GetProperty* op) {
  taco_iassert(varMap.count(op) > 0) <<
      "Property " << Expr(op) << " of " << op->tensor << " not found in varMap";
  out << varMap[op];
}

void CodeGen_Spatial::visit(const Min* op) {
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

void CodeGen_Spatial::visit(const Max* op) {
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

void CodeGen_Spatial::visit(const Allocate* op) {
  string elementType = printCType(op->var.type(), false);

  doIndent();
  stream << "val ";
  op->var.accept(this);
  stream << " = ";
  if (op->is_realloc) {
    stream << "realloc(";
    op->var.accept(this);
    stream << ", ";
  }
  else {
    stream << "SRAM[T](";
  }
  parentPrecedence = MUL;
  op->num_elements.accept(this);
  parentPrecedence = TOP;
  stream << ")";
    stream << endl;
}

void CodeGen_Spatial::visit(const Sqrt* op) {
  taco_tassert(op->type.isFloat() && op->type.getNumBits() == 64) <<
      "Codegen doesn't currently support non-double sqrt";
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}

void CodeGen_Spatial::visit(const Assign* op) {
  if (op->use_atomics) {
    doIndent();
    stream << getAtomicPragma() << endl;
  }

  doIndent();
  op->lhs.accept(this);
  parentPrecedence = Precedence::TOP;
  bool printed = false;
//  if (simplify) {
//    if (isa<ir::Add>(op->rhs)) {
//      auto add = to<Add>(op->rhs);
//      if (add->a == op->lhs) {
//        const Literal* lit = add->b.as<Literal>();
//        if (lit != nullptr && ((lit->type.isInt()  && lit->equalsScalar(1)) ||
//                               (lit->type.isUInt() && lit->equalsScalar(1)))) {
//          stream << "++";
//        }
//        else {
//          stream << " += ";
//          add->b.accept(this);
//        }
//        printed = true;
//      }
//    }
//    else if (isa<Mul>(op->rhs)) {
//      auto mul = to<Mul>(op->rhs);
//      if (mul->a == op->lhs) {
//        stream << " *= ";
//        mul->b.accept(this);
//        printed = true;
//      }
//    }
//    else if (isa<BitOr>(op->rhs)) {
//      auto bitOr = to<BitOr>(op->rhs);
//      if (bitOr->a == op->lhs) {
//        stream << " |= ";
//        bitOr->b.accept(this);
//        printed = true;
//      }
//    }
//  }
  if (!printed) {
    stream << " = ";
    op->rhs.accept(this);
  }

  //stream << ";";
  stream << endl;
}

void CodeGen_Spatial::visit(const Store* op) {
  if (op->use_atomics) {
    doIndent();
    stream << getAtomicPragma() << endl;
  }

  doIndent();
  op->arr.accept(this);
  stream << "(";
  parentPrecedence = Precedence::TOP;
  op->loc.accept(this);
  stream << ") = ";
  parentPrecedence = Precedence::TOP;
  op->data.accept(this);
  //stream << ";";
  stream << endl;
}

void CodeGen_Spatial::visit(const MemStore* op) {
  doIndent();
  op->lhsMem.accept(this);
  stream << "(";
  parentPrecedence = Precedence::TOP;
  op->start.accept(this);
  stream << ":";
  op->start.accept(this);
  stream << "+";
  op->offset.accept(this);
  stream << ") store ";
  parentPrecedence = Precedence::TOP;
  op->rhsMem.accept(this);
  stream << endl;
}

void CodeGen_Spatial::visit(const Load* op) {
  parentPrecedence = Precedence::LOAD;
  op->arr.accept(this);
  stream << "(";
  parentPrecedence = Precedence::LOAD;
  op->loc.accept(this);
  stream << ")";
}

void CodeGen_Spatial::visit(const MemLoad* op) {
  doIndent();
  op->lhsMem.accept(this);
  stream << "(";
  parentPrecedence = Precedence::TOP;
  op->start.accept(this);
  stream << ":";
  op->start.accept(this);
  stream << "+";
  op->offset.accept(this);
  stream << ")";
  stream << " load ";
  parentPrecedence = Precedence::TOP;
  op->rhsMem.accept(this);
  stream << endl;
}

void CodeGen_Spatial::visit(const Free* op) {
  parentPrecedence = Precedence::TOP;
}

void CodeGen_Spatial::generateShim(const Stmt& func, stringstream &ret) {
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

string CodeGen_Spatial::unpackTensorPropertyAccel(string varname, const GetProperty* op,
                          bool is_output_prop) {
  stringstream ret;
  string indentation = "    ";
  ret << indentation;

  auto tensor = op->tensor.as<Var>();
  if (op->property == TensorProperty::Values) {
    // for the values, it's in the last slot
    ret << "val " << varname << " = SRAM[T](";
    if (op->index == 0) {
      ret << "1";
    }
    else {
      for (int i = 1; i < op->index + 1; i++) {
        ret << tensor->name << i << "_dimension";

        if (i < op->index) {
          if (should_use_Spatial_multi_dim())
            ret << ", ";
          else
            ret << " * ";
        }
      }
    }

    ret << ")" << endl; 

    // Load from DRAM into SRAM
    if (!is_output_prop)
      ret << indentation << varname << " load " << varname << "_dram" << endl;

    //stringstream newVarname;
    //newVarname << varname << "_sram";
    //varMap[op] = newVarname.str();

    return ret.str();
  } else if (op->property == TensorProperty::ValuesSize) {
    ret << "int " << varname << " = " << tensor->name << "->vals_size;\n";

    return ret.str();
  }

  string tp;

  // for a Dense level, nnz is an int
  // for a Fixed level, ptr is an int
  // all others are int*
  if (op->property == TensorProperty::Dimension) {
    ret << "val " << varname << " = " << op->index << endl;

//    stringstream newVarname;
//    newVarname << varname << "_sram";
//    varMap[op] = newVarname.str();

  } else {
    taco_iassert(op->property == TensorProperty::Indices);
    tp = "int*";
    auto nm = op->index;
    ret << tp << " " << restrictKeyword() << " " << varname << " = ";
    ret << "(int*)(" << tensor->name << "->indices[" << op->mode;
    ret << "][" << nm << "]);\n";
  }

  return ret.str();
  
}

string CodeGen_Spatial::unpackTensorProperty(string varname, const GetProperty* op,
                            bool is_output_prop) {
  stringstream ret;
  ret << "  ";

  auto tensor = op->tensor.as<Var>();
  if (op->property == TensorProperty::Values) {
// FIXME: [Spatial] add this in for scalar outputs. 
//    if (is_output_prop) {
//      ret << "val " << varname << " = ArgOut[T]()" << endl;
//      
//    } else {
      // for the values, it's in the last slot
      //ret << "val " << varname << "_dram = DRAM[T](";
      string loc = "DRAM";
      if (tensor->memoryLocation == MemoryLocation::SpatialSRAM)
        loc = "SRAM";
      else if (tensor->memoryLocation == MemoryLocation::SpatialReg)
        loc = "Reg";

      ret << "val " << varname << " = " << loc << "[T](";
      if (op->index == 0) {
        ret << "1";
      }
      else {
        for (int i = 1; i < op->index + 1; i++) {
          ret << tensor->name << i << "_dimension_dram";

          if (i < op->index) {
            if (should_use_Spatial_multi_dim())
              ret << ", ";
            else
              ret << " * ";
          }
        }
      }
      ret << ")" << endl; 

    return ret.str();
  } else if (op->property == TensorProperty::ValuesSize) {
    ret << "int " << varname << " = " << tensor->name << "->vals_size;\n";
    return ret.str();
  }

  string tp;

  // for a Dense level, nnz is an int
  // for a Fixed level, ptr is an int
  // all others are int*
  if (op->property == TensorProperty::Dimension) {
    ret << "val " << varname << "_dram = " << op->index << endl;
  } else {
    taco_iassert(op->property == TensorProperty::Indices);
    tp = "int*";
    auto nm = op->index;
    ret << tp << " " << restrictKeyword() << " " << varname << " = ";
    ret << "(int*)(" << tensor->name << "->indices[" << op->mode;
    ret << "][" << nm << "]);\n";
  }

  return ret.str();
}
// helper to print declarations
string CodeGen_Spatial::printDeclsAccel(map<Expr, string, ExprCompare> varMap,
                           vector<Expr> inputs, vector<Expr> outputs) {
  stringstream ret;
  unordered_set<string> propsAlreadyGenerated;

  vector<const GetProperty*> sortedProps;

  for (auto const& p: varMap) {
    if (p.first.as<GetProperty>())
      sortedProps.push_back(p.first.as<GetProperty>());
  }
  // sort the properties in order to generate them in a canonical order
  sort(sortedProps.begin(), sortedProps.end(),
       [&](const GetProperty *a,
           const GetProperty *b) -> bool {
         // first, use a total order of outputs,inputs
         auto a_it = find(outputs.begin(), outputs.end(), a->tensor);
         auto b_it = find(outputs.begin(), outputs.end(), b->tensor);
         auto a_pos = distance(outputs.begin(), a_it);
         auto b_pos = distance(outputs.begin(), b_it);
         if (a_it == outputs.end())
           a_pos += distance(inputs.begin(), find(inputs.begin(), inputs.end(),
                                                  a->tensor));
         if (b_it == outputs.end())
           b_pos += distance(inputs.begin(), find(inputs.begin(), inputs.end(),
                                                  b->tensor));

         // if total order is same, have to do more, otherwise we know
         // our answer
         if (a_pos != b_pos)
           return a_pos < b_pos;

         // if they're different properties, sort by property
         if (a->property != b->property)
           return a->property < b->property;

         // now either the mode gives order, or index #
         if (a->mode != b->mode)
           return a->mode < b->mode;

         return a->index < b->index;
       });

  for (auto prop: sortedProps) {
    bool isOutputProp = (find(outputs.begin(), outputs.end(),
                              prop->tensor) != outputs.end());
    
    auto var = prop->tensor.as<Var>();
    if (!var->is_parameter) {
      ret << unpackTensorPropertyAccel(varMap[prop], prop, isOutputProp);
    }
    propsAlreadyGenerated.insert(varMap[prop]);
  }

  return ret.str();
}

// helper to print declarations
string CodeGen_Spatial::printInitMem(map<Expr, string, ExprCompare> varMap,
                           vector<Expr> inputs, vector<Expr> outputs) {
  stringstream ret;
  unordered_set<string> propsAlreadyGenerated;

  vector<const GetProperty*> sortedProps;

  for (auto const& p: varMap) {
    if (p.first.as<GetProperty>())
      sortedProps.push_back(p.first.as<GetProperty>());
  }
  // sort the properties in order to generate them in a canonical order
  sort(sortedProps.begin(), sortedProps.end(),
       [&](const GetProperty *a,
           const GetProperty *b) -> bool {
         // first, use a total order of outputs,inputs
         auto a_it = find(outputs.begin(), outputs.end(), a->tensor);
         auto b_it = find(outputs.begin(), outputs.end(), b->tensor);
         auto a_pos = distance(outputs.begin(), a_it);
         auto b_pos = distance(outputs.begin(), b_it);
         if (a_it == outputs.end())
           a_pos += distance(inputs.begin(), find(inputs.begin(), inputs.end(),
                                                  a->tensor));
         if (b_it == outputs.end())
           b_pos += distance(inputs.begin(), find(inputs.begin(), inputs.end(),
                                                  b->tensor));

         // if total order is same, have to do more, otherwise we know
         // our answer
         if (a_pos != b_pos)
           return a_pos < b_pos;

         // if they're different properties, sort by property
         if (a->property != b->property)
           return a->property < b->property;

         // now either the mode gives order, or index #
         if (a->mode != b->mode)
           return a->mode < b->mode;

         return a->index < b->index;
       });

  // Output initMem(...) function for Spatial App  
  ret << "  initMem[T](";
  for (int i = 0; i < (int)sortedProps.size(); i++) {
    auto prop = sortedProps[i];
    bool isOutputProp = (find(outputs.begin(), outputs.end(),
                              prop->tensor) != outputs.end());
    
    auto var = prop->tensor.as<Var>();
    if (!var->is_parameter) {
      ret << outputInitMemArgs(varMap[prop], prop, isOutputProp, i == (int)sortedProps.size() - 1);
    }
    propsAlreadyGenerated.insert(varMap[prop]);
  }
  ret << ")" << endl;

  return ret.str();
}

string CodeGen_Spatial::outputInitMemArgs(string varname, const GetProperty* op,
                          bool is_output_prop, bool last) {
  stringstream ret;
  string indentation = "";
  ret << indentation;

  //auto tensor = op->tensor.as<Var>();
  if (op->property == TensorProperty::Values && op->index == 0) {
    ret << "1, " << varname;
  }
  else if (op->property == TensorProperty::Values) {
    ret << varname << "_dram";
  } else if (op->property == TensorProperty::Dimension) {
    ret << varname << "_dram";
  }

  if (!last)
    ret << ", ";

  return ret.str();
  
}

// helper to print output store
string CodeGen_Spatial::printOutputCheck(map<tuple<Expr, TensorProperty, int, int>,
        string> outputProperties, vector<Expr> outputs) {
  stringstream ret;
  vector<tuple<Expr, TensorProperty, int, int>> sortedProps;

  for (auto &prop: outputProperties) {
    sortedProps.push_back(prop.first);
  }
  sort(sortedProps.begin(), sortedProps.end(),
       [&](const tuple<Expr, TensorProperty, int, int> &a,
           const tuple<Expr, TensorProperty, int, int> &b) -> bool {
         // first, use a total order of outputs,inputs
         auto a_it = find(outputs.begin(), outputs.end(), get<0>(a));
         auto b_it = find(outputs.begin(), outputs.end(), get<0>(b));
         auto a_pos = distance(outputs.begin(), a_it);
         auto b_pos = distance(outputs.begin(), b_it);

         // if total order is same, have to do more, otherwise we know
         // our answer
         if (a_pos != b_pos)
           return a_pos < b_pos;

         // if they're different properties, sort by property
         if (get<1>(a) != get<1>(b))
           return get<1>(a) < get<1>(b);

         // now either the mode gives order, or index #
         if (get<2>(a) != get<2>(b))
           return get<2>(a) < get<2>(b);

         return get<3>(a) < get<3>(b);
       });

  ret << "  checkOutput[T](";

  for (int i = 0; i < (int)sortedProps.size(); i++) {
    auto prop = sortedProps[i];
    auto var = get<0>(prop).as<Var>();
    if (!var->is_parameter) {
      ret << outputCheckOutputArgs(outputProperties[prop], get<0>(prop),
                              get<1>(prop), get<2>(prop), get<3>(prop), i == (int)sortedProps.size() - 1);
    }
  }
  ret << ")" << endl;
  return ret.str();
}

string CodeGen_Spatial::outputCheckOutputArgs(string varname, Expr tnsr,
                                   TensorProperty property,
                                   int mode, int index, bool last) {
  stringstream ret;
  ret << "";

  //auto tensor = tnsr.as<Var>();
  if (property == TensorProperty::Values && index == 0) {
    ret << "1, " << varname;
  } else if (property == TensorProperty::Values) {
      ret << varname << "_dram";
  } else if (property == TensorProperty::Dimension) {
    ret << varname << "_dram";
  } 

  if (!last) 
    ret << ", ";

  return ret.str();
}

// helper to print output store
string CodeGen_Spatial::printOutputStore(map<tuple<Expr, TensorProperty, int, int>,
        string> outputProperties, vector<Expr> outputs) {
  stringstream ret;
  vector<tuple<Expr, TensorProperty, int, int>> sortedProps;

  for (auto &prop: outputProperties) {
    sortedProps.push_back(prop.first);
  }
  sort(sortedProps.begin(), sortedProps.end(),
       [&](const tuple<Expr, TensorProperty, int, int> &a,
           const tuple<Expr, TensorProperty, int, int> &b) -> bool {
         // first, use a total order of outputs,inputs
         auto a_it = find(outputs.begin(), outputs.end(), get<0>(a));
         auto b_it = find(outputs.begin(), outputs.end(), get<0>(b));
         auto a_pos = distance(outputs.begin(), a_it);
         auto b_pos = distance(outputs.begin(), b_it);

         // if total order is same, have to do more, otherwise we know
         // our answer
         if (a_pos != b_pos)
           return a_pos < b_pos;

         // if they're different properties, sort by property
         if (get<1>(a) != get<1>(b))
           return get<1>(a) < get<1>(b);

         // now either the mode gives order, or index #
         if (get<2>(a) != get<2>(b))
           return get<2>(a) < get<2>(b);

         return get<3>(a) < get<3>(b);
       });

  for (auto prop: sortedProps) {
    auto var = get<0>(prop).as<Var>();
    if (!var->is_parameter) {
      ret << outputTensorProperty(outputProperties[prop], get<0>(prop),
                              get<1>(prop), get<2>(prop), get<3>(prop));
    }
  }
  return ret.str();
}

string CodeGen_Spatial::outputTensorProperty(string varname, Expr tnsr,
                                   TensorProperty property,
                                   int mode, int index) {
  stringstream ret;
  ret << "    ";

  auto tensor = tnsr.as<Var>();
  if (property == TensorProperty::Values) {
    ret << varname << "_dram store " << varname << endl;
    return ret.str();
  } else if (property == TensorProperty::Dimension) {
    return "";
  } 
  return ret.str();
}


} // namespace ir
} // namespace taco
