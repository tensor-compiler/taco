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
#include "taco/ir/ir_rewriter.h"
#include "taco/util/scopedmap.h"
#include "taco/ir/simplify.h"

#define DEBUG_PRINT 1

#if DEBUG_PRINT
#define debug_print(_x) std::cerr << _x
#else
#define debug_print(_x)
#endif

using namespace std;

namespace {
using namespace taco::ir;

// This pass is heavily inspired by the one in Halide.
struct VectorSubstitute : public IRRewriter {

  string name; // the variable we are vectorizing
  Expr replacement; // what we're replacing it with
  
  taco::util::ScopedMap<string, const Expr> varMap;
  
  VectorSubstitute(string name, Expr replacement) : name(name), replacement(replacement) {}
  
  Expr widen(Expr original, int lanes) {
    if (original.type().getNumLanes() == lanes) {
      // Already widened
      return original;
    } else {
      taco_iassert(original.type().getNumLanes() == 1)
        << "Attempting to widen an expression that is already widened "
        << " with a mismatched number of lanes: " << original << "\n";
      auto ret =  Broadcast::make(original, lanes);
      debug_print("widened "<< "(" << original.type() << ")" << original << " to " << (Expr)ret <<
                "with lanes=" << ret.type().getNumLanes() <<  "\n") ;
      return ret;
    }
  }
  
  void visit(const Var* op) {
    if (op->name == name) {
      debug_print("rewriting var to " << (Expr)replacement);
      debug_print(" with " << replacement.type().getNumLanes() << " lanes \n");
      expr = replacement;
    } else if (varMap.contains(op->name)
               && (varMap.get(op->name).type().getNumLanes() > op->type.getNumLanes())) {
      expr = varMap.get(op->name);
    } else {
      expr = op;
    }
  }
  
  template<typename T>
  const Expr rewrite_bin_op(const T* op) {
    debug_print("rewriting " << (Expr)op << "\n");
    debug_print(" types: " << op->a.type() << ", " << op->b.type() << "\n");
    auto a = rewrite(op->a);
    auto b = rewrite(op->b);
    debug_print(" a rewritten to " << a << " and b rewritten to " << b << "\n");
    if (a == op->a && b == op->b) {
      return op;
    } else {
      int lanes = std::max(a.type().getNumLanes(), b.type().getNumLanes());
      auto ret =  T::make(widen(a, lanes), widen(b, lanes));
      debug_print(" ret is " << (Expr)ret << " with lanes=" << ret.type().getNumLanes() << "\n");
      return ret;
    }
  }
  
  void visit(const Neg* op) {
    auto a = rewrite(op->a);
    if (a == op->a) {
      expr = op;
    } else {
      expr = Neg::make(a);
    }
  }
  
  void visit(const Literal* op) {
    expr = op;
  }
  
  void visit(const Sqrt* op) {
    auto a = rewrite(op->a);
    if (a == op->a) {
      expr = op;
    } else {
      expr = Neg::make(a);
    }
  }
  
  void visit(const Mul* op) {
    expr = rewrite_bin_op(op);
  }
  
  void visit(const Add* op) {
    expr = rewrite_bin_op(op);
  }
  
  void visit(const Sub* op) {
    expr = rewrite_bin_op(op);
  }
  
  void visit(const Div* op) {
    expr = rewrite_bin_op(op);
  }
  
  void visit(const Rem* op) {
    expr = rewrite_bin_op(op);
  }
  
  void visit(const BitAnd* op) {
    expr = rewrite_bin_op(op);
  }
  
  void visit(const BitOr* op) {
    rewrite_bin_op(op);
  }
  
  void visit(const Eq* op) {
    rewrite_bin_op(op);
  }
  
  void visit(const Neq* op) {
    rewrite_bin_op(op);
  }
  
  void visit(const Gt* op) {
    rewrite_bin_op(op);
  }
  
  void visit(const Lt* op) {
    rewrite_bin_op(op);
  }
  
  void visit(const Gte* op) {
    rewrite_bin_op(op);
  }
  
  void visit(const Lte* op) {
    rewrite_bin_op(op);
  }
  
  void visit(const And* op) {
    rewrite_bin_op(op);
  }
  
  void visit(const Or* op) {
    rewrite_bin_op(op);
  }
  
  // Temporarily ignore if conditions
  void visit(const IfThenElse* op) {
    stmt = Comment::make("Note: ignoring IfThenElse here");
  }
  
  void visit(const VarDecl* op) {
    debug_print("vardecl rewriting " << op->rhs << "\n");
    auto rhs = rewrite(op->rhs);
    if (rhs == op->rhs) {
      stmt = op;
    } else {
      debug_print("new type has lanes=" << rhs.type().getNumLanes() << "\n");
      auto var = op->var.as<Var>();
      debug_print("VAR: " << var->name << "\n");
      taco::Datatype dt(var->type.getKind(), rhs.type().getNumLanes());
      auto wide = Var::make(var->name + "_widened", dt);
      auto wide_var = wide.as<Var>();
      debug_print("WIDE VAR: " << (Expr)wide_var << "\n");
      taco_iassert(wide_var != nullptr);
      stmt = VarDecl::make(wide, rhs);
      // This is a bit subtle, but because we want to propagate
      // all ramps/broadcasts, we will replace instances of the var
      // with its _value_, not it's _name_
      varMap.insert({var->name, rhs});
      debug_print("  New var " << (Expr)wide_var  << " has " << wide_var->type.getNumLanes() << " lanes\n");
    }
  }
  
  void visit(const Load* op) {
    debug_print("rewriting " << (Expr)op << "\n");
    auto arr = rewrite(op->arr);
    auto loc = rewrite(op->loc);
    
    if (arr == op->arr && loc == op->loc) {
      expr = op;
    } else {
      int lanes = loc.type().getNumLanes();
      expr = Load::make(arr, loc, arr.type().with_lanes(lanes));
      debug_print("  rewrote to " << (Expr)expr << "with lanes=" << expr.as<Load>()->type.getNumLanes() << "\n");
    }
  }
  
  void visit(const Store* op) {
    auto arr = rewrite(op->arr);
    auto loc = rewrite(op->loc);
    auto data = rewrite(op->data);
    if (arr == op->arr &&
        loc == op->loc &&
        data == op->data) {
      stmt = op;
    } else {
      stmt = Store::make(arr, loc, data, op->use_atomics);
    }
  }
  
};

struct FunctionVectorizer : public IRRewriter {
  void visit(const For* op) {
    if (op->parallel_unit == taco::PARALLEL_UNIT::CPU_VECTOR) {
      debug_print("Before:\n" << (Stmt)(op) << "\n");
      //taco_iassert(op->end > 1 && (op->vec_width == 4 || op->vec_width == 8));
      auto ramp = Ramp::make(op->start, 1, op->end.as<Literal>()->getIntValue());
      VectorSubstitute vs(op->var.as<Var>()->name, ramp);
      auto new_contents = vs.rewrite(op->contents);
      debug_print("==========\nAfter:\n" << (Stmt)new_contents<< "\n");
      stmt = new_contents;
    } else {
      auto var = rewrite(op->var);
      auto start = rewrite(op->start);
      auto end = rewrite(op->end);
      auto increment = rewrite(op->increment);
      auto contents = rewrite(op->contents);
      if (var == op->var && start == op->start && end == op->end
          && increment == op->increment
          && contents == op->contents) {
        stmt = op;
      } else {
        stmt = For::make(var, start, end, increment, contents);
      }
    }
  }

};
Stmt vectorize(const Function* op) {
    FunctionVectorizer fs;
    return fs.rewrite((Stmt)op);
}
  

} // anonymous namespace

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
  "#include \"simd.h\"\n"
  "#include <immintrin.h>\n"
  "#ifndef SIMDPP_ARCH_X86_AVX2\n"
  "#warning Must be compiled with -DSIMDPP_ARCH_X86_AVX2\n"
  "#endif\n"
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
  "\n"
  "template<typename vecT, typename T>\n"
  "inline vecT populate_vec(T a0, T a1, T a2, T a3, T a4, T a5, T a6, T a7) {\n"
  "  SIMDPP_ALIGN(4) T tmp[8];\n"
  "  tmp[0] = a0;\n"
  "  tmp[1] = a1;\n"
  "  tmp[2] = a2;\n"
  "  tmp[3] = a3;\n"
  "  tmp[4] = a4;\n"
  "  tmp[5] = a5;\n"
  "  tmp[6] = a6;\n"
  "  tmp[7] = a7;\n"
  "  return simdpp::load<vecT>(tmp);\n"
  "}\n"
  "template<typename vecT, typename T>\n"
  "inline vecT populate_vec(T a0, T a1, T a2, T a3) {\n"
  "  SIMDPP_ALIGN(4) T tmp[4];\n"
  "  tmp[0] = a0;\n"
  "  tmp[1] = a1;\n"
  "  tmp[2] = a2;\n"
  "  tmp[3] = a3;\n"
  "  return simdpp::load<vecT>(tmp);\n"
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

CodeGen_C::CodeGen_C(std::ostream &dest, OutputKind outputKind, bool simplify)
    : CodeGen(dest, false, simplify, C), out(dest), outputKind(outputKind) {}

CodeGen_C::~CodeGen_C() {}

void CodeGen_C::compile(Stmt stmt, bool isFirst) {
  if (isFirst) {
    // output the headers
    out << cHeaders;
  }
  out << endl;

  debug_print("Generating code for: " << stmt << "\n");
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
  
  // insert using for simd library
  out << "using namespace simdpp;\n";

  // Transform vectorized loops
  Stmt funcStmt = vectorize(func);
  funcStmt = taco::ir::simplify(funcStmt);
  auto newFunc = funcStmt.as<Function>();
  debug_print("After vectorization pass: \n" << (Stmt)newFunc);
  

  indent++;

  // find all the vars that are not inputs or outputs and declare them
  resetUniqueNameCounters();
  FindVars varFinder(newFunc->inputs, newFunc->outputs, this);
  newFunc->body.accept(&varFinder);
  varMap = varFinder.varMap;
  localVars = varFinder.localVars;

  // Print variable declarations
  out << printDecls(varFinder.varDecls, newFunc->inputs, newFunc->outputs) << endl;

  if (emittingCoroutine) {
    out << printContextDeclAndInit(varMap, localVars, numYields, newFunc->name)
        << endl;
  }

  // output body
  print(newFunc->body);

  // output repack only if we allocated memory
  if (checkForAlloc(newFunc))
    out << endl << printPack(varFinder.outputProperties, newFunc->outputs);

  if (emittingCoroutine) {
    out << printCoroutineFinish(numYields, funcName);
  }

  doIndent();
  out << "return 0;\n";
  indent--;

  doIndent();
  out << "}\n";
}

namespace {
bool isStride1Ramp(Expr e) {
  const Ramp *ramp = e.as<Ramp>();
  if (ramp == nullptr) {
    return false;
  } else if (ramp->increment.as<Literal>() &&
             ramp->increment.as<Literal>()->equalsScalar(1)) {
    return true;
  } else {
    debug_print("Ramp has increment " << ramp->increment << "\n");
    return false;
  }
}

inline bool isStride1Var(Expr e, map<string, Expr> vars) {
  return (e.as<Var>() && vars.count(e.as<Var>()->name));
}

inline Expr getStride1RampBase(Expr e, map<string, Expr> vars) {
  auto ramp = e.as<Ramp>();
  if (ramp == nullptr) {
    ramp = vars[e.as<Var>()->name].as<Ramp>();
  }
  return ramp->value;
}
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
  // TODO: properly scope these!
  if (isStride1Ramp(op->rhs)) {
    stride_1_ramp_vars[op->var.as<Var>()->name] = op->rhs;
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

static string getAtomicPragma() {
  return "#pragma omp atomic";
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
      //out << genVectorizePragma(op->vec_width);
      //out << "\n";
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

void CodeGen_C::visit(const Assign* op) {
  if (op->use_atomics) {
    doIndent();
    stream << getAtomicPragma() << endl;
  }
  IRPrinter::visit(op);
}

namespace {
// We can only handle reductions of the form:
// a[scalar] = _broadcast(a[scalar]) + ...
// TODO: expand the set of supported reductions
bool check_if_supported_reduction(const Store* op) {
  return (op->data.as<Add>() &&
          op->data.as<Add>()->a.as<Broadcast>() &&
          op->data.as<Add>()->a.as<Broadcast>()->value.as<Load>() &&
          op->data.as<Add>()->a.as<Broadcast>()->value.as<Load>()->arr == op->arr &&
          op->data.as<Add>()->a.as<Broadcast>()->value.as<Load>()->loc == op->loc);
}

}

void CodeGen_C::visit(const Store* op) {
  if (op->use_atomics) {
    doIndent();
    stream << getAtomicPragma() << endl;
  }
  debug_print("In Store " << (Stmt)op << "\n");
  if (op->data.type().getNumLanes() == 1) {
    taco_iassert(op->loc.as<Ramp>() == nullptr) <<
        "Unhandled store: storing a scalar value " <<
        op->data << " to a vector location.";
    IRPrinter::visit(op);
    return;
  } else {
    // if the stride is 1, and it's a ramp location,
    // we're golden
    if (isStride1Ramp(op->loc) ||
        isStride1Var(op->loc, stride_1_ramp_vars)) {
      doIndent();
      // TODO: alignment :(
      stream << "store_u(";
      op->arr.accept(this);
      stream << " + ";
      //op->loc.accept(this);
      getStride1RampBase(op->loc, stride_1_ramp_vars).accept(this);
      stream << ", ";
      op->data.accept(this);
      stream << ");";
    } else if (check_if_supported_reduction(op)) {

      doIndent();
      op->arr.accept(this);
      stream << "[";
      op->loc.accept(this);
      stream << "] = ";
      // We're going to remove the broadcast from the reduction location
      // so something like "a[i] = _broadcast(a[i], lanes) + ..."
      // becomes "a[i] = a[i] + reduce_add(...)"
      // TODO: introduce explicit reduction IR?
      auto add = op->data.as<Add>();
      auto lhs_bcast = add->a.as<Broadcast>();
      stream << "(";
      lhs_bcast->value.accept(this);
      stream << " + ";
      stream << "reduce_add(";
      add->b.accept(this);
      stream << "))";
      stream << ";";
      stream << endl;
    
    } else {
      taco_tassert(false) << "Unhandled vector store";
    }
  }
}

void CodeGen_C::visit(const Broadcast* op) {
  stream << "splat<";
  stream << op->type << ">(";
  op->value.accept(this);
  stream << ")";
}

void CodeGen_C::visit(const Ramp* op) {
  taco_tassert(op->type.getNumLanes() == 8 || op->type.getNumLanes() == 4);
  stream << "populate_vec<" << op->type << ",";
  stream << op->type.with_lanes(1) << ">(";
  for (int i=0; i<op->lanes; i++) {
    stream << "(";
    op->value.accept(this);
    stream << " + (" << i << " * ";
    op->increment.accept(this);
    stream << "))";
    if (i != op->lanes-1) {
      stream << ", ";
    }
  }
  stream << ")";
  
}

void CodeGen_C::visit(const Load* op) {
  debug_print("In Load for " << (Expr)op << "\n");
  if (op->type.getNumLanes() == 1) {
    IRPrinter::visit(op);
  } else {
    // if the stride is 1, we're golden
    if (isStride1Ramp(op->loc) ||
        isStride1Var(op->loc, stride_1_ramp_vars)) {
      // TODO: ensure alignment
      stream << "load_u<" << op->type << ">((";
      op->arr.accept(this);
      stream << ") + ";
      getStride1RampBase(op->loc, stride_1_ramp_vars).accept(this);
      stream << ")";
    } else {
      // We have a load of a non-ramp, so we'll use a gather
      // TODO: optimize this & make more robust
      taco_tassert(op->loc.type().getNumLanes() == 4);
      taco_tassert(op->type.isFloat() && op->type.getNumBits() == 64);
      taco_tassert(op->loc.type().isInt() && op->loc.type().getNumBits() == 32);
      stream << op->type << "(";
      stream << "_mm256_i32gather_pd(";
      op->arr.accept(this);
      stream << ", (";
      op->loc.accept(this);
      stream << ").eval().native(), 8)";
      stream << ")"; // type(..)
      //taco_tassert(false) << "Unhandled vector load";
    }
  }
}

void CodeGen_C::generateShim(const Stmt& func, stringstream &ret) {
  const Function *funcPtr = func.as<Function>();

  ret << "extern \"C\" int _shim_" << funcPtr->name << "(void** parameterPack) {\n";
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
}
}
