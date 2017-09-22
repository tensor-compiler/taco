#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <algorithm>
#include <unordered_set>

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
  "#include <math.h>\n"
  "#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))\n"
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
  "} taco_tensor_t;\n"
  "#endif\n"
  "#endif\n";

// find variables for generating declarations
// also only generates a single var for each GetProperty
class FindVars : public IRVisitor {
public:
  map<Expr, string, ExprCompare> varMap;
  
  // the variables for which we need to add declarations
  map<Expr, string, ExprCompare> varDecls;
  
  // this maps from tensor, property, mode, index to the unique var
  map<tuple<Expr, TensorProperty, int, int>, string> canonicalPropertyVar;
  
  // this is for convenience, recording just the properties unpacked
  // from the output tensor so we can re-save them at the end
  map<tuple<Expr, TensorProperty, int, int>, string> outputProperties;
  
  // TODO: should replace this with an unordered set
  vector<Expr> outputTensors;
  
  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs)  {
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
    inVarAssignLHSWithDecl = false;
  }

protected:
  bool inVarAssignLHSWithDecl;
  using IRVisitor::visit;

  virtual void visit(const For *op) {
    // Don't need to find/initialize loop bounds
    inVarAssignLHSWithDecl = true;
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    inVarAssignLHSWithDecl = false;

    op->contents.accept(this);
  }

  virtual void visit(const Var *op) {
    if (varMap.count(op) == 0) {
      varMap[op] = CodeGen_C::genUniqueName(op->name);
      if (!inVarAssignLHSWithDecl) {
        varDecls[op] = varMap[op];
      }
    }
  }
  
  virtual void visit(const VarAssign *op) {
    if (op->is_decl)
      inVarAssignLHSWithDecl = true;
    
    op->lhs.accept(this);
    
    if (op->is_decl)
      inVarAssignLHSWithDecl = false;
    
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
        auto unique_name = CodeGen_C::genUniqueName(op->name);
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


// helper to translate from taco type to C type
string toCType(Type type, bool is_ptr) {
  string ret;

  switch (type.getKind()) {
    case Type::Bool:
      ret = "bool";
      break;
    case Type::Int:
      ret = "int"; //TODO: should use a specific width here
      break;
    case Type::UInt:
      break;
    case Type::Float:
      if (type.getNumBits() == 32) {
        ret = "float";
      }
      else if (type.getNumBits() == 64) {
        ret = "double";
      }
      break;
    case Type::Undefined:
      taco_ierror << "undefined type in codegen";
      break;
  }
  if (ret == "") {
    taco_iassert(false) << "Unknown type in codegen";
  }

  if (is_ptr) {
    ret += "*";
  }
  
  return ret;
}

string unpackTensorProperty(string varname, const GetProperty* op,
                            bool is_output_prop) {
  stringstream ret;
  ret << "  ";
  
  auto tensor = op->tensor.as<Var>();
  if (op->property == TensorProperty::Values) {
    // for the values, it's in the last slot
    ret << toCType(tensor->type, true);
    ret << " restrict " << varname << " = (double*)(";
    ret << tensor->name << "->vals);\n";
    return ret.str();
  }
  
  taco_iassert((size_t)op->mode < tensor->format.getOrder()) <<
      "Trying to access a nonexistent mode";
  
  string tp;
  
  // for a Dense level, nnz is an int
  // for a Fixed level, ptr is an int
  // all others are int*
  if ((tensor->format.getModeTypes()[op->mode] == ModeType::Dense &&
       op->property == TensorProperty::Dimension) ||
      (tensor->format.getModeTypes()[op->mode] == ModeType::Fixed &&
       op->property == TensorProperty::Dimension)) {
    tp = "int";
    ret << tp << " " << varname << " = *(int*)("
        << tensor->name << "->indices[" << op->mode << "][0]);\n";
  } else {
    tp = "int*";
    auto nm = op->index;
    ret << tp << " restrict " << varname << " = ";
    ret << "(int*)(" << tensor->name << "->indices[" << op->mode;
    ret << "][" << nm << "]);\n";
  }
  
  return ret.str();
}

string packTensorProperty(string varname, Expr tnsr, TensorProperty property,
                          int mode, int index) {
  stringstream ret;
  ret << "  ";
  
  auto tensor = tnsr.as<Var>();
  if (property == TensorProperty::Values) {
    ret << tensor->name << "->vals";
    ret << " = (uint8_t*)" << varname << ";\n";
    return ret.str();
  }
  
  taco_iassert(mode < (int)tensor->format.getOrder()) <<
      "Trying to access a nonexistent mode";
  
  string tp;
  
  // for a Dense level, nnz is an int
  // for a Fixed level, ptr is an int
  // all others are int*
  if ((tensor->format.getModeTypes()[mode] == ModeType::Dense &&
       property == TensorProperty::Dimension) ||
      (tensor->format.getModeTypes()[mode] == ModeType::Fixed &&
       property == TensorProperty::Dimension)) {
    return "";
  } else {
    tp = "int*";
    auto nm = index;
    ret << tensor->name << "->indices" <<
      "[" << mode << "][" << nm << "] = (uint8_t*)(" << varname
      << ");\n";
  }
  
  return ret.str();
}

  
// helper to print declarations
string printDecls(map<Expr, string, ExprCompare> varMap,
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
    ret << unpackTensorProperty(varMap[prop], prop, isOutputProp);
    propsAlreadyGenerated.insert(varMap[prop]);
  }

  for (auto varpair: varMap) {
    // make sure it's not an input or output
    if (find(inputs.begin(), inputs.end(), varpair.first) == inputs.end() &&
        find(outputs.begin(), outputs.end(), varpair.first) == outputs.end()) {
      auto var = varpair.first.as<Var>();
      if (var) {
        ret << "  " << toCType(var->type, var->is_ptr);
        ret << " " << varpair.second << ";\n";
      } else {
        taco_iassert(varpair.first.as<GetProperty>());
        // we better have already generated these
        taco_iassert(propsAlreadyGenerated.count(varpair.second));
      }
    }
  }

  return ret.str();
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

string printPack(map<tuple<Expr, TensorProperty, int, int>,
                 string> outputProperties,
                 vector<Expr> outputs) {
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
    ret << packTensorProperty(outputProperties[prop], get<0>(prop),
      get<1>(prop), get<2>(prop), get<3>(prop));
  }
  return ret.str();
}

// seed the unique names with all C99 keywords
// from: http://en.cppreference.com/w/c/keyword
map<string, int> uniqueNameCounters;

void resetUniqueNameCounters() {
  uniqueNameCounters =
    {{"auto", 0},
     {"break", 0},
     {"case", 0},
     {"char", 0},
     {"const", 0},
     {"continue", 0},
     {"default", 0},
     {"do", 0},
     {"double", 0},
     {"else", 0},
     {"enum", 0},
     {"extern", 0},
     {"float", 0},
     {"for", 0},
     {"goto", 0},
     {"if", 0},
     {"inline", 0},
     {"int", 0},
     {"long", 0},
     {"register", 0},
     {"restrict", 0},
     {"return", 0},
     {"short", 0},
     {"signed", 0},
     {"sizeof", 0},
     {"static", 0},
     {"struct", 0},
     {"switch", 0},
     {"typedef", 0},
     {"union", 0},
     {"unsigned", 0},
     {"void", 0},
     {"volatile", 0},
     {"while", 0},
     {"bool", 0},
     {"complex", 0},
     {"imaginary", 0}};
}

string printFuncName(const Function *func) {
  stringstream ret;
  
  ret << "int " << func->name << "(";

  string delimiter = "";
  for (size_t i=0; i<func->outputs.size(); i++) {
    auto var = func->outputs[i].as<Var>();
    taco_iassert(var) << "Unable to convert output " << func->outputs[i]
      << " to Var";
    if (var->is_tensor) {
      ret << delimiter << "taco_tensor_t *" << var->name;
    } else {
      auto tp = toCType(var->type, var->is_ptr);
      ret << delimiter << tp << " " << var->name;
    }
    delimiter = ", ";
  }
  for (size_t i=0; i<func->inputs.size(); i++) {
    auto var = func->inputs[i].as<Var>();
    taco_iassert(var) << "Unable to convert output " << func->inputs[i]
      << " to Var";
    if (var->is_tensor) {
      ret << delimiter << "taco_tensor_t *" << var->name;
    } else {
      auto tp = toCType(var->type, var->is_ptr);
      ret << delimiter << tp << " " << var->name;
    }
    delimiter = ", ";
  }
  
  ret << ")";
  return ret.str();
}
  

} // anonymous namespace


string CodeGen_C::genUniqueName(string name) {
  stringstream os;
  os << name;
  if (uniqueNameCounters.count(name) > 0) {
    os << uniqueNameCounters[name]++;
  } else {
    uniqueNameCounters[name] = 0;
  }
  return os.str();
}

CodeGen_C::CodeGen_C(std::ostream &dest, OutputKind outputKind)
    : IRPrinter(dest, false, true), out(dest), outputKind(outputKind) {}

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
  if (outputKind == C99Header) {
    out << "#ifndef TACO_GENERATED_" << func->name << "\n";
    out << "#define TACO_GENERATED_" << func->name << "\n";
  }

  // output function declaration
  doIndent();
  out << printFuncName(func);
  
  // if we're just generating a header, this is all we need to do
  if (outputKind == C99Header) {
    out << ";\n";
    out << "#endif\n";
    return;
  }

  out << " {\n";

  indent++;

  // find all the vars that are not inputs or outputs and declare them
  resetUniqueNameCounters();
  FindVars varFinder(func->inputs, func->outputs);
  func->body.accept(&varFinder);
  varMap = varFinder.varMap;

  // Print variable declarations
  out << printDecls(varFinder.varDecls,
                    func->inputs, func->outputs);

  // output body
  out << endl;
  print(func->body);
  out << endl;

  out << "\n";
  
  // output repack only if we allocated memory
  CheckForAlloc allocChecker;
  func->accept(&allocChecker);
  if (allocChecker.hasAlloc)
    out << printPack(varFinder.outputProperties,
                     func->outputs);
  
  doIndent();
  out << "return 0;\n";
  indent--;

  doIndent();
  out << "}\n";
}

// For Vars, we replace their names with the generated name,
// since we match by reference (not name)
void CodeGen_C::visit(const Var* op) {
  taco_iassert(varMap.count(op) > 0) <<
      "Var " << op->name << " not found in varMap";
  out << varMap[op];
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
void CodeGen_C::visit(const For* op) {
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
  
  IRPrinter::visit(op);
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
      "Property of " << op->tensor << " not found in varMap";
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

void CodeGen_C::visit(const Allocate* op) {
  string elementType = toCType(op->var.type(), false);

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
  for (auto output : funcPtr->outputs) {
    auto var = output.as<Var>();
    auto cast_type = var->is_tensor ? "taco_tensor_t*"
    : toCType(var->type, var->is_ptr);
    
    ret << delimiter << "(" << cast_type << ")(parameterPack[" << i++ << "])";
    delimiter = ", ";
  }
  for (auto input : funcPtr->inputs) {
    auto var = input.as<Var>();
    auto cast_type = var->is_tensor ? "taco_tensor_t*"
    : toCType(var->type, var->is_ptr);
    ret << delimiter << "(" << cast_type << ")(parameterPack[" << i++ << "])";
    delimiter = ", ";
  }
  ret << ");\n";
  ret << "}\n";
}


} // namespace ir
} // namespace taco
