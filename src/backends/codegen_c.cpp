#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <algorithm>
#include <unordered_set>

#include "ir/ir_visitor.h"
#include "codegen_c.h"
#include "taco/error.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace ir {

// Some helper functions
namespace {

// Include stdio.h for printf
// stdlib.h for malloc/realloc
// math.h for sqrt
// MIN preprocessor macro
const string cHeaders = "#ifndef TACO_C_HEADERS\n"
                 "#define TACO_C_HEADERS\n"
                 "#include <stdio.h>\n"
                 "#include <stdlib.h>\n"
                 "#include <math.h>\n"
                 "#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))\n"
                 "#ifndef TACO_TENSOR_T_DEFINED\n"
                 "#define TACO_TENSOR_T_DEFINED\n"
                 "typedef enum { taco_dim_dense, taco_dim_sparse } taco_dim_t;\n"
                 "\n"
                 "typedef struct {\n"
                 "  int32_t     order;      // tensor order (number of dimensions)\n"
                 "  int32_t*    dims;       // tensor dimensions\n"
                 "  taco_dim_t* dim_types;  // dimension storage types\n"
                 "  int32_t     csize;      // component size\n"
                 "\n"
                 "  int32_t*    dim_order;  // dimension storage order\n"
                 "  uint8_t***   indices;   // tensor index data (per dimension)\n"
                 "  uint8_t*    vals;       // tensor values\n"
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
  
  // this maps from tensor, property, dim to the unique var
  map<tuple<Expr, TensorProperty, int>, string> canonicalPropertyVar;
  
  // this is for convenience, recording just the properties unpacked
  // from the output tensor so we can re-save them at the end
  map<tuple<Expr, TensorProperty, int>, string> outputProperties;
  
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
      stringstream name;
      auto tensor = op->tensor.as<Var>();
      //name << "__" << tensor->name << "_";
      name << tensor->name;
      if (op->property == TensorProperty::Values) {
        name << "_vals";
    } else {
      name << "_L" << op->dim;
      if (op->property == TensorProperty::Index)
        name << "_idx";
      if (op->property == TensorProperty::Pointer)
        name << "_pos";
    }
    auto key = tuple<Expr, TensorProperty, int>
                    (op->tensor, op->property, op->dim);
    if (canonicalPropertyVar.count(key) > 0) {
      varMap[op] = canonicalPropertyVar[key];
    } else {
      auto unique_name = CodeGen_C::genUniqueName(name.str());
      canonicalPropertyVar[key] = unique_name;
      varMap[op] = unique_name;
      varDecls[op] = unique_name;
      if (find(outputTensors.begin(), outputTensors.end(), op->tensor)
          != outputTensors.end()) {
        outputProperties[key] = unique_name;
      }
    }
  }
 }
};



// helper to translate from taco type to C type
string toCType(Type type, bool is_ptr) {
  string ret;

  switch (type.kind) {
    case Type::Int:
      ret = "int"; //TODO: should use a specific width here
      break;
    case Type::UInt:
      break;
    case Type::Float:
      if (type.bits == 32) {
        ret = "float";
      }
      else if (type.bits == 64) {
        ret = "double";
      }
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

// helper to count # of slots for a format
int formatSlots(Format format) {
  int i = 0;
  for (auto level : format.getLevels()) {
    if (level.getType() == DimensionType::Dense)
      i += 1;
    else
      i += 2;
  }
  i += 1; // for the vals
  return i;
}


// generate the unpack of a specific property (internal calling interface)
string unpackTensorPropertyInternal(string varname, const GetProperty* op,
                            bool is_output_prop) {
  stringstream ret;
  ret << "  ";
  
  auto tensor = op->tensor.as<Var>();
  if (op->property == TensorProperty::Values) {
    // for the values, it's in the last slot
    ret << toCType(tensor->type, true);
    ret << " restrict " << varname << " = ";
    ret << tensor->name << "[" << formatSlots(tensor->format)-1 << "];\n";
    return ret.str();
  }
  auto levels = tensor->format.getLevels();
  
  taco_iassert(op->dim < levels.size())
    << "Trying to access a nonexistent dimension";
  
  int slot = 0;
  string tp;
  
  for (size_t i=0; i < op->dim; i++) {
    if (levels[i].getType() == DimensionType::Dense)
      slot += 1;
    else
      slot += 2;
  }
  
  // for this level, if the property is index, we add 1
  if (op->property == TensorProperty::Index)
    slot += 1;
  
  // for a Dense level, nnz is an int
  // for a Fixed level, ptr is an int
  // all others are int*
  if ((levels[op->dim].getType() == DimensionType::Dense &&
       op->property == TensorProperty::Pointer) ||
      (levels[op->dim].getType() == DimensionType::Fixed &&
       op->property == TensorProperty::Pointer)) {
    tp = "int";
    ret << tp << " " << varname << " = *(" << tp << "*)" <<
      tensor->name << "[" << slot << "];\n";
  } else {
    tp = "int*";
    ret << tp << " restrict " << varname << " = ";
    ret << "(" << tp << ")" <<
      tensor->name << "[" << slot << "];\n";
  }
  
  return ret.str();
}

string unpackTensorPropertyNormal(string varname, const GetProperty* op,
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
  auto levels = tensor->format.getLevels();
  
  taco_iassert(op->dim < levels.size())
    << "Trying to access a nonexistent dimension";
  
  string tp;
  
  // for a Dense level, nnz is an int
  // for a Fixed level, ptr is an int
  // all others are int*
  if ((levels[op->dim].getType() == LevelType::Dense &&
      op->property == TensorProperty::Pointer)
      ||(levels[op->dim].getType() == LevelType::Fixed &&
      op->property == TensorProperty::Pointer)) {
    tp = "int";
    ret << tp << " " << varname << " = *(" <<
      tensor->name << "->indices[" << op->dim << "][0]);\n";
  } else {
    tp = "int*";
    auto nm = op->property == TensorProperty::Pointer ? "[0]" : "[1]";
    ret << tp << " restrict " << varname << " = ";
    ret << "(int*)(" << tensor->name << "->indices[" << op->dim;
    ret << "]" << nm << ");\n";
  }
  
  return ret.str();
}


// generate the unpack of a specific property
string unpackTensorProperty(string varname, const GetProperty* op,
                            bool is_output_prop,
                            CodeGen_C::InterfaceKind interface) {
  if (interface == CodeGen_C::InterfaceKind::Internal)
    return unpackTensorPropertyInternal(varname, op, is_output_prop);
  else
    return unpackTensorPropertyNormal(varname, op, is_output_prop);
}


string packTensorPropertyInternal(string varname, Expr tnsr, TensorProperty property,
  int dim) {
  stringstream ret;
  ret << "  ";
  
  auto tensor = tnsr.as<Var>();
  if (property == TensorProperty::Values) {
    // for the values, it's in the last slot
    ret << "((double**)" << tensor->name << ")["
        << formatSlots(tensor->format)-1 << "] ";
    ret << " = " << varname << ";\n";
    return ret.str();
  }
  auto levels = tensor->format.getLevels();
  
  taco_iassert(dim < (int)levels.size())
    << "Trying to access a nonexistent dimension";
  
  int slot = 0;
  string tp;
  
  for (int i=0; i<dim; i++) {
    if (levels[i].getType() == DimensionType::Dense)
      slot += 1;
    else
      slot += 2;
  }
  
  // for this level, if the property is index, we add 1
  if (property == TensorProperty::Index)
    slot += 1;
  
  // for a Dense level, nnz is an int
  // for a Fixed level, ptr is an int
  // all others are int*
  if ((levels[dim].getType() == DimensionType::Dense &&
      property == TensorProperty::Pointer)
      ||(levels[dim].getType() == DimensionType::Fixed &&
      property == TensorProperty::Pointer)) {
    tp = "int";
    ret << "*(" << tp << "*)" <<
      tensor->name << "[" << slot << "] = " <<
      varname << ";\n";
  } else {
    tp = "int*";
    ret << "((int**)" << tensor->name
        << ")[" << slot << "] = (" << tp << ")"<< varname
      << ";\n";
  }
  
  return ret.str();
}

string packTensorPropertyNormal(string varname, Expr tnsr, TensorProperty property,
  int dim) {
  stringstream ret;
  ret << "  ";
  
  auto tensor = tnsr.as<Var>();
  if (property == TensorProperty::Values) {
    ret << tensor->name << "->vals";
    ret << " = (uint8_t*)" << varname << ";\n";
    return ret.str();
  }
  auto levels = tensor->format.getLevels();
  
  taco_iassert(dim < (int)levels.size())
    << "Trying to access a nonexistent dimension";
  
  string tp;
  
  // for a Dense level, nnz is an int
  // for a Fixed level, ptr is an int
  // all others are int*
  if ((levels[dim].getType() == LevelType::Dense &&
      property == TensorProperty::Pointer)
      ||(levels[dim].getType() == LevelType::Fixed &&
      property == TensorProperty::Pointer)) {
    tp = "int";
    ret << tensor->name << "->indices[" << dim << "][0] = " <<
      "(uint8_t*)("<< varname << ");\n";
  } else {
    tp = "int*";
    auto nm = property == TensorProperty::Pointer ? "[0]" : "[1]";

    ret << tensor->name << "->indices" <<
      "[" << dim << "]" << nm << " = (uint8_t*)(" << varname
      << ");\n";
  }
  
  return ret.str();
}


string packTensorProperty(string varname, Expr tnsr, TensorProperty property,
  int dim, CodeGen_C::InterfaceKind interface) {
  if (interface == CodeGen_C::InterfaceKind::Internal)
    return packTensorPropertyInternal(varname, tnsr, property, dim);
  else
    return packTensorPropertyNormal(varname, tnsr, property, dim);
}
  
// helper to print declarations
string printDecls(map<Expr, string, ExprCompare> varMap,
                   map<tuple<Expr, TensorProperty, int>, string> uniqueProps,
                   vector<Expr> inputs, vector<Expr> outputs,
                   CodeGen_C::InterfaceKind interface) {
  stringstream ret;
  unordered_set<string> propsAlreadyGenerated;
  
  for (auto varpair: varMap) {
    // make sure it's not an input or output
    if (find(inputs.begin(), inputs.end(), varpair.first) == inputs.end() &&
        find(outputs.begin(), outputs.end(), varpair.first) == outputs.end()) {
      auto var = varpair.first.as<Var>();
      if (var) {
        ret << "  " << toCType(var->type, var->is_ptr);
        ret << " " << varpair.second << ";\n";
      } else {
        auto prop = varpair.first.as<GetProperty>();
        taco_iassert(prop);
        if (!propsAlreadyGenerated.count(varpair.second)) {
          // there is an extra deref for output properties, since
          // they are passed by reference
          bool isOutputProp = (find(outputs.begin(), outputs.end(),
                                    prop->tensor) != outputs.end());
          ret << unpackTensorProperty(varpair.second, prop, isOutputProp,
                                      interface);
          propsAlreadyGenerated.insert(varpair.second);
        }
      }
    }
  }

  return ret.str();
}



// helper to unpack inputs and outputs
// inputs are unpacked to a pointer
// outputs are unpacked to a pointer
// TODO: this will change for tensors
string printUnpack(vector<Expr> inputs, vector<Expr> outputs,
                   CodeGen_C::InterfaceKind interface) {
  
  // when using the non-internal interface, we don't need to unpack
  // anything, because the tensors are named parameters
  if (interface == CodeGen_C::InterfaceKind::Normal)
    return "";
  
  stringstream ret;
  int slot = 0;
  
  for (auto output: outputs) {
    auto var = output.as<Var>();
    if (!var->is_tensor) {

      taco_iassert(var->is_ptr) << "Function outputs must be pointers";

      auto tp = toCType(var->type, var->is_ptr);
      ret << "  " << tp << " " << var->name << " = (" << tp << ")inputPack["
        << slot++ << "];\n";
    } else {
      ret << "  void** " << var->name << " = &(inputPack[" << slot << "]);\n";
      slot += formatSlots(var->format);
    }
  }

  
  for (auto input: inputs) {
    auto var = input.as<Var>();
    if (!var->is_tensor) {
      auto tp = toCType(var->type, var->is_ptr);
      // if the input is not of non-pointer type, we should unpack it
      // here
      auto deref = var->is_ptr ? "" : "*";
      ret << "  " << tp << " " << var->name;
      ret << " = " << deref << "(" << tp << deref << ")inputPack["
        << slot++ << "];\n";
    } else {
      ret << "  void** " << var->name << " = &(inputPack[" << slot << "]);\n";
      slot += formatSlots(var->format);
    }
    
  }
  
  return ret.str();
}

string printPack(map<tuple<Expr, TensorProperty, int>,
                 string> outputProperties, CodeGen_C::InterfaceKind interface) {
  stringstream ret;
  for (auto prop: outputProperties) {
    ret << packTensorProperty(prop.second, get<0>(prop.first),
      get<1>(prop.first), get<2>(prop.first), interface);
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

string printFuncName(const Function *func, CodeGen_C::InterfaceKind interface) {
  stringstream ret;
  
  ret << "int " << func->name << "(";
  
  if (interface == CodeGen_C::InterfaceKind::Internal) {
    ret << "void** inputPack";
  } else {
    for (auto p : func->outputs) {
      auto var = p.as<Var>();
      taco_iassert(var) << "Unable to convert output " << p << " to Var";
      if (var->is_tensor) {
        ret << "taco_tensor_t *" << var->name << ", ";
      } else {
        auto tp = toCType(var->type, var->is_ptr);
        ret << tp << " " << var->name << ", ";
      }
    }
    for (size_t i=0; i<func->inputs.size(); i++) {
      auto var = func->inputs[i].as<Var>();
      taco_iassert(var) << "Unable to convert output " << func->inputs[i]
        << " to Var";
      if (var->is_tensor) {
        ret << "taco_tensor_t *" << var->name;
      } else {
        auto tp = toCType(var->type, var->is_ptr);
        ret << tp << " " << var->name;
      }
      if (i < func->inputs.size() - 1)
        ret << ", ";
    }
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

void CodeGen_C::compile(Stmt stmt, bool isFirst, InterfaceKind interfaceKind) {
  this->interfaceKind = interfaceKind;
  if (isFirst && outputKind == C99Implementation) {
    // output the headers
    out << cHeaders;
  }
  out << endl;
  // generate code for the Stmt
  stmt.accept(this);
}

static bool hasStore(Stmt stmt) {
  struct StoreFinder : public IRVisitor {
    using IRVisitor::visit;
    bool hasStore = false;
    void visit(const Store*) {
      hasStore = true;
    }
  };
  StoreFinder storeFinder;
  stmt.accept(&storeFinder);
  return storeFinder.hasStore;
}

void CodeGen_C::visit(const Function* func) {
  // if generating a header, protect the function declaration with a guard
  if (outputKind == C99Header) {
    out << "#ifndef TACO_GENERATED_" << func->name << "\n";
    out << "#define TACO_GENERATED_" << func->name << "\n";
  }

  // output function declaration
  doIndent();
  out << printFuncName(func, interfaceKind) << "\n";
  
  std::cout << printFuncName(func, InterfaceKind::Normal) << "\n";

  // if we're just generating a header, this is all we need to do
  if (outputKind == C99Header) {
    out << ";\n";
    out << "#endif\n";
    return;
  }

  out << "\n{\n";

  // input/output unpack
  out << printUnpack(func->inputs, func->outputs, interfaceKind);

  indent++;

  // Don't print bodies that don't do anything (e.g. assemble functions when
  // the result is dense.
  if (hasStore(func->body)) {
    // find all the vars that are not inputs or outputs and declare them
    resetUniqueNameCounters();
    FindVars varFinder(func->inputs, func->outputs);
    func->body.accept(&varFinder);
    varMap = varFinder.varMap;

    // Print variable declarations
    out << printDecls(varFinder.varDecls, varFinder.canonicalPropertyVar,
                      func->inputs, func->outputs, interfaceKind);

    // output body
    out << endl;
    print(func->body);
    out << endl;

    out << "\n";
    // output repack
    out << printPack(varFinder.outputProperties, interfaceKind);
  }

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

static string getParallelizePragma() {
  stringstream ret;
  ret << "#pragma omp parallel for";
  return ret.str();
}

// The next two need to output the correct pragmas depending
// on the loop kind (Serial, Parallel, Vectorized)
//
// Docs for vectorization pragmas:
// http://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-loop-hint-optimizations
void CodeGen_C::visit(const For* op) {
  if (op->kind == LoopKind::Vectorized) {
    doIndent();
    out << genVectorizePragma(op->vec_width);
    out << "\n";
  }

  if (op->kind == LoopKind::Parallel) {
    doIndent();
    out << getParallelizePragma();
    out << "\n";
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
  taco_iassert(varMap.count(op) > 0) << "Property of "
      << op->tensor << " not found in varMap";

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
  taco_tassert(op->type.isFloat() && op->type.bits == 64) <<
      "Codegen doesn't currently support non-double sqrt";
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}


} // namespace ir
} // namespace taco
