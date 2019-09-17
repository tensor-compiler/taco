#include "codegen.h"
#include "taco/cuda.h"
#include "codegen_cuda.h"
#include "codegen_c.h"
#include <algorithm>
#include <unordered_set>

using namespace std;

namespace taco {
namespace ir {

const std::string ctxName = "__ctx__";
const std::string coordsName = "__coords__";
const std::string bufCapacityName = "__bufcap__";
const std::string valName = "__val__";
const std::string ctxClassName = "___context___";
const std::string sizeName = "size";
const std::string stateName = "state";
const std::string bufSizeName = "__bufsize__";
const std::string bufCapacityCopyName = "__bufcapcopy__";
const std::string labelPrefix = "resume_";


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



bool CodeGen::checkForAlloc(const Function *func) {
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
  CheckForAlloc checker;
  func->accept(&checker);
  return checker.hasAlloc;
}

// helper to translate from taco type to C type
string CodeGen::printCType(Datatype type, bool is_ptr) {
  stringstream ret;
  ret << type;

  if (is_ptr) {
    ret << "*";
  }
  return ret.str();
}

// helper to translate from taco type to CUDA type
string CodeGen::printCUDAType(Datatype type, bool is_ptr) {
  if (type.isComplex()) {
    stringstream ret;
    if (type.getKind() == Complex64) {
      ret << "thrust::complex<float>";
    }
    else if (type.getKind() == Complex128) {
      ret << "thrust::complex<double>";
    }
    else {
      taco_ierror;
    }

    if(is_ptr) {
      ret << "*";
    }

    return ret.str();
  }
  return CodeGen::printCType(type, is_ptr);
}

string CodeGen::printType(Datatype type, bool is_ptr) {
  switch (codeGenType) {
    case C:
      return printCType(type, is_ptr);
    case CUDA:
      return printCUDAType(type, is_ptr);
    default:
      taco_ierror;
  }
  return "";
}

string CodeGen::printCAlloc(string pointer, string size) {
  return pointer + " = malloc(" + size + ");";
}

string CodeGen::printCUDAAlloc(string pointer, string size) {
  return "gpuErrchk(cudaMallocManaged((void**) &" + pointer + ", " + size + "));";
}

string CodeGen::printAlloc(string pointer, string size) {
  switch (codeGenType) {
    case C:
      return printCAlloc(pointer, size);
    case CUDA:
      return printCUDAAlloc(pointer, size);
    default:
      taco_ierror;
  }
  return "";
}

string CodeGen::printCFree(string pointer) {
  return "free(" + pointer + ");";
}

string CodeGen::printCUDAFree(string pointer) {
  return "cudaFree(" + pointer + ");";
}

string CodeGen::printFree(string pointer) {
  switch (codeGenType) {
    case C:
      return printCFree(pointer);
    case CUDA:
      return printCUDAFree(pointer);
    default:
      taco_ierror;
  }
  return "";
}

string CodeGen::printContextDeclAndInit(map<Expr, string, ExprCompare> varMap,
                               vector<Expr> localVars, int labels,
                               string funcName) {
  stringstream ret;

  ret << "  typedef struct " << ctxClassName << "{" << endl;
  ret << "    int32_t " << sizeName << ";" << endl;
  ret << "    int32_t " << stateName << ";" << endl;
  for (auto& localVar : localVars) {
    ret << "    " << printType(localVar.type(), false) << " " << varMap[localVar] << ";" << endl;
  }
  ret << "  } " << ctxClassName << ";" << endl;

  for (auto& localVar : localVars) {
    ret << "  " << printType(localVar.type(), false) << " " << varMap[localVar] << ";" << endl;
  }
  ret << "  int32_t " << bufSizeName << " = 0;" << endl;
  ret << "  int32_t " << bufCapacityCopyName << " = *" << bufCapacityName << ";"
      << endl;

  ret << "  if (*" << ctxName << ") {" << endl;
  for (auto& localVar : localVars) {
    const string varName = varMap[localVar];
    ret << "    " << varName << " = TACO_DEREF(" << varName << ");" << endl;
  }
  ret << "    switch (TACO_DEREF(" << stateName << ")) {" << endl;
  for (int i = 0; i <= labels; ++i) {
    ret << "      case " << i << ": goto " << labelPrefix << funcName << i
        << ";" << endl;
  }
  ret << "    }" << endl;
  ret << "  } else {" << endl;
  ret << "    " << printAlloc("*" + ctxName, "sizeof(" + ctxClassName + ")")
      << endl;
  ret << "    TACO_DEREF(" << sizeName << ") = sizeof(" << ctxClassName
      << ");" << endl;
  ret << "  }" << endl;

  return ret.str();
}

string CodeGen::unpackTensorProperty(string varname, const GetProperty* op,
                            bool is_output_prop) {
  stringstream ret;
  ret << "  ";

  auto tensor = op->tensor.as<Var>();
  if (op->property == TensorProperty::Values) {
    // for the values, it's in the last slot
    ret << printType(tensor->type, true);
    ret << " " << restrictKeyword() << " " << varname
        << " = (" << printType(tensor->type, true) << ")(";
    ret << tensor->name << "->vals);\n";
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
    tp = "int";
    ret << tp << " " << varname << " = (int)(" << tensor->name
        << "->dimensions[" << op->mode << "]);\n";;
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

string CodeGen::packTensorProperty(string varname, Expr tnsr,
                                   TensorProperty property,
                                   int mode, int index) {
  stringstream ret;
  ret << "  ";

  auto tensor = tnsr.as<Var>();
  if (property == TensorProperty::Values) {
    ret << tensor->name << "->vals";
    ret << " = (uint8_t*)" << varname << ";\n";
    return ret.str();
  } else if (property == TensorProperty::ValuesSize) {
    ret << tensor->name << "->vals_size = " << varname << ";\n";
    return ret.str();
  }

  string tp;

  // for a Dense level, nnz is an int
  // for a Fixed level, ptr is an int
  // all others are int*
  if (property == TensorProperty::Dimension) {
    return "";
  } else {
    taco_iassert(property == TensorProperty::Indices);
    tp = "int*";
    auto nm = index;
    ret << tensor->name << "->indices" <<
        "[" << mode << "][" << nm << "] = (uint8_t*)(" << varname
        << ");\n";
  }

  return ret.str();
}


// helper to print declarations
string CodeGen::printDecls(map<Expr, string, ExprCompare> varMap,
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

  return ret.str();
}


string CodeGen::printPack(map<tuple<Expr, TensorProperty, int, int>,
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
    ret << packTensorProperty(outputProperties[prop], get<0>(prop),
                              get<1>(prop), get<2>(prop), get<3>(prop));
  }
  return ret.str();
}

// seed the unique names with all C99 keywords
// from: http://en.cppreference.com/w/c/keyword
map<string, int> uniqueNameCounters;

void CodeGen::resetUniqueNameCounters() {
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
           {restrictKeyword(), 0},
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

string CodeGen::genUniqueName(string name) {
  stringstream os;
  os << name;
  if (uniqueNameCounters.count(name) > 0) {
    os << uniqueNameCounters[name]++;
  } else {
    uniqueNameCounters[name] = 0;
  }
  return os.str();
}


string CodeGen::printFuncName(const Function *func) {
  stringstream ret;

  ret << "int " << func->name << "(";

  string delimiter = "";
  const auto returnType = func->getReturnType();
  if (returnType.second != Datatype()) {
    ret << "void **" << ctxName << ", ";
    ret << "char *" << coordsName << ", ";
    ret << printType(returnType.second, true) << valName << ", ";
    ret << "int32_t *" << bufCapacityName;
    delimiter = ", ";
  }
  for (size_t i=0; i<func->outputs.size(); i++) {
    auto var = func->outputs[i].as<Var>();
    taco_iassert(var) << "Unable to convert output " << func->outputs[i]
                      << " to Var";
    if (var->is_tensor) {
      ret << delimiter << "taco_tensor_t *" << var->name;
    } else {
      auto tp = printType(var->type, var->is_ptr);
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
      auto tp = printType(var->type, var->is_ptr);
      ret << delimiter << tp << " " << var->name;
    }
    delimiter = ", ";
  }

  ret << ")";
  return ret.str();
}

void CodeGen::doIndentStream(stringstream &stream) {
  for (int i=0; i<indent; i++)
    stream << "  ";
}

string CodeGen::printCoroutineFinish(int numYields, string funcName) {
  stringstream ret;
  doIndentStream(ret);
  ret << "if (" << bufSizeName << " > 0) {" << endl;
  indent++;
  doIndentStream(ret);
  stream << "TACO_DEREF(" << stateName << ") = " << numYields << ";" << endl;
  doIndentStream(ret);
  stream << "return " << bufSizeName << ";" << endl;
  indent--;
  doIndentStream(ret);
  ret << "}" << endl;
  ret << labelPrefix << funcName << numYields << ":" << endl;

  doIndentStream(ret);
  ret << printFree("*" + ctxName) << endl;
  doIndentStream(ret);
  ret << "*" << ctxName << " = NULL;" << endl;
  return ret.str();
}

void CodeGen::printYield(const Yield* op, vector<Expr> localVars,
                           map<Expr, string, ExprCompare> varMap, int labelCount, string funcName) {
  int stride = 0;
  for (auto& coord : op->coords) {
    stride += coord.type().getNumBytes();
  }

  int offset = 0;
  for (auto& coord : op->coords) {
    doIndent();
    stream << "*(" << printType(coord.type(), true) << ")(" << coordsName << " + " << stride
           << " * " << bufSizeName;
    if (offset > 0) {
      stream << " + " << offset;
    }
    stream << ") = ";
    coord.accept(this);
    stream << ";" << endl;
    offset += coord.type().getNumBytes();
  }
  doIndent();
  stream << valName << "[" << bufSizeName << "] = ";
  op->val.accept(this);
  stream << ";" << endl;

  doIndent();
  stream << "if (++" << bufSizeName << " == " << bufCapacityCopyName << ") {"
         << endl;
  indent++;
  for (auto& localVar : localVars) {
    doIndent();
    const string varName = varMap[localVar];
    stream << "TACO_DEREF(" << varName << ") = " << varName << ";" << endl;
  }
  doIndent();
  stream << "TACO_DEREF(" << stateName << ") = " << labelCount << ";" << endl;
  doIndent();
  stream << "return " << bufSizeName << ";" << endl;
  indent--;
  doIndent();
  stream << "}" << endl;

  stream << labelPrefix << funcName << (labelCount++) << ":;" << endl;
}


}}
