#include "codegen_legion_c.h"
#include "codegen_c.h"
#include "taco/util/strings.h"
#include <algorithm>

namespace taco {
namespace ir {

// find variables for generating declarations
// generates a single var for each GetProperty
class CodegenLegionC::FindVars : public IRVisitor {
public:
  std::map<Expr, std::string, ExprCompare> varMap;

  // the variables for which we need to add declarations
  std::map<Expr, std::string, ExprCompare> varDecls;

  std::vector<Expr> localVars;

  // this maps from tensor, property, mode, index to the unique var
  std::map<std::tuple<Expr, TensorProperty, int, int>, std::string> canonicalPropertyVar;

  // this is for convenience, recording just the properties unpacked
  // from the output tensor so we can re-save them at the end
  std::map<std::tuple<Expr, TensorProperty, int, int>, std::string> outputProperties;

  // TODO: should replace this with an unordered set
  std::vector<Expr> outputTensors;
  std::vector<Expr> inputTensors;

  CodegenLegionC *codeGen;

  // copy inputs and outputs into the map
  FindVars(std::vector<Expr> inputs, std::vector<Expr> outputs, CodegenLegionC *codeGen)
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
      varMap[op] = op->is_ptr? op->name : codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const For *op) {
    // Don't count the variables inside the task as being used.
    if (op->isTask) {
      return;
    }

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
    // TODO (rohany): This might be needed.
//    if (!util::contains(inputTensors, op->tensor) &&
//        !util::contains(outputTensors, op->tensor)) {
//      // Don't create header unpacking code for temporaries
//      return;
//    }

    if (varMap.count(op) == 0) {
      auto key =
          std::tuple<Expr,TensorProperty,int,int>(op->tensor,op->property,
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

CodegenLegionC::CodegenLegionC(std::ostream &dest, OutputKind outputKind, bool simplify)
  : CodeGen_C(dest, outputKind, simplify) {}

// TODO (rohany): This is a bunch of duplicated code here, let's see how much we can
//  return back to the superclass.
std::string CodegenLegionC::unpackTensorProperty(std::string varname, const GetProperty* op, bool is_output_prop) {
  std::stringstream ret;
  std::string tp;
  auto tensor = op->tensor.as<Var>();
  ret << "  ";
  if (op->property == TensorProperty::Dimension) {
    tp = "int";
    ret << tp << " " << varname << " = runtime->get_index_space_domain(get_index_space(" << tensor->name <<
    ")).hi()[" << op->mode << "] + 1;\n";
  } else if (op->property == TensorProperty::IndexSpace) {
    tp = "auto";
    ret << tp << " " << varname << " = get_index_space(" << tensor->name << ");\n";
  } else if (op->property == TensorProperty::ValuesReadAccessor) {
    ret << "AccessorRO" << printType(op->type, false) << " " << varname << "(" << tensor->name << ", FID_VAL);\n";
  } else if (op->property == TensorProperty::ValuesWriteAccessor) {
    ret << "AccessorRW" << printType(op->type, false) << " " << varname << "(" << tensor->name << ", FID_VAL);\n";
  } else {
    return CodeGen::unpackTensorProperty(varname, op, is_output_prop);
  }
  return ret.str();
}

void CodegenLegionC::compile(Stmt stmt, bool isFirst) {
  struct TaskCollector : public IRVisitor {
    void visit(const For* node) {
      if (node->isTask) {
        std::stringstream funcName;
        funcName << "task_" << node->taskID;
        auto func = ir::Function::make(
            funcName.str(),
            {},
            {
              // TODO (rohany): Marking these as is_parameter = false stops some weird behavior
              //  in the rest of the code generator.
              ir::Var::make("task", Task, true, false, false),
              ir::Var::make("regions", PhysicalRegionVectorRef, false, false, false),
              ir::Var::make("ctx", Context, false, false, false),
              ir::Var::make("runtime", Runtime, true, false, false),
            },
            node->contents
        );
        this->functions.push_back(func);
      }
      node->contents.accept(this);
    }

    std::vector<Stmt> functions;
  };
  TaskCollector tc;
  stmt.accept(&tc);
  this->functions = tc.functions;

  if (isa<Function>(stmt)) {
    auto func = stmt.as<Function>();
    this->regionArgs.insert(this->regionArgs.end(), func->outputs.begin(), func->outputs.end());
    this->regionArgs.insert(this->regionArgs.end(), func->inputs.begin(), func->inputs.end());
  }

  struct VarsUsedByTask : public IRVisitor {
    void visit(const Var* v) {
      if (this->usedVars.size() == 0) {
        this->usedVars.push_back({});
      }
      this->usedVars.back().insert(v);
    }

    // We don't want to visit the variables within GetProperty objects.
    void visit(const GetProperty* g) {}

    void visit(const VarDecl* v) {
      if (this->varsDeclared.size() == 0) {
        this->varsDeclared.push_back({});
      }
      this->varsDeclared.back().insert(v->var);
    }

    void visit(const For* f) {
      if (f->isTask) {
        this->usedVars.push_back({});
        this->varsDeclared.push_back({});
      }
      // If f is a task, then it needs it's iteration variable passed down. If f is
      // a task, then we can treat it as _using_ the iteration variable.
      if (!f->isTask) {
        this->varsDeclared.back().insert(f->var);
      } else {
        this->usedVars.back().insert(f->var);
      }

      f->start.accept(this);
      f->end.accept(this);
      f->increment.accept(this);
      f->contents.accept(this);
    }

    std::vector<std::set<Expr>> usedVars;
    std::vector<std::set<Expr>> varsDeclared;
  };
  VarsUsedByTask v;
  stmt.accept(&v);
  for (auto it : v.usedVars) {
    std::cout << "Used vars in task: " << util::join(it) << std::endl;
  }
  for (auto it : v.varsDeclared) {
    std::cout << "Vars declared by task: " << util::join(it) << std::endl;
  }

  for (int i = v.usedVars.size() - 1; i > 0; i--) {
    // Try to find the variables needed by a task. It's all the variables it uses that it doesn't
    // declare and are used by tasks above it.
    std::vector<Expr> uses;
    std::set_difference(v.usedVars[i].begin(), v.usedVars[i].end(), v.varsDeclared[i].begin(), v.varsDeclared[i].end(), std::back_inserter(uses));
    // I want to pass this uses set up to the parent so that they know about it.
    std::cout << "Weija: " << util::join(uses) << std::endl;
  }

  for (auto& f : util::reverse(this->functions)) {
    CodeGen_C::compile(f, isFirst);
  }

  CodeGen_C::compile(stmt, isFirst);
}

void CodegenLegionC::visit(const For* node) {
  if (node->isTask) {
    return;
  }
  CodeGen_C::visit(node);
}

// TODO (rohany): This is duplicating alot of code.
void CodegenLegionC::visit(const Function* func) {
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

  out << " {\n";

  indent++;

  // find all the vars that are not inputs or outputs and declare them
  resetUniqueNameCounters();
  FindVars varFinder(func->inputs, func->outputs, this);
  func->body.accept(&varFinder);
  varMap = varFinder.varMap;
  localVars = varFinder.localVars;

//  std::cout << "For function: " << func->name << std::endl;
//  for (auto it : varFinder.varDecls) {
//    std::cout << it.first << std::endl;
//  }

  // For tasks, unpack the regions.
  if (func->name.find("task") != std::string::npos) {
    for (size_t i = 0; i < this->regionArgs.size(); i++) {
      doIndent();
      auto t = this->regionArgs[i];
      out << "PhysicalRegion " << t << " = regions[" << i << "];\n";
    }
    out << "\n";
  }

  // Print variable declarations
  out << printDecls(varFinder.varDecls, func->inputs, func->outputs) << std::endl;

  if (emittingCoroutine) {
    out << printContextDeclAndInit(varMap, localVars, numYields, func->name)
        << std::endl;
  }

  // output body
  print(func->body);

  // output repack only if we allocated memory
  if (checkForAlloc(func))
    out << std::endl << printPack(varFinder.outputProperties, func->outputs);

  if (emittingCoroutine) {
    out << printCoroutineFinish(numYields, funcName);
  }

//  doIndent();
  indent--;

  doIndent();
  out << "}\n";
}

// TODO (rohany): Duplicating alot of code here, but IDK a way around it.
std::string CodegenLegionC::printFuncName(const Function *func,
                              std::map<Expr, std::string, ExprCompare> inputMap,
                              std::map<Expr, std::string, ExprCompare> outputMap) {
  std::stringstream ret;

  // Tasks need to have a void function type.
  ret << "void " << func->name << "(";

  std::string delimiter;
//  const auto returnType = func->getReturnType();
//  if (returnType.second != Datatype()) {
//    ret << "void **" << ctxName << ", ";
//    ret << "char *" << coordsName << ", ";
//    ret << printType(returnType.second, true) << valName << ", ";
//    ret << "int32_t *" << bufCapacityName;
//    delimiter = ", ";
//  }

  bool unfoldOutput = false;
  for (size_t i=0; i<func->outputs.size(); i++) {
    auto var = func->outputs[i].as<Var>();
    taco_iassert(var) << "Unable to convert output " << func->outputs[i]
                      << " to Var";
    if (var->is_parameter) {
      unfoldOutput = true;
      break;
    }

    if (var->is_tensor) {
      ret << delimiter << "LogicalRegion " << var->name;
    } else {
      auto tp = printType(var->type, var->is_ptr);
      ret << delimiter << tp << " " << var->name;
    }
    delimiter = ", ";
  }

  if (unfoldOutput) {
    for (auto prop : sortProps(outputMap)) {
      ret << delimiter << printTensorProperty(outputMap[prop], prop, true);
      delimiter = ", ";
    }
  }

  bool unfoldInput = false;
  for (size_t i=0; i<func->inputs.size(); i++) {
    auto var = func->inputs[i].as<Var>();
    taco_iassert(var) << "Unable to convert output " << func->inputs[i]
                      << " to Var";
    if (var->is_parameter) {
      unfoldInput = true;
      break;
    }

    if (var->is_tensor) {
      ret << delimiter << "LogicalRegion " << var->name;
    } else {
      auto tp = printType(var->type, var->is_ptr);
      ret << delimiter << tp << " " << var->name;
    }
    delimiter = ", ";
  }

  if (unfoldInput) {
    for (auto prop : sortProps(inputMap)) {
      ret << delimiter << printTensorProperty(inputMap[prop], prop, false);
      delimiter = ", ";
    }
  }

  ret << ")";
  return ret.str();
}


}
}