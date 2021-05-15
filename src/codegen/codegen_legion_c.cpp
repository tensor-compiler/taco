#include "codegen_legion_c.h"
#include "codegen_c.h"
#include "taco/util/strings.h"
#include "taco/ir/ir_rewriter.h"
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

  virtual void visit(const PackTaskArgs* args) {
    auto func = this->codeGen->idToFunc.at(args->forTaskID).as<Function>();
    for (auto& e : this->codeGen->taskArgs[func]) {
      e.accept(this);
    }
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
  : CodeGen(dest, false, simplify, C), CodeGen_C(dest, outputKind, simplify) {}

void CodegenLegionC::visit(const PackTaskArgs *node) {
  doIndent();

  auto func = this->idToFunc.at(node->forTaskID).as<Function>();
  auto taskFor = this->idToFor.at(node->forTaskID).as<For>();
  taco_iassert(func) << "must be func";
  taco_iassert(taskFor) << "must be for";

  // Use this information to look up what variables need to be packed into the struct.
  auto stname = taskArgsName(func->name);

  // Make a variable for the raw allocation of the arguments.
  auto tempVar = node->var.as<Var>()->name + "Raw";
  out << stname << " " << tempVar << ";\n";

  // First emit mandatory prefix arguments.
  for (size_t i = 0; i < node->prefixVars.size(); i++) {
    doIndent();
    out << tempVar << "." << node->prefixVars[i] << " = " << node->prefixExprs[i] << ";\n";
  }

  for (auto arg : this->taskArgs[func]) {
    doIndent();
    out << tempVar << "." << arg << " = " << arg << ";\n";
  }

  // Construct the actual TaskArgument from this packed data.
  doIndent();
  out << "TaskArgument " << node->var << " = TaskArgument(&" << tempVar << ", sizeof(" << stname << "));\n";
}

void CodegenLegionC::compile(Stmt stmt, bool isFirst) {
  this->stmt = stmt;
  // Collect all of the individual functions that we need to generate code for.
  this->collectAllFunctions(stmt);
  // Rewrite the task ID's within each function so that they are all unique.
  this->rewriteFunctionTaskIDs();
  // Emit any needed headers.
  this->emitHeaders(out);

  // Emit field accessors.
  this->collectAndEmitAccessors(stmt, out);
  this->analyzeAndCreateTasks(out);

  for (auto& f : this->allFunctions) {
    for (auto func : this->functions[f]) {
      CodeGen_C::compile(func, isFirst);
    }
    CodeGen_C::compile(f, isFirst);
  }

  this->emitRegisterTasks(out);
}

void CodegenLegionC::visit(const For* node) {
  if (node->isTask) {
    return;
  }
  CodeGen_C::visit(node);
}

void CodegenLegionC::emitHeaders(std::ostream &o) {
  struct BLASFinder : public IRVisitor {
    void visit(const Call* node) {
      if (node->func.find("blas") != std::string::npos) {
        this->usesBLAS = true;
      }
    }
    bool usesBLAS = false;
  };
  BLASFinder bs;
  this->stmt.accept(&bs);
  if (bs.usesBLAS) {
    o << "#include \"cblas.h\"\n";
  }
  CodegenLegion::emitHeaders(o);
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

  // For tasks, unpack the regions.
  if (func->name.find("task") != std::string::npos) {
    auto parentFunc = this->funcToParentFunc[func];
    for (size_t i = 0; i < this->regionArgs[parentFunc].size(); i++) {
      doIndent();
      auto t = this->regionArgs[parentFunc][i];
      out << "PhysicalRegion " << t << " = regions[" << i << "];\n";
    }
    out << "\n";
  }

  // If this was a distributed for loop, emit the point as the loop index.
  // TODO (rohany): Hacky way to tell that this function was a task.
  if (func->name.find("task") != std::string::npos) {
    auto forL = this->funcToFor.at(func).as<For>();
    taco_iassert(forL) << "must be a for";
    if (forL->parallel_unit == ParallelUnit::DistributedNode) {
      doIndent();
      out << printType(forL->var.type(), false) << " " << forL->var << " = task->index_point[0];\n";
    }
  }

  // Unpack arguments.
  auto args = this->taskArgs[func];
  if (args.size() > 0) {
    doIndent();
    out << taskArgsName(func->name) << "* args = (" << taskArgsName(func->name) << "*)(task->args);\n";
    // Unpack arguments from the pack;
    for (auto arg : args) {
      doIndent();
      out << printType(getVarType(arg), false) << " " << arg << " = args->" << arg << ";\n";
    }

    out << "\n";
  }

  // TODO (rohany): Hack.
  // TODO (rohany): Hacky way to tell that this function was a task.
  if (func->name.find("task") != std::string::npos) {
    std::vector<Expr> toRemove;
    for (auto it : varFinder.varDecls) {
      if (isa<GetProperty>(it.first)) {
        auto g = it.first.as<GetProperty>();
        if (g->property == TensorProperty::Dimension) {
          toRemove.push_back(g);
        }
      }
    }
    for (auto it : toRemove) {
      varFinder.varDecls.erase(it);
    }
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

}
}