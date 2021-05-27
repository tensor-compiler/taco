#include "codegen_legion_cuda.h"
#include "taco/ir/simplify.h"
#include "taco/util/strings.h"
#include "taco/version.h"

namespace taco {
namespace ir {

// find variables for generating declarations
// also only generates a single var for each GetProperty
class CodegenLegionCuda::FindVars : public IRVisitor {
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

  // Stop searching for variables at device functions (used to generate kernel launches)
  bool stopAtDeviceFunction;

  bool inBlock;

  CodegenLegionCuda *codeGen;

  // copy inputs and outputs into the map
  FindVars(std::vector<Expr> inputs, std::vector<Expr> outputs, CodegenLegionCuda *codeGen,
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
    // Don't count the variables inside the task as being used.
    if (op->isTask) {
      return;
    }

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

  virtual void visit(const PackTaskArgs* args) {
    auto func = this->codeGen->idToFunc.at(args->forTaskID).as<Function>();
    for (auto& e : this->codeGen->taskArgs[func]) {
      e.accept(this);
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

// Finds all for loops tagged with accelerator and adds statements to deviceFunctions
// Also tracks scope of when device function is called and
// tracks which variables must be passed to function.
class CodegenLegionCuda::DeviceFunctionCollector : public IRVisitor {
public:
  std::vector<Stmt> blockFors;
  std::vector<Stmt> threadFors; // contents is device function
  std::vector<Stmt> warpFors;
  std::map<Expr, std::string, ExprCompare> scopeMap;

  // the variables to pass to each device function
  std::vector<std::vector<std::pair<std::string, Expr>>> functionParameters;
  std::vector<std::pair<std::string, Expr>> currentParameters; // keep as vector so code generation is deterministic
  std::set<Expr> currentParameterSet;

  std::set<Expr> variablesDeclaredInKernel;

  std::vector<std::pair<std::string, Expr>> threadIDVars;
  std::vector<std::pair<std::string, Expr>> blockIDVars;
  std::vector<std::pair<std::string, Expr>> warpIDVars;
  std::vector<Expr> numThreads;
  std::vector<Expr> numWarps;

  CodegenLegionCuda *codeGen;
  // copy inputs and outputs into the map
  DeviceFunctionCollector(std::vector<Expr> inputs, std::vector<Expr> outputs, CodegenLegionCuda *codeGen) : codeGen(codeGen)  {
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
    // Don't follow into task calls.
    if (op->isTask) {
      return;
    }
    // Don't need to find/initialize loop bounds
    if (op->parallel_unit == ParallelUnit::GPUBlock) {
      op->var.accept(this);
      taco_iassert(!inDeviceFunction) << "Nested Device functions not supported";
      blockFors.push_back(op);
      blockIDVars.push_back(std::pair<std::string, Expr>(scopeMap[op->var], op->var));
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
      warpIDVars.push_back(std::pair<std::string, Expr>(scopeMap[op->var], op->var));
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
      threadIDVars.push_back(std::pair<std::string, Expr>(scopeMap[op->var], op->var));
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
      std::string name = codeGen->genUniqueName(op->name);
      if (!inDeviceFunction) {
        scopeMap[op] = name;
      }
    }
    else if (scopeMap.count(op) == 1 && inDeviceFunction && currentParameterSet.count(op) == 0
             && (threadIDVars.empty() || op != threadIDVars.back().second)
             && (blockIDVars.empty() || op != blockIDVars.back().second)
             && (warpIDVars.empty() || op != warpIDVars.back().second)
             && !variablesDeclaredInKernel.count(op)) {
      currentParameters.push_back(std::pair<std::string, Expr>(scopeMap[op], op));
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
          std::tuple<Expr,TensorProperty,int,int>(op->tensor,op->property,
                                                  (size_t)op->mode,
                                                  (size_t)op->index);
      auto unique_name = codeGen->genUniqueName(op->name);
      scopeMap[op->tensor] = unique_name;
    }
    else if (scopeMap.count(op->tensor) == 1 && inDeviceFunction && currentParameterSet.count(op->tensor) == 0) {
      currentParameters.push_back(std::pair<std::string, Expr>(op->tensor.as<Var>()->name, op->tensor));
      currentParameterSet.insert(op->tensor);
    }
  }
};

CodegenLegionCuda::CodegenLegionCuda(std::ostream &dest, OutputKind outputKind, bool simplify)
  : CodeGen(dest, false, simplify, CUDA), CodeGen_CUDA(dest, outputKind), CodegenLegion(dest, CUDA) {}

void CodegenLegionCuda::compile(Stmt stmt, bool isFirst) {
  stmt = simplifyFunctionBodies(stmt);
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
      CodeGen_CUDA::compile(func, isFirst);
    }
    CodeGen_CUDA::compile(f, isFirst);
  }

  this->emitRegisterTasks(out);
}

// TODO (rohany): See if we can deduplicate and pull this into the CodegenLegion.
void CodegenLegionCuda::visit(const PackTaskArgs *node) {
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

void CodegenLegionCuda::visit(const For* node) {
  if (node->isTask) {
    return;
  }
  CodeGen_CUDA::visit(node);
}

void CodegenLegionCuda::visit(const Function* func) {
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

  // Added.
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
  FindVars varFinder(func->inputs, func->outputs, this, true);
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
    if (distributedParallelUnit(forL->parallel_unit)) {
      doIndent();
      out << printType(forL->var.type(), false) << " " << forL->var << " = task->index_point[0];\n";
    }
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

  // We may have to emit variable declarations for get properties that child kernels use.
  FindVars gpFinder(func->inputs, func->outputs, this);
  func->body.accept(&gpFinder);
  for (auto it : gpFinder.varDecls) {
    if (isa<GetProperty>(it.first)) {
      auto gp = it.first.as<GetProperty>();
      switch (gp->property) {
        case TensorProperty::ValuesReadAccessor:
        case TensorProperty::ValuesWriteAccessor:
        case TensorProperty::ValuesReductionAccessor:
          if (varFinder.varDecls.count(it.first) == 0) {
            varFinder.varDecls[it.first] = it.second;
          }
          break;
        default:
          break;
      }
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

  indent--;

  doIndent();
  out << "}\n";
}

void CodegenLegionCuda::printDeviceFunctions(const Function* func) {
  // Collect device functions
  resetUniqueNameCounters();
  deviceFunctionLoopDepth = 0;
  DeviceFunctionCollector deviceFunctionCollector(func->inputs, func->outputs, this);
  func->body.accept(&deviceFunctionCollector);
  deviceFunctions = deviceFunctionCollector.blockFors;
  deviceFunctionParameters = deviceFunctionCollector.functionParameters;

  // TODO (rohany): I think that we need to explicitly do parameter passing here,
  //  we know more than the FindVars visitor.
  struct TensorAccessFinder : public IRVisitor {
    void visit(const GetProperty* op) {
      switch (op->property) {
        case TensorProperty::ValuesReadAccessor:
        case TensorProperty::ValuesWriteAccessor:
        case TensorProperty::ValuesReductionAccessor:
          this->gps[op->tensor] = op;
          break;
        default:
          return;
      }
    }
    std::map<Expr, const GetProperty*> gps;
  };
  TensorAccessFinder taf;
  func->accept(&taf);
  for (auto& params : deviceFunctionParameters) {
    // Collect the original set of parameters. We don't want to double count
    // any of the parameters that we manually are adding.
    std::set<Expr> existingParams;
    for (auto param : params) {
      existingParams.insert(param.second);
    }
    auto addParam = [&](std::pair<std::string, Expr> param) {
      if (!util::contains(existingParams, param.second)) {
        params.push_back(param);
      }
    };

    for (auto arg : this->regionArgs[this->funcToParentFunc[func]]) {
      auto op = taf.gps[arg];
      auto param = ir::Var::make(op->name, Datatype(accessorType(op)));
      addParam(std::make_pair(getVarName(param), param));
    }

    for (auto arg : this->taskArgs[func]) {
      Expr param = arg;
      if (arg.as<GetProperty>()) {
        auto gp = arg.as<GetProperty>();
        param = ir::Var::make(gp->name, gp->type);
      }
      addParam(std::make_pair(getVarName(param), param));
    }

    // If this was a distributed for loop, emit the point as the loop index.
    // TODO (rohany): Hacky way to tell that this function was a task.
    if (func->name.find("task") != std::string::npos) {
      auto forL = this->funcToFor.at(func).as<For>();
      taco_iassert(forL) << "must be a for";
      if (distributedParallelUnit(forL->parallel_unit)) {
        addParam(std::make_pair(getVarName(forL->var), forL->var));
      }
    }
  }

  // Remove tensor arguments from the parameters. We do this because for a Legion
  // backend, we don't have taco_tensor_t* objects to pass around!
  for (size_t i = 0; i < deviceFunctionParameters.size(); i++) {
    std::vector<std::pair<std::string, Expr>> newParams;
    for (auto& param : deviceFunctionParameters[i]) {
      if (!param.second.as<Var>()->is_tensor) {
        newParams.push_back(param);
      }
    }
    deviceFunctionParameters[i] = newParams;
  }

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
    std::vector<std::pair<std::string, Expr>> parameters = deviceFunctionParameters[i];

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
    out << " {\n";
    indent++;

    // Generate device function code
    resetUniqueNameCounters();
    std::vector<Expr> inputs;
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

    // We shouldn't emit var declarations for certain get properties -- these will be passed in
    // from the parent.
    std::vector<Expr> toRemove;
    for (auto it : varFinder.varDecls) {
      if (isa<GetProperty>(it.first)) {
        auto g = it.first.as<GetProperty>();
        switch (g->property) {
          case TensorProperty::ValuesReadAccessor:
          case TensorProperty::ValuesWriteAccessor:
          case TensorProperty::ValuesReductionAccessor:
          case TensorProperty::Dimension:
            toRemove.push_back(g);
            break;
          default:
            break;
        }
      }
    }
    for (auto it : toRemove) {
      varFinder.varDecls.erase(it);
    }

    // Print variable declarations
    out << printDecls(varFinder.varDecls, inputs, {}) << std::endl;
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
      out << std::endl << printPack(varFinder.outputProperties, func->outputs);
    indent--;
    doIndent();
    out << "}\n\n";
  }
}

std::string CodegenLegionCuda::procForTask(Stmt target, Stmt task) {
  // Walk the statement to figure out what kind of distribution
  // unit this task is within.
  auto forL = this->funcToFor.at(task).as<For>();
  taco_iassert(forL) << "should have found a for";
  struct Walker : IRVisitor {
    void visit(const For* node) {
      if (node->parallel_unit == ParallelUnit::DistributedNode) {
        if (TACO_FEATURE_OPENMP) {
          procKind = "Processor::OMP_PROC";
        } else {
          procKind = "Processor::LOC_PROC";
        }
      } else if (node->parallel_unit == ParallelUnit::DistributedGPU) {
        procKind = "Processor::TOC_PROC";
      }
      if (node->taskID == func->taskID) {
        return;
      }
      node->contents.accept(this);
    }
    std::string procKind;
    const For* func;
  } walker; walker.func = forL;
  target.accept(&walker);
  taco_iassert(walker.procKind.size() > 0);
  return walker.procKind;
}

void CodegenLegionCuda::emitHeaders(std::ostream &o) {
  // For simplicity, let's always just include the cublas headers.
  o << "#include \"cublas_v2.h\"\n";
  o << "#include \"cudalibs.h\"\n";
  CodegenLegion::emitHeaders(o);
}

}
}
