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
    ret << "AccessorRO" << printType(op->type, false) << op->mode << " " << varname << "(" << tensor->name << ", FID_VAL);\n";
  } else if (op->property == TensorProperty::ValuesWriteAccessor) {
    ret << "AccessorRW" << printType(op->type, false) << op->mode << " " << varname << "(" << tensor->name
        << ", FID_VAL);\n";
  } else if (op->property == TensorProperty::ValuesReductionAccessor) {
    ret << "AccessorReduce" << printType(op->type, false) << op->mode << " " << varname << "(" << tensor->name
        << ", FID_VAL, " << LegionRedopString(op->type) << ");\n";
  } else {
    return CodeGen::unpackTensorProperty(varname, op, is_output_prop);
  }
  return ret.str();
}

std::string getVarName(Expr e) {
  if (isa<Var>(e)) {
    return e.as<Var>()->name;
  }
  if (isa<GetProperty>(e)) {
    return e.as<GetProperty>()->name;
  }
  taco_ierror;
  return "";
}

Datatype getVarType(Expr e) {
  if (isa<Var>(e)) {
    return e.as<Var>()->type;
  }
  if (isa<GetProperty>(e)) {
    return e.as<GetProperty>()->type;
  }
  taco_ierror;
  return Datatype();
}

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
  for (auto arg : this->taskArgs[func]) {
    doIndent();
    out << tempVar << "." << arg << " = " << arg << ";\n";
  }

  // Construct the actual TaskArgument from this packed data.
  doIndent();
  out << "TaskArgument " << node->var << " = TaskArgument(&" << tempVar << ", sizeof(" << stname << "));\n";
}

struct AccessorInfo {
  TensorProperty prop;
  int dims;
  Datatype typ;

  friend bool operator<(const AccessorInfo& a, const AccessorInfo& b) {
    if (a.prop < b.prop) {
      return true;
    }
    if (a.dims < b.dims) {
      return true;
    }
    if (a.typ < b.typ) {
      return true;
    }
    return false;
  }
};

void CodegenLegionC::compile(Stmt stmt, bool isFirst) {

  // Figure out what accessors we need to emit.
  struct AccessorCollector : public IRVisitor {
    void visit(const GetProperty* op) {
      switch (op->property) {
        case TensorProperty::ValuesReadAccessor:
        case TensorProperty::ValuesWriteAccessor:
        case TensorProperty::ValuesReductionAccessor:
          this->accessors.insert(AccessorInfo{op->property, op->mode, op->type});
          break;
        default:
          return;
      }
    }
    std::set<AccessorInfo> accessors;
  };
  AccessorCollector acol;
  stmt.accept(&acol);

  // Collect all of the individual functions that we need to generate code for.
  struct FunctionFinder : public IRVisitor {
    void visit(const Function* func) {
      this->funcs.push_back(func);
    }
    std::vector<Stmt> funcs;
  };
  FunctionFinder ff;
  stmt.accept(&ff);

  // Rewrite task ID's using a scan-like algorithm.
  auto maxTaskID = 0;
  for (size_t i = 0; i < ff.funcs.size(); i++) {
    // Increment all task ID's present in the function by the maxTaskID.
    struct TaskIDRewriter : public IRRewriter {
      void visit(const For* node) {
        auto body = rewrite(node->contents);
        if (node->isTask) {
          stmt = ir::For::make(node->var, node->start, node->end, node->increment, body, node->kind, node->parallel_unit, node->unrollFactor, node->vec_width, node->isTask, node->taskID + maxTaskID);
        } else {
          stmt = ir::For::make(node->var, node->start, node->end, node->increment, body, node->kind, node->parallel_unit, node->unrollFactor, node->vec_width, node->isTask, node->taskID);
        }
      }

      void visit(const Call* call) {
        if (call->func == "taskID") {
          auto oldTaskID = call->args[0].as<Literal>()->getValue<int>();
          expr = ir::Call::make("taskID", {oldTaskID + maxTaskID}, Auto);
        } else {
          std::vector<Expr> newArgs;
          for (auto e : call->args) {
            newArgs.push_back(rewrite(e));
          }
          expr = ir::Call::make(call->func, newArgs, call->type);
        }
      }

      void visit(const PackTaskArgs* p) {
        stmt = PackTaskArgs::make(p->var, p->forTaskID + this->maxTaskID);
      }

      int maxTaskID;
    };
    TaskIDRewriter rw; rw.maxTaskID = maxTaskID;
    ff.funcs[i] = rw.rewrite(ff.funcs[i]);

    struct MaxTaskIDFinder : public IRVisitor {
      void visit(const For* node) {
        if (node->isTask) {
          this->maxTaskID = std::max(this->maxTaskID, node->taskID);
        }
        if (node->contents.defined()) {
          node->contents.accept(this);
        }
      }
      int maxTaskID;
    };
    MaxTaskIDFinder mf; mf.maxTaskID = maxTaskID;
    ff.funcs[i].accept(&mf);
    maxTaskID = mf.maxTaskID;
  }


  // Emit the include.
  out << "#include \"taco_legion_header.h\"\n";
  out << "using namespace Legion;\n";

  // Emit a field accessor for each kind.
  for (auto info : acol.accessors) {
    if (info.prop == TensorProperty::ValuesReductionAccessor) {
      out << "typedef ReductionAccessor<SumReduction<" << printType(info.typ, false)
          << ">,true," << info.dims << ",coord_t,Realm::AffineAccessor<" << printType(info.typ, false)
          << "," << info.dims << ",coord_t>> AccessorReduce" << printType(info.typ, false) << info.dims << ";\n";
    } else {
      std::string priv, suffix;
      if (info.prop == TensorProperty::ValuesWriteAccessor) {
        priv = "READ_WRITE";
        suffix = "RW";
      } else {
        priv = "READ_ONLY";
        suffix = "RO";
      }
      out << "typedef FieldAccessor<" << priv << "," << printType(info.typ, false) << ","
          << info.dims << ",coord_t,Realm::AffineAccessor<" << printType(info.typ, false) << ","
          << info.dims << ",coord_t>> Accessor" << suffix << printType(info.typ, false) << info.dims << ";\n";
    }
  }
  out << "\n";

  for (auto ffunc : ff.funcs) {
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
          this->idToFor[node->taskID] = node;
          this->idToFunc[node->taskID] = func;
          this->funcToFor[func] = node;
        }
        node->contents.accept(this);
      }

      std::vector<Stmt> functions;

      std::map<int, Stmt> idToFor;
      std::map<int, Stmt> idToFunc;
      std::map<Stmt, Stmt> funcToFor;
    };
    TaskCollector tc;
    ffunc.accept(&tc);
    for (auto f : util::reverse(tc.functions)) {
      this->functions[ffunc].push_back(f);
      this->funcToParentFunc[f] = ffunc;
    }
    this->idToFor.insert(tc.idToFor.begin(), tc.idToFor.end());
    this->idToFunc.insert(tc.idToFunc.begin(), tc.idToFunc.end());
    this->funcToFor.insert(tc.funcToFor.begin(), tc.funcToFor.end());

    // Collect the region arguments that each function needs.
    if (isa<Function>(ffunc)) {
      auto func = ffunc.as<Function>();
      for (auto& arg : func->outputs) {
        if (arg.as<Var>()->is_tensor) {
          this->regionArgs[func].push_back(arg);
        }
      }
      for (auto& arg : func->inputs) {
        if (arg.as<Var>()->is_tensor) {
          this->regionArgs[func].push_back(arg);
        }
      }
    }

    // Find variables used by each task in the task call hierarchy.
    struct VarsUsedByTask : public IRVisitor {
      void visit(const Var* v) {
        if (this->usedVars.size() == 0) {
          this->usedVars.push_back({});
        }
        if (v->type.getKind() != Datatype::CppType && !v->is_tensor) {
          this->usedVars.back().insert(v);
        }
      }

      // We don't want to visit the variables within GetProperty objects.
      void visit(const GetProperty* g) {
        if (g->property == TensorProperty::Dimension) {
          if (this->usedVars.size() == 0) {
            this->usedVars.push_back({});
          }
          this->usedVars.back().insert(g);
        }
      }

      void visit(const VarDecl* v) {
        if (this->varsDeclared.size() == 0) {
          this->varsDeclared.push_back({});
        }
        this->varsDeclared.back().insert(v->var);
        v->rhs.accept(this);
      }

      void visit(const For* f) {
        if (f->isTask) {
          this->usedVars.push_back({});
          this->varsDeclared.push_back({});
        }
        // TODO (rohany): This comment doesn't make sense.
        // If f is a task, then it needs it's iteration variable passed down. If f is
        // a task, then we can treat it as _using_ the iteration variable.
        if (!f->isTask) {
          taco_iassert(this->varsDeclared.size() > 0);
          this->varsDeclared.back().insert(f->var);
        } else {
          taco_iassert(this->usedVars.size() > 0);
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
    ffunc.accept(&v);

    // TODO (rohany): Clean up this code.
    auto funcIdx = 0;
    for (int i = v.usedVars.size() - 1; i > 0; i--) {
      auto func = this->functions[ffunc][funcIdx].as<Function>();
      taco_iassert(func) << "must be func";
      // Try to find the variables needed by a task. It's all the variables it uses that it doesn't
      // declare and are used by tasks above it.
      std::vector<Expr> uses;
      std::set_difference(v.usedVars[i].begin(), v.usedVars[i].end(), v.varsDeclared[i].begin(), v.varsDeclared[i].end(), std::back_inserter(uses));
      v.usedVars[i-1].insert(uses.begin(), uses.end());

      // TODO (rohany): For a distributed for loop, remove the iterator variable?
      auto forL = this->funcToFor.at(func).as<For>();
      if (forL->parallel_unit == ParallelUnit::DistributedNode) {
        auto matchedIdx = -1;
        for (size_t pos = 0; pos < uses.size(); pos++) {
          if (uses[pos] == forL->var) {
            matchedIdx = pos;
            break;
          }
        }
        if (matchedIdx != -1) {
          uses.erase(uses.begin() + matchedIdx);
        }
      }

      // Deduplicate any GetProperty uses so that they aren't emitted twice.
      std::vector<const GetProperty*> collected;
      std::vector<Expr> newUses;
      for (auto& e : uses) {
        if (isa<GetProperty>(e)) {
          // See if this GetProperty is already present.
          bool found = false;
          auto gp = e.as<GetProperty>();
          for (auto c : collected) {
            if (gp->tensor == c->tensor && gp->property == c->property && gp->mode == c->mode) {
              found = true;
              break;
            }
          }
          if (!found) {
            newUses.push_back(e);
            collected.push_back(gp);
          }
        } else {
          newUses.push_back(e);
        }
      }
      uses = newUses;

      out << "struct " << this->taskArgsName(func->name) << " {\n";
      this->indent++;
      for (auto& it : uses) {
        doIndent();
        out << printType(getVarType(it), false) << " " << it << ";\n";
      }
      this->indent--;
      out << "};\n";

      this->taskArgs[func] = uses;

      funcIdx++;
    }
  }


  for (auto& f : ff.funcs) {
    for (auto func : this->functions[f]) {
      CodeGen_C::compile(func, isFirst);
    }
    CodeGen_C::compile(f, isFirst);
  }

  // Output a function performing all of the task registrations.
  out << "void registerTacoTasks() {\n";
  indent++;

  for (auto ffunc : ff.funcs) {
    for (auto& f : this->functions[ffunc]) {
      auto func = f.as<Function>();
      auto forL = this->funcToFor.at(func).as<For>();

      // Tasks that launch no tasks are leaf tasks, so let Legion know about that.
      struct LeafTaskFinder : public IRVisitor {
        void visit(const For* node) {
          if (node->isTask) {
            this->isLeaf = false;
          }
          node->contents.accept(this);
        }
        bool isLeaf = true;
      };
      LeafTaskFinder finder;
      forL->contents.accept(&finder);

      doIndent();
      out << "{\n";
      indent++;

      doIndent();
      out << "TaskVariantRegistrar registrar(taskID(" << forL->taskID << "), \"" << func->name << "\");\n";

      doIndent();
      out << "registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));\n";

      if (finder.isLeaf) {
        doIndent();
        out << "registrar.set_leaf();\n";
      }

      doIndent();
      out << "Runtime::preregister_task_variant<" << func->name << ">(registrar, \"" <<  func->name << "\");\n";

      indent--;

      doIndent();
      out << "}\n";
    }
  }

  out << "}\n";
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

// TODO (rohany): Duplicating alot of code here, but IDK a way around it.
std::string CodegenLegionC::printFuncName(const Function *func,
                              std::map<Expr, std::string, ExprCompare> inputMap,
                              std::map<Expr, std::string, ExprCompare> outputMap) {
  std::stringstream ret;

  // Tasks need to have a void function type.
  if (func->name.find("place") != std::string::npos) {
    ret << "LogicalPartition " << func->name << "(";
  } else {
    ret << "void " << func->name << "(";
  }

  std::string delimiter;
//  const auto returnType = func->getReturnType();
//  if (returnType.second != Datatype()) {
//    ret << "void **" << ctxName << ", ";
//    ret << "char *" << coordsName << ", ";
//    ret << printType(returnType.second, true) << valName << ", ";
//    ret << "int32_t *" << bufCapacityName;
//    delimiter = ", ";
//  }

  if (func->name.find("task") == std::string::npos) {
    // Add the context and runtime arguments.
    ret << "Context ctx, Runtime* runtime, ";
  }

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