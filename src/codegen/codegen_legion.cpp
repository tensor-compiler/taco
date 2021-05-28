#include "codegen_legion.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/version.h"
#include "taco/util/strings.h"

namespace taco {
namespace ir {

std::string CodegenLegion::unpackTensorProperty(std::string varname, const GetProperty* op, bool is_output_prop) {
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
    ret << accessorType(op) << " " << varname << "(" << tensor->name << ", FID_VAL);\n";
  } else if (op->property == TensorProperty::ValuesWriteAccessor) {
    ret << accessorType(op) << " " << varname << "(" << tensor->name
        << ", FID_VAL);\n";
  } else if (op->property == TensorProperty::ValuesReductionAccessor) {
    ret << accessorType(op) << " " << varname << "(" << tensor->name
        << ", FID_VAL, " << LegionRedopString(op->type) << ");\n";
  } else {
    return CodeGen::unpackTensorProperty(varname, op, is_output_prop);
  }
  return ret.str();
}

std::string CodegenLegion::printFuncName(const Function *func,
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

  for (size_t i=0; i<func->inputs.size(); i++) {
    auto var = func->inputs[i].as<Var>();
    taco_iassert(var) << "Unable to convert output " << func->inputs[i]
                      << " to Var";
    if (var->is_tensor) {
      ret << delimiter << "LogicalRegion " << var->name;
    } else {
      auto tp = printType(var->type, var->is_ptr);
      ret << delimiter << tp << " " << var->name;
    }
    delimiter = ", ";
  }

  ret << ")";
  return ret.str();
}

// TODO (rohany): It's possible that this should be 2 calls -- collect and emit.
//  In that way, we can put all of the data structure collection up front.
void CodegenLegion::collectAndEmitAccessors(ir::Stmt stmt, std::ostream& out) {
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
  this->accessors = acol.accessors;

  // Emit a field accessor for each kind.
  for (auto info : this->accessors) {
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
}

void CodegenLegion::emitHeaders(std::ostream &out) {
  out << "#include \"taco_legion_header.h\"\n";
  out << "#include \"taco_mapper.h\"\n";
  out << "#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))\n";
  out << "using namespace Legion;\n";
}

void CodegenLegion::collectAllFunctions(ir::Stmt stmt) {
  struct FunctionFinder : public IRVisitor {
    void visit(const Function* func) {
      this->funcs.push_back(func);
    }
    std::vector<Stmt> funcs;
  };
  FunctionFinder ff;
  stmt.accept(&ff);
  this->allFunctions = ff.funcs;
}

void CodegenLegion::rewriteFunctionTaskIDs() {
  taco_iassert(this->allFunctions.size() > 0) << "must be called after collectAllFunctions()";
  // Rewrite task ID's using a scan-like algorithm.
  auto maxTaskID = 0;
  for (size_t i = 0; i < this->allFunctions.size(); i++) {
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
        stmt = PackTaskArgs::make(p->var, p->forTaskID + this->maxTaskID, p->prefixVars, p->prefixExprs);
      }

      int maxTaskID;
    };
    TaskIDRewriter rw; rw.maxTaskID = maxTaskID;
    this->allFunctions[i] = rw.rewrite(this->allFunctions[i]);

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
    this->allFunctions[i].accept(&mf);
    maxTaskID = mf.maxTaskID;
  }
}

void CodegenLegion::analyzeAndCreateTasks(std::ostream& out) {
  for (auto ffunc : this->allFunctions) {
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
        // If f is a task, then it needs it's iteration variable passed down. So f is
        // a task, then we can treat it as _using_ the iteration variable. Otherwise,
        // the for loop declares its iterator variable. However, we only want to do
        // this if we are already in a task.
        if (!f->isTask && this->varsDeclared.size() > 0) {
          this->varsDeclared.back().insert(f->var);
        } else if (f->isTask && this->usedVars.size() > 0) {
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

      // TODO (rohany): For a distributed for loop, remove the iterator variable?
      auto forL = this->funcToFor.at(func).as<For>();
      if (distributedParallelUnit(forL->parallel_unit)) {
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

      v.usedVars[i-1].insert(uses.begin(), uses.end());

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
      struct ExprSorter {
        bool operator() (Expr e1, Expr e2) {
          auto e1Name = getVarName(e1);
          auto e2Name = getVarName(e2);
          return e1Name < e2Name;
        }
      } exprSorter;
      std::sort(uses.begin(), uses.end(), exprSorter);

      // Find any included arguments from PackTaskArgs for this function.
      struct PackFinder : public IRVisitor {
        void visit(const PackTaskArgs* pack) {
          if (pack->forTaskID == taskID) {
            this->packVars = pack->prefixVars;
          }
        }

        std::vector<Expr> packVars;
        int taskID;
      };
      PackFinder pf; pf.taskID = forL->taskID;
      ffunc.accept(&pf);

      out << "struct " << this->taskArgsName(func->name) << " {\n";
      this->indent++;
      for (auto& var : pf.packVars) {
        doIndent();
        out << printType(getVarType(var), false) << " " << var << ";\n";
      }
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
}

std::string CodegenLegion::procForTask(Stmt, Stmt) {
  if (TACO_FEATURE_OPENMP) {
    return "Processor::OMP_PROC";
  }
  return "Processor::LOC_PROC";
}

void CodegenLegion::emitRegisterTasks(std::ostream &out) {
  // Output a function performing all of the task registrations.
  out << "void registerTacoTasks() {\n";
  indent++;

  for (auto ffunc : this->allFunctions) {
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

      // TODO (rohany): Make this delegation a virtual function that needs to be overridden.
      doIndent();
      std::string proc = this->procForTask(ffunc, func);
      out << "registrar.add_constraint(ProcessorConstraint(" << proc << "));\n";

      doIndent();
      if (finder.isLeaf) {
        out << "registrar.set_leaf();\n";
      } else {
        out << "registrar.set_inner();\n";
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

}
}
