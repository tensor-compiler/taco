#include "codegen_legion_c.h"

namespace taco {
namespace ir {

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
    ret << tp << " " << varname << " = runtime->get_index_space_domain(" << tensor->name <<
    ".get_index_space()).hi()[" << op->mode << "] + 1;\n";
//    ret << tp << " " << varname << " = (int)(" << tensor->name
//        << "->dimensions[" << op->mode << "]);\n";
  } else if (op->property == TensorProperty::IndexSpace) {
    tp = "auto";
    ret << tp << " " << varname << " = " << tensor->name << ".get_index_space();\n";
  } else if (op->property == TensorProperty::ValuesReadAccessor) {
    ret << "AccessorRO" << printType(op->type, false) << " " << varname << "(regions[?], FID_VAL);\n";
  } else if (op->property == TensorProperty::ValuesWriteAccessor) {
    ret << "AccessorRW" << printType(op->type, false) << " " << varname << "(regions[?], FID_VAL);\n";
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

  for (auto& f : util::reverse(this->functions)) {
    CodeGen_C::compile(f, isFirst);
//    f.accept(this);
  }

  CodeGen_C::compile(stmt, isFirst);
}

void CodegenLegionC::visit(const For* node) {
  if (node->isTask) {
    return;
  }
  CodeGen_C::visit(node);
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