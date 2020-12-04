#include "llvm/IR/Module.h"

#include "codegen_llvm.h"

using namespace std;

namespace taco{
namespace ir{

class CodeGen_LLVM::FindVars : public IRVisitor{
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
  vector<Expr> inputTensors;
  vector<Expr> outputTensors;

  bool inBlock;

  CodeGen_LLVM *codeGen;

  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs, CodeGen_LLVM *codeGen)
      : codeGen(codeGen){
    for (auto v : inputs){
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(varMap.count(var) == 0) << "Duplicate input found in codegen: " << var->name;
      inputTensors.push_back(v);
      varMap[var] = var->name;
    }
    for (auto v : outputs){
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(varMap.count(var) == 0) << "Duplicate output found in codegen";

      outputTensors.push_back(v);
      varMap[var] = var->name;
    }
    inBlock = false;
  }

protected:
  using IRVisitor::visit;

  virtual void visit(const For *op){
    if (!util::contains(localVars, op->var)){
      localVars.push_back(op->var);
    }
    if (op->parallel_unit == ParallelUnit::GPUThread){
      // Want to collect the start, end, increment for the thread loop, but no other variables
      taco_iassert(inBlock);
      inBlock = false;
    }
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    if (op->parallel_unit == ParallelUnit::GPUBlock){
      inBlock = true;
    }
    if (op->parallel_unit == ParallelUnit::GPUThread){
      return;
    }
    op->contents.accept(this);
  }

  virtual void visit(const Var *op){
    if (varMap.count(op) == 0 && !inBlock){
      varMap[op] = codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl *op){
    if (!util::contains(localVars, op->var) && !inBlock){
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const GetProperty *op){
    if (varMap.count(op) == 0 && !inBlock){
      auto key =
          tuple<Expr, TensorProperty, int, int>(op->tensor, op->property,
                                                (size_t)op->mode,
                                                (size_t)op->index);
      if (canonicalPropertyVar.count(key) > 0){
        varMap[op] = canonicalPropertyVar[key];
      }
      else {
        auto unique_name = codeGen->genUniqueName(op->name);
        canonicalPropertyVar[key] = unique_name;
        varMap[op] = unique_name;
        varDecls[op] = unique_name;
        if (util::contains(outputTensors, op->tensor))
        {
          outputProperties[key] = unique_name;
        }
      }
    }
  }
};

void CodeGen_LLVM::compile(Stmt stmt, bool isFirst) {
  // throw logic_error("Not Implemented.");
  init_codegen();
  stmt.accept(this);
}

void CodeGen_LLVM::visit(const Literal* op){
  std::cout << "LLVM CodeGen Visiting Literal\n";
}

void CodeGen_LLVM::visit(const Var* op){
  std::cout << "LLVM CodeGen Visiting Var\n";
}

void CodeGen_LLVM::visit(const Neg* op){
  std::cout << "LLVM CodeGen Visiting Neg\n";
}

void CodeGen_LLVM::visit(const Sqrt* op){
  std::cout << "LLVM CodeGen Visiting Sqrt\n";
}

void CodeGen_LLVM::visit(const Add* op){
  std::cout << "LLVM CodeGen Visiting Add\n";
}

void CodeGen_LLVM::visit(const Sub* op){
  std::cout << "LLVM CodeGen Visiting Sub\n";
}

void CodeGen_LLVM::visit(const Mul* op){
  std::cout << "LLVM CodeGen Visiting Mul\n";
}

void CodeGen_LLVM::visit(const Div* op){
  std::cout << "LLVM CodeGen Visiting Div\n";
}

void CodeGen_LLVM::visit(const Rem* op){
  std::cout << "LLVM CodeGen Visiting Rem\n";
}

void CodeGen_LLVM::visit(const Min* op){
  std::cout << "LLVM CodeGen Visiting Min\n";
}

void CodeGen_LLVM::visit(const Max* op){
  std::cout << "LLVM CodeGen Visiting Max\n";
}

void CodeGen_LLVM::visit(const BitAnd* op){
  std::cout << "LLVM CodeGen Visiting BitAnd\n";
}

void CodeGen_LLVM::visit(const BitOr* op){
  std::cout << "LLVM CodeGen Visiting BitOr\n";
}

void CodeGen_LLVM::visit(const Eq* op){
  std::cout << "LLVM CodeGen Visiting Eq\n";
}

void CodeGen_LLVM::visit(const Neq* op){
  std::cout << "LLVM CodeGen Visiting Neq\n";
}

void CodeGen_LLVM::visit(const Gt* op){
  std::cout << "LLVM CodeGen Visiting Gt\n";
}

void CodeGen_LLVM::visit(const Lt* op){
  std::cout << "LLVM CodeGen Visiting Lt\n";
}

void CodeGen_LLVM::visit(const Gte* op){
  std::cout << "LLVM CodeGen Visiting Gte\n";
}

void CodeGen_LLVM::visit(const Lte* op){
  std::cout << "LLVM CodeGen Visiting Lte\n";
}

void CodeGen_LLVM::visit(const And* op){
  std::cout << "LLVM CodeGen Visiting And\n";
}

void CodeGen_LLVM::visit(const Or* op){
  std::cout << "LLVM CodeGen Visiting Or\n";
}

void CodeGen_LLVM::visit(const Cast* op){
  std::cout << "LLVM CodeGen Visiting Cast\n";
}

void CodeGen_LLVM::visit(const Call* op){
  std::cout << "LLVM CodeGen Visiting Call\n";
}

void CodeGen_LLVM::visit(const IfThenElse* op){
  std::cout << "LLVM CodeGen Visiting IfThenElse\n";
}

void CodeGen_LLVM::visit(const Case* op){
  std::cout << "LLVM CodeGen Visiting Case\n";
}

void CodeGen_LLVM::visit(const Switch* op){
  std::cout << "LLVM CodeGen Visiting Switch\n";
}

void CodeGen_LLVM::visit(const Load* op){
  std::cout << "LLVM CodeGen Visiting Load\n";
}

void CodeGen_LLVM::visit(const Malloc* op){
  std::cout << "LLVM CodeGen Visiting Malloc\n";
}

void CodeGen_LLVM::visit(const Sizeof* op){
  std::cout << "LLVM CodeGen Visiting Sizeof\n";
}

void CodeGen_LLVM::visit(const Store* op){
  std::cout << "LLVM CodeGen Visiting Store\n";
}

void CodeGen_LLVM::visit(const For* op){
  std::cout << "LLVM CodeGen Visiting For\n";
}

void CodeGen_LLVM::visit(const While* op){
  std::cout << "LLVM CodeGen Visiting While\n";
}

void CodeGen_LLVM::visit(const Block* op){
  std::cout << "LLVM CodeGen Visiting Block\n";
}

void CodeGen_LLVM::visit(const Scope* op){
  std::cout << "LLVM CodeGen Visiting Scope\n";
}

void CodeGen_LLVM::init_codegen(){
  llvm::IRBuilder<>(this->Context);

  auto i32 = llvm::Type::getInt32Ty(this->Context);
  auto i32p = i32->getPointerTo();

  auto u8 = llvm::Type::getInt8Ty(this->Context);
  auto u8p = u8->getPointerTo();
  auto u8ppp = u8->getPointerTo()->getPointerTo()->getPointerTo();

  /* See file include/taco/taco_tensor_t.h for the struct tensor definition */
  this->tensorType = llvm::StructType::create(this->Context, {
    i32, /* order */
    i32p, /* dimension */
    i32, /* csize */
    i32p, /* mode_ordering */
    i32p, /* mode_types */
    u8ppp, /* indices */
    u8p, /* vals */
    i32, /* vals_size */
  }, "TensorType");

}

void CodeGen_LLVM::visit(const Function* func){
  llvm::Module *M = new llvm::Module("my compiler", this->Context);
  std::cout << "LLVM CodeGen Visiting Function\n";

  // 1. find the arguments to @func
  FindVars varFinder(func->inputs, func->outputs, this);

  // 2. get the arguments types
  // Are all arguments tensors?

  // 3. convert the types to the LLVM correspondent ones
  int n_args = func->inputs.size() + func->outputs.size();
  std::vector<llvm::Type*> args;
  for (int i=0; i< n_args; i++){
    args.push_back(this->tensorType);
  }
  auto i32 = llvm::Type::getInt32Ty(this->Context);
  auto *FT = llvm::FunctionType::get(i32, args, false);

  // 4. create a new function in the module with the given types
  llvm::Function *F = llvm::Function::Create(FT, llvm::GlobalValue::ExternalLinkage, "my function", M);

  llvm::errs() << *M << "\n";

  // 5. visit function body
  func->body.accept(&varFinder);
}

void CodeGen_LLVM::visit(const VarDecl* op){
  std::cout << "LLVM CodeGen Visiting VarDecl\n";
}

void CodeGen_LLVM::visit(const Assign* op){
  std::cout << "LLVM CodeGen Visiting Assign\n";
}

void CodeGen_LLVM::visit(const Yield* op){
  std::cout << "LLVM CodeGen Visiting Yield\n";
}

void CodeGen_LLVM::visit(const Allocate* op){
  std::cout << "LLVM CodeGen Visiting Allocate\n";
}

void CodeGen_LLVM::visit(const Free* op){
  std::cout << "LLVM CodeGen Visiting Free\n";
}

void CodeGen_LLVM::visit(const Comment* op){
  std::cout << "LLVM CodeGen Visiting Comment\n";
}

void CodeGen_LLVM::visit(const BlankLine* op){
  std::cout << "LLVM CodeGen Visiting BlankLine\n";
}

void CodeGen_LLVM::visit(const Break* op){
  std::cout << "LLVM CodeGen Visiting Break\n";
}

void CodeGen_LLVM::visit(const Print* op){
  std::cout << "LLVM CodeGen Visiting Print\n";
}

void CodeGen_LLVM::visit(const GetProperty* op){
  std::cout << "LLVM CodeGen Visiting GetProperty\n";
}


} // namespace ir
} // namespace taco