#include "llvm/IR/Module.h"

#include "codegen_llvm.h"

using namespace std;

namespace taco {
namespace ir {

class CodeGen_LLVM::FindVars : public IRVisitor {
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
      : codeGen(codeGen) {
    for (auto v : inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(varMap.count(var) == 0)
          << "Duplicate input found in codegen: " << var->name;
      inputTensors.push_back(v);
      varMap[var] = var->name;
    }
    for (auto v : outputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(varMap.count(var) == 0)
          << "Duplicate output found in codegen";

      outputTensors.push_back(v);
      varMap[var] = var->name;
    }
    inBlock = false;
  }

protected:
  using IRVisitor::visit;

  virtual void visit(const For *op) {
    llvm::errs() << "LLVM FindVars Visiting For\n";
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
  }

  virtual void visit(const Var *op) {
    llvm::errs() << "LLVM FindVars Visiting Var \"" << op->name << "\"\n";
    if (varMap.count(op) == 0 && !inBlock) {
      varMap[op] = codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl *op) {
    llvm::errs() << "LLVM FindVars Visiting VarDecl\n";
    if (!util::contains(localVars, op->var) && !inBlock) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const GetProperty *op) {
    if (varMap.count(op) == 0 && !inBlock) {
      auto key = tuple<Expr, TensorProperty, int, int>(
          op->tensor, op->property, (size_t)op->mode, (size_t)op->index);
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

void CodeGen_LLVM::pushSymbol(const std::string &name, llvm::Value *v) {
  this->symbolTable.insert({name, v});
}

llvm::Value *CodeGen_LLVM::getSymbol(const std::string &name) {
  return this->symbolTable.get(name);
}

void CodeGen_LLVM::pushScope() { this->symbolTable.scope(); }

void CodeGen_LLVM::popScope() { this->symbolTable.unscope(); }

// Convert from taco type to LLVM type
llvm::Type *CodeGen_LLVM::llvmTypeOf(Datatype t) {
  taco_tassert(!t.isComplex()) << "LLVM codegen for complex not yet supported";

  if (t.isFloat()) {
    switch (t.getNumBits()) {
    case 32:
      return llvm::Type::getFloatTy(this->Context);
    case 64:
      return llvm::Type::getDoubleTy(this->Context);
    default:
      taco_ierror << "Unable to find LLVM type for " << t;
      return nullptr;
    }
  } else if (t.isInt()) {
    return llvm::Type::getIntNTy(this->Context, t.getNumBits());
  } else {
    taco_ierror << "Unable to find llvm type for " << t;
    return nullptr;
  }
}

void CodeGen_LLVM::compile(Stmt stmt, bool isFirst) {
  // throw logic_error("Not Implemented.");
  init_codegen();
  stmt.accept(this);
}

// ; ModuleID = 'my compiler'
// source_filename = "my compiler"
//
// %TensorType = type { i32, i32*, i32, i32*, i32*, i8***, i8*, i32 }
//
// define i32 @"my function"(%TensorType %0, %TensorType %1) {
// entry:
//   %2 = alloca %TensorType*, align 8
//   store %TensorType %0, %TensorType** %2, align 8
//   %3 = alloca %TensorType*, align 8
//   store %TensorType %1, %TensorType** %3, align 8
// }

void CodeGen_LLVM::codegen(Stmt stmt) {
  llvm::errs() << "LLVM CodeGen Visiting stmt\n";
  value = nullptr;
  stmt.accept(this);
}

llvm::Value *CodeGen_LLVM::codegen(Expr expr) {
  llvm::errs() << "LLVM CodeGen Visiting Expr\n";
  value = nullptr;
  expr.accept(this);
  taco_iassert(value) << "Codegen of expression " << expr
                      << " did not produce an LLVM value";
  return value;
}

void CodeGen_LLVM::visit(const Literal *e) {
  if (e->type.isFloat()) {
    if (e->type.getNumBits() == 32) {
      value = llvm::ConstantFP::get(llvmTypeOf(e->type), e->getValue<float>());
    } else {
      value = llvm::ConstantFP::get(llvmTypeOf(e->type), e->getValue<double>());
    }
  } else if (e->type.isUInt()) {
    switch (e->type.getNumBits()) {
    case 8:
      value =
          llvm::ConstantInt::get(llvmTypeOf(e->type), e->getValue<uint8_t>());
      return;
    case 16:
      value =
          llvm::ConstantInt::get(llvmTypeOf(e->type), e->getValue<uint16_t>());
      return;
    case 32:
      value =
          llvm::ConstantInt::get(llvmTypeOf(e->type), e->getValue<uint32_t>());
      return;
    case 64:
      value =
          llvm::ConstantInt::get(llvmTypeOf(e->type), e->getValue<uint64_t>());
      return;
    case 128:
      value = llvm::ConstantInt::get(llvmTypeOf(e->type),
                                     e->getValue<unsigned long long>());
      return;
    default:
      taco_ierror << "Unable to generate LLVM for literal " << e;
    }
  } else if (e->type.isInt()) {
    switch (e->type.getNumBits()) {
    case 8:
      value = llvm::ConstantInt::getSigned(llvmTypeOf(e->type),
                                           e->getValue<int8_t>());
      return;
    case 16:
      value = llvm::ConstantInt::getSigned(llvmTypeOf(e->type),
                                           e->getValue<int16_t>());
      return;
    case 32:
      value = llvm::ConstantInt::getSigned(llvmTypeOf(e->type),
                                           e->getValue<int32_t>());
      return;
    case 64:
      value = llvm::ConstantInt::getSigned(llvmTypeOf(e->type),
                                           e->getValue<int64_t>());
      return;
    case 128:
      value = llvm::ConstantInt::getSigned(llvmTypeOf(e->type),
                                           e->getValue<long long>());
      return;
    default:
      taco_ierror << "Unable to generate LLVM for literal " << e;
    }
  } else {
    taco_ierror << "Unable to generate LLVM for literal " << e;
  }
}

void CodeGen_LLVM::visit(const Var *op) {
  llvm::errs() << "LLVM CodeGen Visiting Var '" << op->name << "'\n";
  // llvm::errs() << getSymbol(op->name) << "\n";
  // value = this->Builder->CreateLoad()
}

void CodeGen_LLVM::visit(const Neg *op) {
  llvm::errs() << "LLVM CodeGen Visiting Neg\n";
}

void CodeGen_LLVM::visit(const Sqrt *op) {
  llvm::errs() << "LLVM CodeGen Visiting Sqrt\n";
}

void CodeGen_LLVM::visit(const Add *op) {
  llvm::errs() << "LLVM CodeGen Visiting Add\n";
  auto *a = codegen(op->a);
  auto *b = codegen(op->b);
  value = this->Builder->CreateAdd(a, b);
}

void CodeGen_LLVM::visit(const Sub *op) {
  llvm::errs() << "LLVM CodeGen Visiting Sub\n";
}

void CodeGen_LLVM::visit(const Mul *op) {
  llvm::errs() << "LLVM CodeGen Visiting Mul\n";
  auto *a = codegen(op->a);
  auto *b = codegen(op->b);
  value = this->Builder->CreateMul(a, b);
}

void CodeGen_LLVM::visit(const Div *op) {
  llvm::errs() << "LLVM CodeGen Visiting Div\n";
}

void CodeGen_LLVM::visit(const Rem *op) {
  llvm::errs() << "LLVM CodeGen Visiting Rem\n";
}

void CodeGen_LLVM::visit(const Min *op) {
  llvm::errs() << "LLVM CodeGen Visiting Min\n";
}

void CodeGen_LLVM::visit(const Max *op) {
  llvm::errs() << "LLVM CodeGen Visiting Max\n";
}

void CodeGen_LLVM::visit(const BitAnd *op) {
  llvm::errs() << "LLVM CodeGen Visiting BitAnd\n";
}

void CodeGen_LLVM::visit(const BitOr *op) {
  llvm::errs() << "LLVM CodeGen Visiting BitOr\n";
}

void CodeGen_LLVM::visit(const Eq *op) {
  llvm::errs() << "LLVM CodeGen Visiting Eq\n";
}

void CodeGen_LLVM::visit(const Neq *op) {
  llvm::errs() << "LLVM CodeGen Visiting Neq\n";
}

void CodeGen_LLVM::visit(const Gt *op) {
  llvm::errs() << "LLVM CodeGen Visiting Gt\n";
}

void CodeGen_LLVM::visit(const Lt *op) {
  llvm::errs() << "LLVM CodeGen Visiting Lt\n";
}

void CodeGen_LLVM::visit(const Gte *op) {
  llvm::errs() << "LLVM CodeGen Visiting Gte\n";
}

void CodeGen_LLVM::visit(const Lte *op) {
  llvm::errs() << "LLVM CodeGen Visiting Lte\n";
}

void CodeGen_LLVM::visit(const And *op) {
  llvm::errs() << "LLVM CodeGen Visiting And\n";
}

void CodeGen_LLVM::visit(const Or *op) {
  llvm::errs() << "LLVM CodeGen Visiting Or\n";
}

void CodeGen_LLVM::visit(const Cast *op) {
  llvm::errs() << "LLVM CodeGen Visiting Cast\n";
}

void CodeGen_LLVM::visit(const Call *op) {
  llvm::errs() << "LLVM CodeGen Visiting Call\n";
}

void CodeGen_LLVM::visit(const IfThenElse *op) {
  llvm::errs() << "LLVM CodeGen Visiting IfThenElse\n";
}

void CodeGen_LLVM::visit(const Case *op) {
  llvm::errs() << "LLVM CodeGen Visiting Case\n";
}

void CodeGen_LLVM::visit(const Switch *op) {
  llvm::errs() << "LLVM CodeGen Visiting Switch\n";
}

void CodeGen_LLVM::visit(const Load *op) {
  llvm::errs() << "LLVM CodeGen Visiting Load\n";
}

void CodeGen_LLVM::visit(const Malloc *op) {
  llvm::errs() << "LLVM CodeGen Visiting Malloc\n";
}

void CodeGen_LLVM::visit(const Sizeof *op) {
  llvm::errs() << "LLVM CodeGen Visiting Sizeof\n";
}

void CodeGen_LLVM::visit(const Store *op) {
  llvm::errs() << "LLVM CodeGen Visiting Store\n";
}

void CodeGen_LLVM::visit(const For *op) {
  llvm::errs() << "LLVM CodeGen Visiting For\n";
  cout << "op->start " << op->start << "\n";
  cout << op->var << "\n";
  cout << "op->end " << op->end << "\n";

  llvm::errs() << "codegen(op->start)\n";
  auto start = codegen(op->start);
  llvm::errs() << "codegen(op->end)\n";
  auto end = codegen(op->end);
  llvm::errs() << "done!\n";
  taco_iassert(start->getType()->isIntegerTy());
  taco_iassert(end->getType()->isIntegerTy());

  llvm::BasicBlock *pre_header = this->Builder->GetInsertBlock();

  // Create a new basic block for the loop
  llvm::BasicBlock *header =
      llvm::BasicBlock::Create(this->Context, "for_header", this->F);

  llvm::BasicBlock *body =
      llvm::BasicBlock::Create(this->Context, "for_body", this->F);

  llvm::BasicBlock *latch =
      llvm::BasicBlock::Create(this->Context, "for_latch", this->F);
  
  llvm::BasicBlock *exit =
      llvm::BasicBlock::Create(this->Context, "for_exit", this->F);

  // Connect body to latch
  this->Builder->SetInsertPoint(body);
  this->Builder->CreateBr(latch);

  // Connect latch to header
  this->Builder->SetInsertPoint(latch);
  this->Builder->CreateBr(header);  // this must change

  // Initialize header with PHI node
  this->Builder->SetInsertPoint(header);
  const Var *var = op->var.as<Var>();
  auto phi = this->Builder->CreatePHI(start->getType(), 2 /* num values */, var->name);

  // Compute increment
  this->Builder->SetInsertPoint(latch);
  auto incr = this->Builder->CreateAdd(phi, codegen(op->increment));

  // Add values to the PHI node
  phi->addIncoming(start, pre_header);
  phi->addIncoming(incr, latch);
  
  // Compute exit condition
  llvm::errs() << *phi << "\n" << *start << "\n" << *end << "\n";
  auto cond = this->Builder->CreateICmpSLT(phi, end);
  this->Builder->CreateCondBr(cond, body, exit);

  llvm::errs() << *this->F << "\n";

  // op->var.accept(this);
  // op->end.accept(this);
  // op->increment.accept(this);
  // op->contents.accept(this);
}

void CodeGen_LLVM::visit(const While *op) {
  llvm::errs() << "LLVM CodeGen Visiting While\n";
}

void CodeGen_LLVM::visit(const Block *op) {
  llvm::errs() << "LLVM CodeGen Visiting Block\n";
  for (const auto &s : op->contents) {
    s.accept(this);
  }
}

void CodeGen_LLVM::visit(const Scope *op) {
  llvm::errs() << "LLVM CodeGen Visiting Scope\n";
  pushScope();
  op->scopedStmt.accept(this);
  popScope();
}

void CodeGen_LLVM::init_codegen() {
  this->Builder = new llvm::IRBuilder<>(this->Context);

  auto i32 = llvm::Type::getInt32Ty(this->Context);
  auto i32p = i32->getPointerTo();

  auto u8 = llvm::Type::getInt8Ty(this->Context);
  auto u8p = u8->getPointerTo();
  auto u8ppp = u8->getPointerTo()->getPointerTo()->getPointerTo();

  /* See file include/taco/taco_tensor_t.h for the struct tensor definition */
  this->tensorType = llvm::StructType::create(this->Context,
                                              {
                                                  i32,   /* order */
                                                  i32p,  /* dimension */
                                                  i32,   /* csize */
                                                  i32p,  /* mode_ordering */
                                                  i32p,  /* mode_types */
                                                  u8ppp, /* indices */
                                                  u8p,   /* vals */
                                                  i32,   /* vals_size */
                                              },
                                              "TensorType");
}

void CodeGen_LLVM::visit(const Function *func) {
  auto M = std::make_unique<llvm::Module>("my compiler", this->Context);
  llvm::errs() << "LLVM CodeGen Visiting Function\n";

  // 1. find the arguments to @func
  FindVars varFinder(func->inputs, func->outputs, this);

  // 2. get the arguments types
  // Are all arguments tensors?

  // 3. convert the types to the LLVM correspondent ones
  int n_args = func->inputs.size() + func->outputs.size();
  std::vector<llvm::Type *> args;
  for (int i = 0; i < n_args; i++) {
    args.push_back(this->tensorType);
  }
  auto i32 = llvm::Type::getInt32Ty(this->Context);

  // 4. create a new function in the module with the given types
  this->F = llvm::Function::Create(llvm::FunctionType::get(i32, args, false),
                                   llvm::GlobalValue::ExternalLinkage,
                                   "my function", M.get());

  // 5. Create the first basic block
  this->Builder->SetInsertPoint(
      llvm::BasicBlock::Create(this->Context, "entry", F));

  // 6. Push arguments to symbol table
  pushScope();
  size_t idx = 0;
  for (auto &arg : F->args()) {
    // 6.1 Create ptr for tensor type
    auto ptr = Builder->CreateAlloca(tensorType->getPointerTo());
    Builder->CreateStore(&arg, ptr);

    auto var = idx < func->inputs.size()
                   ? func->inputs[idx++].as<Var>()
                   : func->outputs[idx++ % func->inputs.size()].as<Var>();
    // Shouldn't all arguments here be a parameter?
    // assert(var->is_parameter);
    pushSymbol(var->name, ptr);
  }

  llvm::errs() << *M << "\n";

  // 7. visit function body
  func->body.accept(this);
}

void CodeGen_LLVM::visit(const VarDecl *op) {
  llvm::errs() << "LLVM CodeGen Visiting VarDecl\n";
  // Create the pointer
  llvm::Type *rhs_llvm_type = llvmTypeOf(op->rhs.type());
  auto *ptr = this->Builder->CreateAlloca(rhs_llvm_type);

  // visit op rhs to produce a value
  // codegen ensures that a LLVM value was produced
  this->Builder->CreateStore(codegen(op->rhs), ptr);

  // Store the symbol/ptr in the symbol table
  const Var *lhs = op->var.as<Var>();
  llvm::errs() << "Storing symbol " << lhs->name << " on symbol table\n";
  pushSymbol(lhs->name, ptr);
}

void CodeGen_LLVM::visit(const Assign *op) {
  llvm::errs() << "LLVM CodeGen Visiting Assign\n";
}

void CodeGen_LLVM::visit(const Yield *op) {
  llvm::errs() << "LLVM CodeGen Visiting Yield\n";
}

void CodeGen_LLVM::visit(const Allocate *op) {
  llvm::errs() << "LLVM CodeGen Visiting Allocate\n";
}

void CodeGen_LLVM::visit(const Free *op) {
  llvm::errs() << "LLVM CodeGen Visiting Free\n";
}

void CodeGen_LLVM::visit(const Comment *op) {
  llvm::errs() << "LLVM CodeGen Visiting Comment\n";
}

void CodeGen_LLVM::visit(const BlankLine *op) {
  llvm::errs() << "LLVM CodeGen Visiting BlankLine\n";
}

void CodeGen_LLVM::visit(const Break *op) {
  llvm::errs() << "LLVM CodeGen Visiting Break\n";
}

void CodeGen_LLVM::visit(const Print *op) {
  llvm::errs() << "LLVM CodeGen Visiting Print\n";
}

void CodeGen_LLVM::visit(const GetProperty *op) {
  llvm::errs() << "LLVM CodeGen Visiting GetProperty\n";
  const std::string &name = op->tensor.as<Var>()->name;
  llvm::Value *tensor = getSymbol(name);
  llvm::errs() << "tensor : " << *tensor << "\n";
  switch (op->property) {
    case TensorProperty::Dimension:
    {
      auto *dim = this->Builder->CreateStructGEP(this->tensorType, tensor, (int)TensorProperty::Dimension, "tensor.dimension");
      llvm::errs() << "here!\n";
      value = this->Builder->CreateLoad(tensor, dim, "tensor.dimension");
      break;
    }
    case TensorProperty::Order:
    case TensorProperty::ComponentSize:
    case TensorProperty::ModeOrdering:
    case TensorProperty::ModeTypes:
    case TensorProperty::Indices:
    case TensorProperty::Values:
    case TensorProperty::ValuesSize:
    default:
      throw logic_error("Not implemented!");

  }
  llvm::errs() << "name: " << name << "\n";
  llvm::errs() << "Value: " << *value << "\n";
}

} // namespace ir
} // namespace taco