#include "codegen/codegen_llvm.h"
#include "codegen/llvm_headers.h"

namespace taco {
namespace ir {

using namespace llvm;

bool CodeGen_LLVM::LLVMInitialized = false;

CodeGen_LLVM::CodeGen_LLVM(const Target &target,
               llvm::LLVMContext &context) :
  target(target),
  function(nullptr),
  context(&context),
  builder(nullptr),
  value(nullptr) {
  if (!LLVMInitialized) {
    module = make_unique<llvm::Module>("taco module", context);
  }
}
CodeGen_LLVM::~CodeGen_LLVM() { }

  
void CodeGen_LLVM::compile(Stmt stmt, bool isFirst) {
  codegen(stmt);
}

void CodeGen_LLVM::codegen(Stmt stmt) {
  value = nullptr;
  stmt.accept(this);
}

Value *CodeGen_LLVM::codegen(Expr expr) {
  value = nullptr;
  expr.accept(this);
  taco_iassert(value) << "Codegen of expression " << expr <<
    " did not produce an LLVM value";
  return value;
}

void CodeGen_LLVM::pushScope() {
  symbolTable.scope();
}

void CodeGen_LLVM::popScope() {
  symbolTable.unscope();
}

llvm::Value* CodeGen_LLVM::getSymbol(const std::string &name) {
  return symbolTable.get(name);
}

void CodeGen_LLVM::pushSymbol(const std::string &name, llvm::Value *value) {
  symbolTable.insert({name, value});
}

namespace {

llvm::Type *llvmTypeOf(LLVMContext *context, Datatype t) {
  taco_tassert(!t.isComplex()) << "LLVM codegen for complex not yet supported";
  
  if (t.isFloat()) {
    switch (t.getNumBits()) {
      case 32:
        return llvm::Type::getFloatTy(*context);
      case 64:
        return llvm::Type::getDoubleTy(*context);
      default:
        taco_ierror << "Unabe to find LLVM type for " << t;
        return nullptr;
    }
  } else {
    return llvm::Type::getIntNTy(*context, t.getNumBits());
  }
}

} // anonymous namespace


void CodeGen_LLVM::visit(const Literal *e) {
  if (e->type.isFloat()) {
    if (e->type.getNumBits() == 32) {
      value = ConstantFP::get(llvmTypeOf(context, e->type), e->getValue<float>());
    } else {
      value = ConstantFP::get(llvmTypeOf(context, e->type), e->getValue<double>());
    }
  } else if (e->type.isUInt()) {
    switch (e->type.getNumBits()) {
      case 8:
        value = ConstantInt::get(llvmTypeOf(context, e->type), e->getValue<uint8_t>());
        return;
      case 16:
        value = ConstantInt::get(llvmTypeOf(context, e->type), e->getValue<uint16_t>());
        return;
      case 32:
        value = ConstantInt::get(llvmTypeOf(context, e->type), e->getValue<uint32_t>());
        return;
      case 64:
        value = ConstantInt::get(llvmTypeOf(context, e->type), e->getValue<uint64_t>());
        return;
      case 128:
        value = ConstantInt::get(llvmTypeOf(context, e->type), e->getValue<unsigned long long>());
        return;
      default:
        taco_ierror << "Unable to generate LLVM for literal " << e;
    }
  } else if (e->type.isInt()) {
    switch (e->type.getNumBits()) {
      case 8:
        value = ConstantInt::getSigned(llvmTypeOf(context, e->type), e->getValue<int8_t>());
        return;
      case 16:
        value = ConstantInt::getSigned(llvmTypeOf(context, e->type), e->getValue<int16_t>());
        return;
      case 32:
        value = ConstantInt::getSigned(llvmTypeOf(context, e->type), e->getValue<int32_t>());
        return;
      case 64:
        value = ConstantInt::getSigned(llvmTypeOf(context, e->type), e->getValue<int64_t>());
        return;
      case 128:
        value = ConstantInt::getSigned(llvmTypeOf(context, e->type), e->getValue<long long>());
        return;
      default:
        taco_ierror << "Unable to generate LLVM for literal " << e;
    }
  } else {
    taco_ierror << "Unable to generate LLVM for literal " << e;
  }
}

void CodeGen_LLVM::visit(const Var *e) {
  value = getSymbol(e->name);
}

void CodeGen_LLVM::visit(const Neg *e) {
  if (e->type.isFloat()) {
    value = builder->CreateFSub(0, codegen(e));
  } else {
    value = builder->CreateSub(0, codegen(e));
  }
}

void CodeGen_LLVM::visit(const Add *e) {
  if (e->type.isFloat()) {
    value = builder->CreateFAdd(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateAdd(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Sub *e) {
  if (e->type.isFloat()) {
    value = builder->CreateFSub(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateSub(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Mul *e) {
  if (e->type.isFloat()) {
    value = builder->CreateFMul(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateMul(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Div *e) {
  // TODO: Turning integer division into shifts/etc can sometimes be
  // fruitful.  We should implement the same ops as Halide.
  if (e->type.isFloat()) {
    value = builder->CreateFDiv(codegen(e->a), codegen(e->b));
  } else if (e->type.isUInt()){
    value = builder->CreateExactUDiv(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateExactSDiv(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Min *e) {
  // LLVM's minnum intrinsic only does binary ops
  value = builder->CreateMinNum(codegen(e->operands[0]),
                                codegen(e->operands[1]));
  for (size_t i=2; i<e->operands.size(); i++) {
    value = builder->CreateMinNum(value, codegen(e->operands[i]));
  }
}

void CodeGen_LLVM::visit(const Max *e) {
  // Taco's Max IR node only deals with two operands.
  value = builder->CreateMaxNum(codegen(e->a),
                                codegen(e->b));
}

void CodeGen_LLVM::visit(const BitAnd *e) {
  value = builder->CreateAnd(codegen(e->a), codegen(e->b));
}

void CodeGen_LLVM::visit(const BitOr *e) {
  value = builder->CreateOr(codegen(e->a), codegen(e->b));
}

void CodeGen_LLVM::visit(const Eq *e) {
  if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    builder->CreateFCmpOEQ(codegen(e->a), codegen(e->b));
  } else {
    builder->CreateICmpEQ(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Neq *e) {
  if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    builder->CreateFCmpONE(codegen(e->a), codegen(e->b));
  } else {
    builder->CreateICmpNE(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Gt *e) {
  if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    builder->CreateFCmpOGT(codegen(e->a), codegen(e->b));
  } else if (e->type.isUInt()){
    builder->CreateICmpUGT(codegen(e->a), codegen(e->b));
  } else {
    builder->CreateICmpSGT(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Lt *e) {
  if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    builder->CreateFCmpOLT(codegen(e->a), codegen(e->b));
  } else if (e->type.isUInt()){
    builder->CreateICmpULT(codegen(e->a), codegen(e->b));
  } else {
    builder->CreateICmpSLT(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Gte *e) {
 if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    builder->CreateFCmpOGE(codegen(e->a), codegen(e->b));
  } else if (e->type.isUInt()){
    builder->CreateICmpUGE(codegen(e->a), codegen(e->b));
  } else {
    builder->CreateICmpSGE(codegen(e->a), codegen(e->b));
  }
}
void CodeGen_LLVM::visit(const Lte *e) {
 if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    builder->CreateFCmpOLE(codegen(e->a), codegen(e->b));
  } else if (e->type.isUInt()){
    builder->CreateICmpULE(codegen(e->a), codegen(e->b));
  } else {
    builder->CreateICmpSLE(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const And *e) {
  value = builder->CreateAnd(codegen(e->a), codegen(e->b));
}

void CodeGen_LLVM::visit(const Or *e) {
  value = builder->CreateOr(codegen(e->a), codegen(e->b));
}

void CodeGen_LLVM::visit(const Cast *e) {
  // TODO: Not sure about whether these are the correct instructions.
  if (e->type.isFloat()) {
    value = builder->CreateFPCast(codegen(e->a), llvmTypeOf(context, e->type));
  } else {
    value = builder->CreateIntCast(codegen(e->a), llvmTypeOf(context, e->type),
            !e->type.isUInt());
  }
}

void CodeGen_LLVM::visit(const IfThenElse* e) {
  // Create the basic blocks
  BasicBlock *true_bb = BasicBlock::Create(*context, "true_bb", function);
  BasicBlock *false_bb = BasicBlock::Create(*context, "false_bb", function);
  BasicBlock *after_bb = BasicBlock::Create(*context, "after_bb", function);

  // Create condition
  builder->CreateCondBr(codegen(e->cond), true_bb, false_bb);
  
  // true case
  builder->SetInsertPoint(true_bb);
  codegen(e->then);
  builder->CreateBr(after_bb);
  
  // false case
  builder->SetInsertPoint(false_bb);
  codegen(e->otherwise);
  builder->CreateBr(after_bb);
  
  builder->SetInsertPoint(after_bb);
}

void CodeGen_LLVM::visit(const Comment* e) {
  // No-op
}

void CodeGen_LLVM::visit(const BlankLine*) {
  // No-op
}

void CodeGen_LLVM::visit(const Scope* e) {
  pushScope();
  codegen(e->scopedStmt);
  popScope();
}

void CodeGen_LLVM::visit(const Sqrt* e) {
  std::vector<llvm::Type*> argTypes = {llvmTypeOf(context, e->a.type())};
  llvm::Function *sqrtFunction = Intrinsic::getDeclaration(module.get(), Intrinsic::sqrt, argTypes);
  builder->CreateCall(sqrtFunction, codegen(e->a));
}

namespace {
  Stmt caseToIfThenElse(std::vector<std::pair<Expr,Stmt>> clauses, bool alwaysMatch) {
    std::vector<std::pair<Expr,Stmt>> rest(clauses.begin()+1, clauses.end());
    if (rest.size() == 0) {
      // if alwaysMatch is true, then this one goes into the else clause,
      // otherwise, we generate an empty else clause
      return !alwaysMatch ? clauses[0].second :
        IfThenElse::make(clauses[0].first, clauses[0].second, Comment::make(""));
    } else {
      return IfThenElse::make(clauses[0].first,
                            clauses[0].second,
                            caseToIfThenElse(rest, alwaysMatch));
    }
  }
} // anonymous namespace

// For Case statements, we turn them into nested If/Then/Elses and codegen that
void CodeGen_LLVM::visit(const Case* e) {
  codegen(caseToIfThenElse(e->clauses, e->alwaysMatch));
}

void CodeGen_LLVM::visit(const Switch* e) {
  // By default, we do nothing, so this is the default jump target
  BasicBlock *after_bb = BasicBlock::Create(*context, "after_bb", function);
  
  // Create the condition
  auto cond = codegen(e->controlExpr);
  // Create the switch
  auto theSwitch = builder->CreateSwitch(cond, after_bb, e->cases.size());

  // Create all the basic blocks
  std::vector<BasicBlock*> basicBlocks;
  for (size_t i=0; i<e->cases.size(); i++) {
    basicBlocks.push_back(BasicBlock::Create(*context, "case_bb", function));
    builder->SetInsertPoint(basicBlocks[i]);
    codegen(e->cases[i].second);
    // set a jump to the after block
    builder->CreateBr(after_bb);
    // TODO: Make sure this works for ints and unsigned ints
    taco_iassert(e->cases[i].first.as<Literal>() && e->cases[i].first.type().isUInt());
    auto c = ConstantInt::get(llvmTypeOf(context, e->cases[i].first.type()), e->cases[i].first.as<Literal>()->getValue<int32_t>());
    theSwitch->addCase(static_cast<ConstantInt*>(c), basicBlocks[i]);
  }
  
  // Set the insertion point
  builder->SetInsertPoint(after_bb);
}


void CodeGen_LLVM::beginFunc(const Function *f) {
  std::copy(f->inputs.begin(), f->inputs.end(), std::back_inserter(currentFunctionArgs));
  std::copy(f->outputs.begin(), f->outputs.end(), std::back_inserter(currentFunctionArgs));

  // get the type for the parameters
  std::vector<llvm::Type*> argTypes(currentFunctionArgs.size());
  for (size_t i=0; i<currentFunctionArgs.size(); i++) {
    argTypes[i] = tacoTensorType->getPointerTo();
  }
  
  // our return type is an int32_t
  auto *functionType = FunctionType::get(llvm::Type::getInt32Ty(*context), argTypes, false);
  
  // create a declaration for our function
  
}

void CodeGen_LLVM::endFunc(const Function *f) {

}


void CodeGen_LLVM::visit(const Function *f) {
  // use a helper function to generate the function declaration and argument
  // unpacking code
  beginFunc(f);
  
  // Generate the function body
  f->body.accept(this);
  
  // Use a helper function to cleanup
  endFunc(f);
}

void CodeGen_LLVM::visit(const Load*) { }
void CodeGen_LLVM::visit(const Store*) { }
void CodeGen_LLVM::visit(const For*) { }
void CodeGen_LLVM::visit(const While*) { }
void CodeGen_LLVM::visit(const Block*) { }
void CodeGen_LLVM::visit(const VarAssign*) { }
void CodeGen_LLVM::visit(const Allocate*) { }
void CodeGen_LLVM::visit(const Print*) { }
void CodeGen_LLVM::visit(const GetProperty*) { }

void CodeGen_LLVM::visit(const Rem*) { /* Will be removed from IR */ }

void CodeGen_LLVM::init_context() {
  // Get rid of any previous IRBuilder, which could be using a different
  // LLVM context
  delete builder;
  builder = new IRBuilder<>(*context);
  
  // TODO: set fastmath flags
  
  // Set up useful types
  // TODO: we probably cannot assume that an enum is int32_t
  auto int32Type = llvm::Type::getInt32Ty(*context);
  auto uint8Type = llvm::Type::getInt8Ty(*context);
  
  orderType = int32Type;
  dimensionsType = int32Type->getPointerTo();
  csizeType = int32Type;
  mode_orderingType = int32Type->getPointerTo();
  indicesType = uint8Type->getPointerTo()
                         ->getPointerTo()
                         ->getPointerTo();
  valsType = uint8Type->getPointerTo();
  vals_sizeType = int32Type;
  
  tacoTensorType = llvm::StructType::get(*context,
                   { orderType,
                     dimensionsType,
                     csizeType,
                     mode_orderingType,
                     mode_typesType,
                     indicesType,
                     valsType,
                     vals_sizeType
                   },
                   "taco_tensor_t");
  
}


} // namespace ir
} // namespace taco
