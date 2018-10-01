#include "codegen/codegen_llvm.h"
#include "codegen/llvm_headers.h"
#include "taco/util/strings.h"

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
    init_context();
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

bool CodeGen_LLVM::containsSymbol(const std::string &name) {
  return symbolTable.contains(name);
}

namespace {

// Convert from taco type to LLVM type
llvm::Type *llvmTypeOf(LLVMContext *context, Datatype t) {
  taco_tassert(!t.isComplex()) << "LLVM codegen for complex not yet supported";
  
  if (t.isFloat()) {
    switch (t.getNumBits()) {
      case 32:
        return llvm::Type::getFloatTy(*context);
      case 64:
        return llvm::Type::getDoubleTy(*context);
      default:
        taco_ierror << "Unable to find LLVM type for " << t;
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
  value = builder->CreateLoad(getSymbol(e->name));
}

void CodeGen_LLVM::visit(const Neg *e) {
  Expr zero = Literal::make(0);
  value = codegen(Sub::make(Cast::make(zero, e->type), e->a, e->type));
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
  // taco's Max IR node only deals with two operands.
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
    value = builder->CreateFCmpOEQ(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateICmpEQ(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Neq *e) {
  if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    value = builder->CreateFCmpONE(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateICmpNE(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Gt *e) {
  if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    value = builder->CreateFCmpOGT(codegen(e->a), codegen(e->b));
  } else if (e->type.isUInt()){
    value = builder->CreateICmpUGT(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateICmpSGT(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Lt *e) {
  if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    value = builder->CreateFCmpOLT(codegen(e->a), codegen(e->b));
  } else if (e->type.isUInt()){
    value = builder->CreateICmpULT(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateICmpSLT(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Gte *e) {
 if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    value = builder->CreateFCmpOGE(codegen(e->a), codegen(e->b));
  } else if (e->type.isUInt()){
    value = builder->CreateICmpUGE(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateICmpSGE(codegen(e->a), codegen(e->b));
  }
}
void CodeGen_LLVM::visit(const Lte *e) {
 if (e->type.isFloat()) {
    // TODO: This says neither can be a NaN.  May want to use a different
    // instruction
    value = builder->CreateFCmpOLE(codegen(e->a), codegen(e->b));
  } else if (e->type.isUInt()){
    value = builder->CreateICmpULE(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateICmpSLE(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const And *e) {
  value = builder->CreateAnd(codegen(e->a), codegen(e->b));
}

void CodeGen_LLVM::visit(const Or *e) {
  value = builder->CreateOr(codegen(e->a), codegen(e->b));
}

void CodeGen_LLVM::visit(const Cast *e) {
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
  if (e->otherwise != nullptr) {
    codegen(e->otherwise);
  }
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
  value = builder->CreateCall(sqrtFunction, codegen(e->a));
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
  std::copy(f->outputs.begin(), f->outputs.end(), std::back_inserter(currentFunctionArgs));
  std::copy(f->inputs.begin(), f->inputs.end(), std::back_inserter(currentFunctionArgs));

  // get the type for the parameters
  std::vector<llvm::Type*> argTypes(currentFunctionArgs.size());
  for (size_t i=0; i<currentFunctionArgs.size(); i++) {
    argTypes[i] = tacoTensorType->getPointerTo();
  }
  
  // our return type is an int32_t
  auto *functionType = FunctionType::get(llvm::Type::getInt32Ty(*context), argTypes, false);
  
  // create a declaration for our function
  function = llvm::Function::Create(functionType,
                                    llvm::GlobalValue::LinkageTypes::ExternalLinkage,
                                    f->name,
                                    module.get());
  
  // inputs/outputs cannot alias
  for (size_t i=0; i<currentFunctionArgs.size(); i++) {
    function->addParamAttr(i, Attribute::NoAlias);
  }
  
  // create the initial basic block & set insertion point
  builder->SetInsertPoint(BasicBlock::Create(*context, "entry", function));
  
  // add arguments to symbol table
  pushScope();
  size_t argIndex = 0;
  for (auto &arg : function->args()) {
    auto argLoc = builder->CreateAlloca(tacoTensorType->getPointerTo());
    builder->CreateStore(&arg, argLoc);
    pushSymbol((currentFunctionArgs[argIndex]).as<Var>()->name, argLoc);
    argIndex++;
  }
}

void CodeGen_LLVM::endFunc(const Function *f) {
  // generate code for repacking tensor properties

  // return the success code
  builder->CreateRet(ConstantInt::get(llvm::Type::getInt32Ty(*context), 0));
  
  // pop arguments
  popScope();
  
  // clear arguments
  currentFunctionArgs.clear();
}


void CodeGen_LLVM::visit(const Function *f) {

  //std::cerr << "Codegen of function:\n" << (Stmt)f << "\n";
  
  // use a helper function to generate the function declaration and argument
  // unpacking code
  beginFunc(f);
  
  // Generate the function body
  f->body.accept(this);
  
  // Use a helper function to cleanup
  endFunc(f);
  
  // construct a shim for calling via void**
  auto shimFunctionType = FunctionType::get(llvm::Type::getInt32Ty(*context),
                                            llvm::Type::getInt8Ty(*context)->getPointerTo()->getPointerTo(),
                                            false);
  auto shimFunction = llvm::Function::Create(shimFunctionType,
                                             llvm::GlobalValue::LinkageTypes::ExternalLinkage,
                                             "_shim_"+f->name,
                                             module.get());
  builder->SetInsertPoint(BasicBlock::Create(*context, "entry", shimFunction));

  auto parameterPack = shimFunction->arg_begin();
  
  std::vector<llvm::Value*> shimArgs;
  for (size_t i=0; i<f->inputs.size() + f->outputs.size(); i++) {
    auto GEP = builder->CreateInBoundsGEP(parameterPack, {codegen(Cast::make(Literal::make(i), Int32))});
    auto load = builder->CreateLoad(GEP);
    shimArgs.push_back(builder->CreateBitCast(load, tacoTensorType->getPointerTo()));
  }
  
  builder->CreateRet(builder->CreateCall(function, shimArgs));
  
  // always validate the LLVM IR
  llvm::verifyFunction(*function, &errs());
  llvm::verifyModule(*module.get(), &errs());

  //module->print(llvm::errs(), nullptr);

}

void CodeGen_LLVM::visit(const Allocate* e) {
  Value *storeLoc, *oldPtr;
  
  if (e->var.as<GetProperty>()) {
    storeLoc = visit_GetProperty(e->var.as<GetProperty>(), false);
  } else {
    storeLoc = codegen(e->var);
  }
  
  // if it's not a realloc, we don't know if the pointer is valid, so we'll just
  // use a nullptr.  otherwise, use the old pointer
  if (!e->is_realloc) {
    oldPtr = llvm::ConstantPointerNull::get(llvm::Type::getInt8PtrTy(*context));
  } else {
    oldPtr = builder->CreateLoad(storeLoc);
  }
  
  // construct the call
  std::vector<llvm::Type*> argTypes = {llvm::Type::getInt8PtrTy(*context), llvm::Type::getInt64Ty(*context)};
  auto functionType = FunctionType::get(llvm::Type::getInt8PtrTy(*context),
                                          argTypes, false);
  auto mallocFunction = module->getOrInsertFunction("realloc", functionType);
  auto call = builder->CreateCall(mallocFunction,
      {oldPtr, codegen(Mul::make(Cast::make(e->num_elements, Int64),
                                                      Cast::make(Literal::make(e->var.type().getNumBytes()), Int64)))});
  
  // finally, store it
  builder->CreateStore(call, storeLoc);
  
}

void CodeGen_LLVM::visit(const Block* e) {
  for (auto &s : e->contents) {
    codegen(s);
  }
}

// TODO: currently, none of the new lowering tests generate while loops.
// This needs to be tested.
void CodeGen_LLVM::visit(const While* e) {
  taco_tassert(e->kind == LoopKind::Serial) <<
    "Only serial loop codegen supported by LLVM backend";
  
  BasicBlock *preheader_bb = builder->GetInsertBlock();
  
  // new basic blocks for the loop & loop end
  BasicBlock *loop_bb = BasicBlock::Create(*context, "while", function);
  BasicBlock *after_bb = BasicBlock::Create(*context, "end_while", function);
  
  // entry condition
  auto checkValue = codegen(e->cond);
  builder->CreateCondBr(checkValue, loop_bb, after_bb);
  builder->SetInsertPoint(loop_bb);
  
  // create phi node
  PHINode *phi = builder->CreatePHI(checkValue->getType(), 2);
  phi->addIncoming(checkValue, preheader_bb);
  
  // codegen body
  codegen(e->contents);
  
  // create unconditional branch to check
  auto branchToCheck = builder->CreateBr(preheader_bb);
  
  // phi backedge
  phi->addIncoming(branchToCheck, builder->GetInsertBlock());
  
  // set the insert point for after the loop
  builder->SetInsertPoint(after_bb);
  
}

void CodeGen_LLVM::visit(const For* e) {
  taco_tassert(e->kind == LoopKind::Serial) <<
    "Only serial loop codegen supported by LLVM backend";
  
  // the start value is emitted first; we don't put it in scope yet
  auto startValue = codegen(e->start);
  auto endValue = codegen(e->end);
  
  BasicBlock *preheader_bb = builder->GetInsertBlock();
  
  // create a stack value to store the value of the loop iterator
  auto loopVarVal = builder->CreateAlloca(startValue->getType());
  
  // new basic blocks for the loop & loop end
  BasicBlock *loop_bb = BasicBlock::Create(*context, "for", function);
  BasicBlock *after_bb = BasicBlock::Create(*context, "end_for", function);
  
  // entry condition
  taco_iassert(startValue->getType() == endValue->getType());
  taco_iassert(e->start.type() == e->var.type());
  auto entryCondition = builder->CreateICmpSLT(startValue, endValue);
  builder->CreateCondBr(entryCondition, loop_bb, after_bb);
  builder->SetInsertPoint(loop_bb);

  // create phi node
  PHINode *phi = builder->CreatePHI(startValue->getType(), 2);
  phi->addIncoming(startValue, preheader_bb);
  
  // store the value of the phi node
  builder->CreateStore(phi, loopVarVal);
  
  // add entry for loop variable to symbol table
  auto loopVar = e->var.as<Var>();
  taco_iassert(loopVar) << "Loop variable is not a Var";
  pushScope();
  pushSymbol(loopVar->name, loopVarVal);
  
  // codegen body
  codegen(e->contents);
  
  // update loop variable
  auto nextValue = builder->CreateNSWAdd(phi, codegen(e->increment));
  
  // phi backedge
  phi->addIncoming(nextValue, builder->GetInsertBlock());
  
  // check whether to exit loop
  auto endCondition = builder->CreateICmpSLT(nextValue, endValue);
  builder->CreateCondBr(endCondition, loop_bb, after_bb);
  
  // pop the scope
  popScope();
  
  // set the insert point for after the loop
  builder->SetInsertPoint(after_bb);
  
}

void CodeGen_LLVM::visit(const Assign* e) {
  
  auto val = codegen(e->rhs);

  llvm::Value *var;
  if (e->lhs.as<Var>()) {
    var = getSymbol(e->lhs.as<Var>()->name);
  } else if (e->lhs.as<GetProperty>()) {
    var = visit_GetProperty(e->lhs.as<GetProperty>(), false);
  }
  else {
    var = codegen(e->lhs);
  }
  value = builder->CreateStore(val, var);
}

void CodeGen_LLVM::visit(const Load* e) {
  Value *loc = codegen(e->loc);
  Value *gep;
  if (e->arr.as<GetProperty>()) {
    gep = visit_GetProperty(e->arr.as<GetProperty>(), true);
    gep = builder->CreateInBoundsGEP(gep, loc);
  } else {
    auto arr = codegen(e->arr);
    gep = builder->CreateInBoundsGEP(arr, loc);

  }
  
  // load from the GEP
  value  = builder->CreateLoad(gep);
}

void CodeGen_LLVM::visit(const Store* e) {
  Value *loc = codegen(e->loc);
  Value *gep;
  if (e->arr.as<GetProperty>()) {
    gep = visit_GetProperty(e->arr.as<GetProperty>(), true);
    gep = builder->CreateInBoundsGEP(gep, loc);
  } else {
    auto arr = codegen(e->arr);
    gep = builder->CreateInBoundsGEP(arr, loc);
  }

  // store
  builder->CreateStore(codegen(e->data), gep);
}

void CodeGen_LLVM::visit(const VarDecl* e) {
  auto var = builder->CreateAlloca(llvmTypeOf(context, e->rhs.type()));
  builder->CreateStore(codegen(e->rhs), var);
  pushSymbol(util::toString(e->var), var);
}

void CodeGen_LLVM::visit(const Print* e) {

  // create the parameters to pass in
  std::vector<llvm::Value*> params;
 
  // create a global variable string for the format string
  params.push_back(builder->CreateGlobalStringPtr(e->fmt));
  
  for (auto &x: e->params) {
    params.push_back(codegen(x));
  }
  
  // get the printf function in the module
  auto functionType = FunctionType::get(llvm::Type::getInt32Ty(*context),
                                        {llvm::Type::getInt8PtrTy(*context)}, true);
  auto printFunc = module->getOrInsertFunction("printf", functionType);
  
  // generate the call.  we ignore the return value, as Print is stmt
  builder->CreateCall(printFunc, params);
  
}

namespace {
std::map<TensorProperty, int> indexForProp =
  {
   {TensorProperty::Order, 0},
   {TensorProperty::Dimension, 1},
   {TensorProperty::ComponentSize, 2},
   {TensorProperty::ModeOrdering, 3},
   {TensorProperty::ModeTypes, 4},
   {TensorProperty::Indices, 5},
   {TensorProperty::Values, 6},
   {TensorProperty::ValuesSize, 7}
  };

} // anonymous namespace

// TODO: This does not currently handle all the possible GetProperty cases.  In addition,
// it does not correctly respect mode ordering.
llvm::Value* CodeGen_LLVM::visit_GetProperty(const GetProperty *e, bool loadPtr) {
  auto tensor = builder->CreateLoad(getSymbol(e->tensor.as<Var>()->name));
  // first, we access the correct struct field
  auto val = builder->CreateInBoundsGEP(tensor,
                     {codegen(Literal::make(0)),
                      codegen(Literal::make(indexForProp[e->property]))
                     });

  // depending on the property, we have to access further pointers
  if (e->property == TensorProperty::Dimension ||
      e->property == TensorProperty::ModeOrdering ||
      e->property == TensorProperty::ModeTypes ||
      e->property == TensorProperty::Indices) {
    val = builder->CreateInBoundsGEP(builder->CreateLoad(val), codegen(Literal::make(e->mode)));
  }
  
  // if it's into indices, we then have to access which index
  if (e->property == TensorProperty::Indices) {
    val = builder->CreateInBoundsGEP(builder->CreateLoad(val), codegen(Literal::make(e->index)));
  }
  
  if (loadPtr) {
    val = builder->CreateLoad(val);
    
    // if it's vals, cast to the correct type
    if (e->property == TensorProperty::Values) {
      val = builder->CreateBitCast(val, llvm::Type::getDoublePtrTy(*context));
    } else if (e->property == TensorProperty::Indices) {
      // cast to the correct type
      val = builder->CreateBitCast(val, llvm::Type::getInt32PtrTy(*context));
    }
  }
  
  return val;
}

void CodeGen_LLVM::visit(const GetProperty* e) {
  value = visit_GetProperty(e, true);
}

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
  mode_typesType = int32Type->getPointerTo();
  indicesType = uint8Type->getPointerTo()
                         ->getPointerTo()
                         ->getPointerTo();
  valsType = uint8Type->getPointerTo();
  vals_sizeType = int32Type;
  
  tacoTensorType = llvm::StructType::create(*context,
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

void CodeGen_LLVM::writeToFile(std::string fileName) {
  std::error_code EC;
  raw_fd_ostream outputStream(fileName, EC, llvm::sys::fs::OpenFlags::F_None);
  WriteBitcodeToFile(*(module.get()), outputStream);
  outputStream.flush();
}

void CodeGen_LLVM::optimizeModule() {
  //std::cerr << "Before optimization:\n";
  //module->print(llvm::errs(), nullptr);
  
  llvm::legacy::FunctionPassManager functionPassManager(module.get());
  llvm::legacy::PassManager modulePassManager;
  
  PassManagerBuilder b;
  b.OptLevel = 3;
  b.LoopVectorize = true;
  b.SLPVectorize = true;
  
  b.populateModulePassManager(modulePassManager);
  b.populateFunctionPassManager(functionPassManager);
  
  functionPassManager.doInitialization();
  for (auto i = module->begin(); i != module->end(); i++ ) {
    functionPassManager.run(*i);
  }
  
  functionPassManager.doFinalization();
  modulePassManager.run(*module);

  //std::cerr << "After optimization:\n";
  //module->print(llvm::errs(), nullptr);
}

} // namespace ir
} // namespace taco
