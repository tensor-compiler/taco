//
// Created by 张 on 2022/3/10.
//

#include "taco/lower/lowerer_impl_cuda.h"
#include "taco/index_notation/index_notation_visitor.h"

using namespace std;
using namespace taco::ir;

namespace taco {
    class LowererImplCUDA::Visitor : public IndexNotationVisitorStrict {
    public:
        Visitor(LowererImplCUDA* impl):impl(impl) {}
        Stmt lower(IndexStmt stmt) {
            this->stmt = Stmt();
            impl->accessibleIterators.scope();
            IndexStmtVisitorStrict::visit(stmt);
            impl->accessibleIterators.unscope();
            return this->stmt;
        }
        Expr lower(IndexExpr expr) {
            this->expr = Expr();
            IndexExprVisitorStrict::visit(expr);
            return this->expr;
        }

    private:
        LowererImplCUDA* impl;
        Expr expr;
        Stmt stmt;
        using IndexNotationVisitorStrict::visit;
        void visit(const AssignmentNode* node)    { stmt = impl->lowerAssignment(node); }
        void visit(const YieldNode* node)         { stmt = impl->lowerYield(node); }
        void visit(const ForallNode* node)        { stmt = impl->lowerForall(node); }
        void visit(const WhereNode* node)         { stmt = impl->lowerWhere(node); }
        void visit(const MultiNode* node)         { stmt = impl->lowerMulti(node); }
        void visit(const SuchThatNode* node)      { stmt = impl->lowerSuchThat(node); }
        void visit(const SequenceNode* node)      { stmt = impl->lowerSequence(node); }
        void visit(const AssembleNode* node)      { stmt = impl->lowerAssemble(node); }
        void visit(const AccessNode* node)        { expr = impl->lowerAccess(node); }
        void visit(const LiteralNode* node)       { expr = impl->lowerLiteral(node); }
        void visit(const NegNode* node)           { expr = impl->lowerNeg(node); }
        void visit(const AddNode* node)           { expr = impl->lowerAdd(node); }
        void visit(const SubNode* node)           { expr = impl->lowerSub(node); }
        void visit(const MulNode* node)           { expr = impl->lowerMul(node); }
        void visit(const DivNode* node)           { expr = impl->lowerDiv(node); }
        void visit(const SqrtNode* node)          { expr = impl->lowerSqrt(node); }
        void visit(const CastNode* node)          { expr = impl->lowerCast(node); }
        void visit(const CallIntrinsicNode* node) { expr = impl->lowerCallIntrinsic(node); }
        void visit(const CallNode* node)      { expr = impl->lowerTensorOp(node); }
        void visit(const ReductionNode* node)  {
            taco_ierror << "Reduction nodes not supported in concrete index notation";
        }
        void visit(const IndexVarNode* node)       { expr = impl->lowerIndexVar(node); }
    };
    LowererImplCUDA::LowererImplCUDA(): visitor(new Visitor(this)) {}
}