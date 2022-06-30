//
// Created by 张 on 2022/3/10.
//

#ifndef TACO_LOWERER_IMPL_C_H
#define TACO_LOWERER_IMPL_C_H

#include <memory>
#include "taco/lower/lowerer_impl_imperative.h"
namespace taco {
    class LowererImplC: public LowererImplImperative {
    public:
        LowererImplC();
        virtual ~LowererImplC() = default;

    protected:
        std::vector<ir::Stmt> codeToInitializeDenseAcceleratorArrays(Where where, bool parallel = false);
        std::vector<ir::Stmt> codeToInitializeTemporaryParallel(Where where, ParallelUnit parallelUnit);
        std::vector<ir::Stmt> codeToInitializeTemporary(Where where);
        std::pair<bool,bool> canAccelerateDenseTemp(Where where);
        std::vector<ir::Stmt> codeToInitializeLocalTemporaryParallel(Where where, ParallelUnit parallelUnit);
        /**
         * Generate code to initialize values array in range
         * [begin * size, (begin + 1) * size) with the fill value.
         */
        ir::Stmt initValues(ir::Expr tensor, ir::Expr initVal, ir::Expr begin, ir::Expr size);
        ir::Stmt lowerWhere(Where where);
        ir::Stmt lowerForall(Forall forall);

        /// Lower a forall that needs to be cloned so that one copy does not have guards
        /// used for vectorized and unrolled loops
        ir::Stmt lowerForallCloned(Forall forall);

    private:
        class Visitor;
        friend class Visitor;
        std::shared_ptr<Visitor> visitor;
    };
}


#endif //TACO_LOWERER_IMPL_C_H
