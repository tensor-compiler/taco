//
// Created by 张 on 2022/3/12.
//

#ifndef TACO_LOWERER_IMPL_CUDA_H
#define TACO_LOWERER_IMPL_CUDA_H

#include <memory>
#include "taco/lower/lowerer_impl_imperative.h"
namespace taco {
    class LowererImplCUDA: public LowererImplImperative {
    public:
        LowererImplCUDA();
        virtual ~LowererImplCUDA() = default;

    private:
        class Visitor;
        friend class Visitor;
        std::shared_ptr<Visitor> visitor;
    };
}

#endif //TACO_LOWERER_IMPL_CUDA_H
