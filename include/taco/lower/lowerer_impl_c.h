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

    private:
        class Visitor;
        friend class Visitor;
        std::shared_ptr<Visitor> visitor;
    };
}


#endif //TACO_LOWERER_IMPL_C_H
