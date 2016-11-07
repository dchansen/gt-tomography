#pragma once

#include "primalDualOperator.h"
#include <cuNDArray.h>
namespace Gadgetron {
    template<class T>
    class cuWTVPrimalDualOperator : public primalDualOperator<cuNDArray<T>> {


    public:
        cuWTVPrimalDualOperator() : primalDualOperator<cuNDArray<T>>(), alpha(0){}
        cuWTVPrimalDualOperator(T alpha_ ) : primalDualOperator<cuNDArray<T>>(),alpha(alpha_){}
        virtual void primalDual(cuNDArray<T>* in, cuNDArray<T>* out,T sigma, bool accumulate) override;

        virtual void  update_weights(cuNDArray<T>* x) override;



    private:
        T alpha;

    };
}