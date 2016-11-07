#pragma once

#include "primalDualOperator.h"
#include <cuNDArray.h>
namespace Gadgetron {
    template<class T>
    class cuTVPrimalDualOperator : public primalDualOperator<cuNDArray<T>> {


    public:
        cuTVPrimalDualOperator() : primalDualOperator<cuNDArray<T>>(), alpha(0){}
        cuTVPrimalDualOperator(T alpha_ ) : primalDualOperator<cuNDArray<T>>(),alpha(alpha_){}
        virtual void primalDual(cuNDArray<T>* in, cuNDArray<T>* out,T sigma, bool accumulate) override;



    private:
        T alpha;

    };
}