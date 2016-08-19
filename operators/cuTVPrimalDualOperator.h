#pragma once

#include "primalDualOperator.h"
#include <cuNDArray.h>
namespace Gadgetron {
    template<class T>
    class cuTVPrimalDualOperator : public primalDualOperator<cuNDArray<T>> {


    public:
        cuTVPrimalDualOperator() : alpha(0),weight(0){}
        cuTVPrimalDualOperator(T alpha_ ) : alpha(alpha_),weight(0){}
        virtual void primalDual(cuNDArray<T>* in, cuNDArray<T>* out,T sigma, bool accumulate) override;

        void set_weight(T weight_){ weight = weight_;}

    private:
        T alpha;
        T weight;
    };
}