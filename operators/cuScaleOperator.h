#pragma once

#include "cuNDArray_math.h"
#include "linearOperator.h"



namespace Gadgetron {
    template<class T,unsigned int D>
    class cuScaleOperator : public linearOperator<cuNDArray<T>> {

    public:
        cuScaleOperator() : linearOperator<cuNDArray<T>>() {

        }
        virtual void mult_M(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate=false) override;
        virtual void mult_MH(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate=false) override;

    };
}