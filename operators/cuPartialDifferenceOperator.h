#pragma once

#include "cuNDArray_math.h"
#include "linearOperator.h"



namespace Gadgetron {
    template<class T,unsigned int D>
    class cuPartialDifferenceOperator : public linearOperator<cuNDArray<T>> {

    public:
        cuPartialDifferenceOperator(int diff_dim) : linearOperator<cuNDArray<T>>(), dim(diff_dim) {

        }
        virtual void mult_M(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate) override;
        virtual void mult_MH(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate) override;

        int dim;
    };
}