#pragma once

#include "cuNDArray_math.h"
#include "linearOperator.h"



namespace Gadgetron {
    template<class T,unsigned int D,unsigned int STENCIL_SIZE>
    class cuSmallConvOperator : public linearOperator<cuNDArray<T>> {

    public:
        cuSmallConvOperator(vector_td<T,STENCIL_SIZE> input_stencil, int conv_dim, int step_size = 1) : linearOperator<cuNDArray<T>>(), stencil(input_stencil),dim(conv_dim),stride(step_size) {
            for (int i = 0; i < STENCIL_SIZE; i++)
                reverse_stencil[i] = stencil[STENCIL_SIZE-i-1];

        }
        virtual void mult_M(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate) override;
        virtual void mult_MH(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate) override;

    private:
        vector_td<T,STENCIL_SIZE> stencil;
        vector_td<T,STENCIL_SIZE> reverse_stencil;
        int dim;
        int stride;
    };
}