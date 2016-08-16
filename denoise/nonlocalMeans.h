#pragma once
#include <cuNDArray.h>
namespace Gadgetron{
    void nonlocal_means( cuNDArray<float> *input, cuNDArray<float> *output , float Noise);
    void nonlocal_meansPoisson( cuNDArray<float> *input, cuNDArray<float> *output , float Noise);
    void nonlocal_means_block( cuNDArray<float> *input, cuNDArray<float> *output , float Noise);
    void nonlocal_means_ref( cuNDArray<float> *input,cuNDArray<float>* output, cuNDArray<float> *ref , float Noise);
    void nonlocal_means2D( cuNDArray<float> *input, cuNDArray<float> *output , float Noise);
    void nonlocal_means2DPoisson( cuNDArray<float> *input, cuNDArray<float> *output , float Noise);

};