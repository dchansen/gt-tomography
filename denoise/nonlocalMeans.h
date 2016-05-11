#pragma once
#include <cuNDArray.h>
namespace Gadgetron{
    void nonlocal_means( cuNDArray<float> *input, cuNDArray<float> *output , float Noise);
    void nonlocal_means2D( cuNDArray<float> *input, cuNDArray<float> *output , float Noise);

};