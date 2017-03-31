#pragma once
#include <cuNDArray.h>
#include <vector_td.h>
namespace Gadgetron{


    float mssim(cuNDArray<float>* image,cuNDArray<float>* reference,floatd3 sigma, int scales=3);

}