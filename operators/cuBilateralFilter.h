#pragma once
#include "cuNDArray.h"

namespace Gadgetron {
    void bilateral_filter(cuNDArray<float>* image, cuNDArray<float>* ref_image,float  sigma_spatial,float sigma_int);
    void bilateral_filter_unnormalized(cuNDArray<float>* image, cuNDArray<float>* ref_image,float  sigma_spatial,float sigma_int);
}
