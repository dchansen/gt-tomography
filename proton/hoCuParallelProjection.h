#pragma once

#include "hoCuNDArray.h"
#include "cuNDArray.h"
#include "vector_td.h"

namespace Gadgetron {
void parallel_backprojection(hoCuNDArray<float>* projections, cuNDArray<float>* image, float  angle, floatd3 image_dims, floatd3 projection_dims);
void parallel_backprojection(cuNDArray<float>* projections, cuNDArray<float>* image, float  angle, floatd3 image_dims, floatd3 projection_dims);
void interpolate_missing( cuNDArray<float>* image );
}
