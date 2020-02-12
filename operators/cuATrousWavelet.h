#pragma once

#include <thrust/device_vector.h>
#include "cuNDArray.h"
namespace Gadgetron {


template<class T> void aTrousWavelet(cuNDArray<T>* in, cuNDArray<T>* out, thrust::device_vector<typename realType<T>::Type>* kernel, int stepsize,int dim, bool accumulate);


}
