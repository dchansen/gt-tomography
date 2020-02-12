#pragma once

#include <thrust/device_vector.h>
#include "cuNDArray.h"
namespace Gadgetron {


template<class T> void EdgeWavelet(cuNDArray<T>* in, cuNDArray<T>* out, thrust::device_vector<typename realType<T>::Type>* kernel, int stepsize,int dim,typename realType<T>::Type sigma, bool accumulate);


}
