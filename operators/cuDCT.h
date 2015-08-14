#pragma once
#include "cuNDArray.h"

namespace Gadgetron {
template<class T,unsigned int S>	void dct2(cuNDArray<T>* image,int offset=0);
template<class T,unsigned int S>	void idct2(cuNDArray<T>* image, int offset=0);
template<class T,unsigned int S>	void dct(cuNDArray<T>* image,int dim, int offset=0);
template<class T,unsigned int S>	void idct(cuNDArray<T>* image, int dim, int offset=0);
}
