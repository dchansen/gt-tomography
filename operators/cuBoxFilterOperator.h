#pragma once
#include "linearOperator.h"
#include "cuNDArray.h"
#include "cuNDArray_math.h"
namespace Gadgetron{


template<class T, unsigned int D> class cuBoxFilterOperator : public linearOperator<cuNDArray<T> > {
	typedef typename realType<T>::Type REAL;
public:
	void mult_M( cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate = false);
	void mult_MH( cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate = false);



};

}
