#pragma once
#include "linearOperator.h"
#include "cuPartialDerivativeOperator.h"

namespace Gadgetron {

template<class T> class cuDCTDerivativeOperator : public  linearOperator<cuNDArray<T>> {


public:

	cuDCTDerivativeOperator(size_t dim) : linearOperator<cuNDArray<T>>() {
		pD = cuPartialDerivativeOperator<float,4>(dim);
	}


	virtual void mult_M(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate) override {

		auto tmp = out;
		if (accumulate) tmp = new cuNDArray<T>(out->get_dimensions());
		pD.mult_M(in,tmp,false);
		dct<T,10>(tmp,3);
		if (accumulate){
			*out += *tmp;
			delete tmp;
		}
	}

	virtual void mult_MH(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate) override {

		auto tmp = *in;
		idct<T,10>(&tmp,3);
		pD.mult_MH(&tmp,out,accumulate);

	}


protected:
	cuPartialDerivativeOperator<float,4> pD;
};


}
