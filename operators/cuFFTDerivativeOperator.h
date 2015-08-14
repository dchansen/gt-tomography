
#pragma once
#include "linearOperator.h"
#include "cuPartialDerivativeOperator.h"
#include "cuRFFTOperator.h"

namespace Gadgetron {

template<class T> class cuDCTDerivativeOperator : public  linearOperator<cuNDArray<T>> {


public:

	cuDCTDerivativeOperator(size_t dim) : linearOperator<cuNDArray<T>>(), fftOp(cuRFFTOperator<float>(3)) {
	}


	virtual void mult_M(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate) override {

		cuNDArray<T> tmp(in->get_dimensions());
		pD.mult_M(in,&tmp,false);

		fftOp.mult_M(&tmp,out,accumulate);
		/*dct<T,10>(tmp,3);
		if (accumulate){
			*out += *tmp;
			delete tmp;
		}*/
	}

	virtual void mult_MH(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate) override {

		auto tmp = *out;
		//idct<T,10>(&tmp,3);
		fftOp.mult_MH(in,&tmp,false);
		pD.mult_MH(&tmp,out,accumulate);

	}


	virtual void set_domain_dimensions(std::vector<size_t>* dims) override {
		fftOp.set_domain_dimensions(dims);
	}

	boost::shared_ptr<std::vector<size_t> > get_codomain_dimensions() override {
		return fftOp.get_codomain_dimensions();
	}

protected:
	cuRFFTOperator<float> fftOp;
};


}
