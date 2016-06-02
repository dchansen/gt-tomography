#pragma once
#include "cuNDArray_math.h"
#include "cuNDArray_utils.h"
#include "multiresRegistrationSolver.h"
#include "cuGaussianFilterOperator.h"
#include "vector_td.h"
#include "cuNDArray.h"

namespace Gadgetron{

	cuNDArray<float> deform_image(cuNDArray<float>* image, cuNDArray<float>* vector_field);
	cuNDArray<float> deform_image(cudaTextureObject_t  texObj,std::vector<size_t> dimensions, cuNDArray<float>* vector_field);
	void deform_vfield(cuNDArray<float>* vfield1, cuNDArray<float>* vector_field);
template<class T, unsigned int D> class cuDemonsSolver : public multiresRegistrationSolver<cuNDArray<T>, D>{


public:
	cuDemonsSolver() : alpha(1.0),beta(1e-6),sigmaDiff(1.0),sigmaFluid(1.0) {};
	virtual ~cuDemonsSolver(){};

	virtual void compute( cuNDArray<T> *fixed_image, cuNDArray<T> *moving_image, cuNDArray<T> *stencil_image, boost::shared_ptr<cuNDArray<T> > &result );
	void set_sigmaDiff(T _sigma){
		sigmaDiff = _sigma;

			}

	void set_sigmaFluid(T _sigma){
		sigmaFluid = _sigma;

	}
	void set_alpha(T _alpha){
		alpha = _alpha;
	}



protected:
	boost::shared_ptr<cuNDArray<T> > demonicStep(cuNDArray<T>*,cuNDArray<T>*);


	T sigmaDiff, sigmaFluid,alpha,beta;
};


}
