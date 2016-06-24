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

	void bilateral_vfield(cuNDArray<float>* vfield1, cuNDArray<float>* image, float sigma_spatial,float sigma_int, float sigma_diff);
	cuNDArray<float> Jacobian(cuNDArray<float>* vfield);


template<class T, unsigned int D> class cuDemonsSolver {


public:
	cuDemonsSolver() : alpha(1.0),beta(1e-6),sigmaDiff(1.0),sigmaFluid(1.0),compositive(false),epsilonNGF(0),exponential(false) {};
	virtual ~cuDemonsSolver(){};

	boost::shared_ptr<cuNDArray<T>> registration( cuNDArray<T> *fixed_image, cuNDArray<T> *moving_image);

	void set_iterations(int i){
		iterations = i;
	}
	void set_sigmaDiff(T _sigma){
		sigmaDiff = _sigma;

			}

	void set_sigmaFluid(T _sigma){
		sigmaFluid = _sigma;

	}
	void set_alpha(T _alpha){
		alpha = _alpha;
	}

	void set_compositive(bool option){
		compositive = option;
	}

	void set_exponential(bool option){
		exponential = option;
	}

	void use_normalized_gradient_field(T epsilon){
		epsilonNGF = epsilon;
	}


	void set_sigmaVDiff(T sigma){
		sigmaVDiff = sigma;
	}

	void set_sigmaInt(T sigma){
		sigmaInt = sigma;
	}


protected:
	boost::shared_ptr<cuNDArray<T> > demonicStep(cuNDArray<T>*,cuNDArray<T>*);


	T sigmaDiff, sigmaFluid,sigmaInt,sigmaVDiff,alpha,beta,epsilonNGF;
	bool compositive, exponential;
	int iterations;
};


}
