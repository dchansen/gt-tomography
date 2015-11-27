#pragma once

#include "hoNDArray.h"
#include "cuNDArray.h"
#include "cgPreconditioner.h"

#include "linearOperator.h"

#include "cuNDFFT.h"
#include "hoNDFFT.h"
#include "hoNDArray_math.h"
#include "cuNDArray_math.h"

namespace Gadgetron{



template<class ARRAY> struct FFTInstance{};

template<class T> struct FFTInstance<cuNDArray<T> >{
	static cuNDFFT<T>* instance(){
		return cuNDFFT<T>::instance();
	}
};
template<class T> struct FFTInstance<hoCuNDArray<T> >{
	static hoNDFFT<T>* instance(){
		return hoNDFFT<T>::instance();
	}
};



template<template<class> class ARRAY, class T> class circulantPreconditioner: public cgPreconditioner<ARRAY<T> > {

public:

	circulantPreconditioner(boost::shared_ptr< linearOperator<ARRAY<T> > > op) : cgPreconditioner<ARRAY<T> >(), op_(op) {

		kernel_ = calcKernel();

	}


	virtual ~circulantPreconditioner(){};
	virtual void apply(ARRAY<T> * in, ARRAY<T> * out){
		std::cout <<"Preconditionaaa-buruuu!" << std::endl;
		ARRAY<T> tmp(*in);
		save_nd_array(&tmp,"in.real");
		boost::shared_ptr<ARRAY<complext<T> > > cplx = real_to_complex<complext<T> >(in);
		FFTInstance<ARRAY<T> >::instance()->fft(cplx.get());
		std::cout <<"FFT-buruu!" << std::endl;
 		*cplx *= *kernel_;
		FFTInstance<ARRAY<T> >::instance()->ifft(cplx.get());

		*out = *real(cplx.get());
		save_nd_array(out,"kernel.real");

		std::cout <<"Done-buruu!" << std::endl;
	}

protected:


	boost::shared_ptr< linearOperator<ARRAY<T> > > op_;
	boost::shared_ptr<ARRAY<complext<T> > > kernel_;

	boost::shared_ptr<ARRAY<complext<T> > >  calcKernel(){
		std::vector<size_t> dims = *op_->get_domain_dimensions();
		std::vector<size_t> central_element;
		for (size_t i = 0; i < dims.size(); i++){
			central_element.push_back(dims[i]/2);
			std::cout << " " <<central_element[i];
		}

		std::cout << std::endl;


		std::cout << "CAlculating the kerrrrrnal!" << std::endl;
		ARRAY<T> real_kernel(dims);
		{
			hoNDArray<T> ho_one(dims);
			clear(&ho_one);

			size_t central_id = 0;
			size_t stride = 1;
			for (size_t i = 0; i < dims.size(); i++){
				central_id += stride*central_element[i];
				stride *= dims[i];
			}
			ho_one.get_data_ptr()[central_id] = T(1);


			ARRAY<T> one(&ho_one);
			std::cout << "Applying the operAaaator" << std::endl;
			op_->mult_MH_M(&one,&real_kernel);
		}


		boost::shared_ptr<ARRAY<complext<T> > > kernel = real_to_complex<complext<T> >(&real_kernel);

		std::cout << "Doing FFT" << std::endl;
		FFTInstance<ARRAY<T> >::instance()->fft(kernel.get());
		reciprocal_inplace(kernel.get());
		sqrt_inplace(kernel.get());
		return kernel;
	}
};
}
