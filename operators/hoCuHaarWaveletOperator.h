#pragma once

#include "hoCuNDArray.h"
#include "cuNDArray.h"
#include "linearOperator.h"
#include "cuHaarWaveletOperator.h"

namespace Gadgetron{

template<class T, unsigned int D> class hoCuHaarWaveletOperator : public linearOperator<hoCuNDArray<T> >{

public:
	hoCuHaarWaveletOperator() : linearOperator<hoCuNDArray<T> >(){};

	virtual ~hoCuHaarWaveletOperator(){};
	virtual void mult_M(hoCuNDArray<T>* in ,hoCuNDArray<T>* out,bool accumulate ){
		cuNDArray<T> cuIn(*in);
		cuNDArray<T> cuOut(*out);

		cuHaar.mult_M(&cuIn,&cuOut,accumulate);

		cudaMemcpy(out->get_data_ptr(),cuOut.get_data_ptr(),cuOut.get_number_of_elements()*sizeof(T),cudaMemcpyDeviceToHost);

	}
	virtual void mult_MH(hoCuNDArray<T>* in,hoCuNDArray<T>* out ,bool accumulate){
		cuNDArray<T> cuIn(*in);
		cuNDArray<T> cuOut(*out);

		cuHaar.mult_MH(&cuIn,&cuOut,accumulate);

		cudaMemcpy(out->get_data_ptr(),cuOut.get_data_ptr(),cuOut.get_number_of_elements()*sizeof(T),cudaMemcpyDeviceToHost);

	}
	virtual void mult_MH_M(hoCuNDArray<T>* in ,hoCuNDArray<T>* out,bool accumulate ){
		if (accumulate){
			*out += *in;
		} else {
			*out = *in;
		}
	}
	virtual boost::shared_ptr< linearOperator< hoCuNDArray<T>  > >  clone(){
				return linearOperator< hoCuNDArray<T> >::clone(this);
			}
	virtual void set_domain_dimensions(std::vector<unsigned int>* dims){
		cuHaar.set_domain_dimensions(dims);
	}

	virtual boost::shared_ptr< std::vector<unsigned int> > get_domain_dimensions(){
		return cuHaar.get_domain_dimensions();
	}

	virtual boost::shared_ptr< std::vector<unsigned int> > get_codomain_dimensions(){
			return cuHaar.get_codomain_dimensions();
		}

protected:
	cuHaarWaveletOperator<T,D> cuHaar;
};

}
