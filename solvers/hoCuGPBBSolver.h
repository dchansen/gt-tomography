#pragma once

#include "gpBbSolver.h"
#include "hoNDArray_elemwise.h"
#include "hoCuNDArray_blas.h"
#include "complext.h"

namespace Gadgetron{

template<class T> class hoCuGPBBSolver: public gpBbSolver<hoCuNDArray<T> >{
	typedef typename realType<T>::Type REAL;
public:
	hoCuGPBBSolver():gpBbSolver<hoCuNDArray<T> >(){

	}

	virtual ~hoCuGPBBSolver(){};

protected:
  virtual void solver_non_negativity_filter(hoCuNDArray<T> *x,hoCuNDArray<T> *g)
  {
    T* x_ptr = x->get_data_ptr();
    T* g_ptr = g->get_data_ptr();
    for (int i =0; i < x->get_number_of_elements(); i++){
      if ( real(x_ptr[i]) < REAL(0) && real(g_ptr[i]) > 0) g_ptr[i]=T(0);
    }

  }

  virtual void solver_reciprocal_clamp( hoCuNDArray<T>* x_arr, REAL threshold){
  	T* x = x_arr->get_data_ptr();
  	for (int i = 0;  i  < x_arr->get_number_of_elements(); i++){
  		if (real(x[i]) < threshold) x[i]= T(0);
  				  else x[i]= T(1)/x[i];
  	}
  }

  virtual void iteration_callback(hoCuNDArray<T>*,int i,REAL,REAL){
  	//std::cout << "Free memory: " << cudaDeviceManager::Instance()->getFreeMemory() << std::endl;
  };
};
}
