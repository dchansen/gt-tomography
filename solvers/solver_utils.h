#pragma once
#include "complext.h"
#include "hoCuNDArray.h"
#include "cuNDArray.h"

namespace Gadgetron{

template<class T> void solver_non_negativity_filter(cuNDArray<T>* x , cuNDArray<T>* g);

template<class T> void solver_non_negativity_filter(hoCuNDArray<T>* x , hoCuNDArray<T>* g);


template<class T> void shrink_diff(cuNDArray<T>* in_out, cuNDArray<T> * diff, typename realType<T>::Type gamma);

template<class T> void hard_shrink(cuNDArray<T>* in_out, typename realType<T>::Type gamma);
template<class T> void save_nd_array(cuNDArray<T>* array, std::string s){
	write_nd_array(array->to_host().get(),s.c_str());
}

template<class T> void save_nd_array(hoNDArray<T>* array, std::string s){
	write_nd_array(array,s.c_str());
}

	template <class T> void huber_norm(cuNDArray<T>*,cuNDArray<T>*,cuNDArray<T>*,T);

	template< class T> void calcC(cuNDArray<T>* in_out,cuNDArray<T>* b);
	template< class T> void exp_inv_inplace(cuNDArray<T>* in_out);
}
