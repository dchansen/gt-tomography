#include "complext.h"
#include "solver_utils.h"
#include "setup_grid.h"
#include <algorithm>
#define MAX_THREADS_PER_BLOCK 512

using namespace Gadgetron;
template <class T> __global__ static void filter_kernel(T* x, T* g, int elements){
	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < elements){
		if ( x[idx] <= T(0) && g[idx] > 0) g[idx]=T(0);
	}
}

template <class REAL> __global__ static void filter_kernel(complext<REAL>* x, complext<REAL>* g, int elements){
	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < elements){
		if ( real(x[idx]) <= REAL(0) && real(g[idx]) > 0) g[idx].vec[0] = REAL(0);
		g[idx].vec[1]=REAL(0);
	}
}

template <class T> void Gadgetron::solver_non_negativity_filter(cuNDArray<T>* x , cuNDArray<T>* g)
{
	int elements = g->get_number_of_elements();

	dim3 dimBlock;
	dim3 dimGrid;
	setup_grid(x->get_number_of_elements(),&dimBlock, &dimGrid);


	CHECK_FOR_CUDA_ERROR();

	filter_kernel<typename realType<T>::Type><<<dimGrid,dimBlock>>>(x->get_data_ptr(),g->get_data_ptr(),elements);

}

template <class T> void Gadgetron::solver_non_negativity_filter(hoCuNDArray<T>* x , hoCuNDArray<T>* g)
{
	typedef typename realType<T>::Type REAL;
	T* xptr = x->get_data_ptr();
	T* gptr = g->get_data_ptr();
	for (int i = 0; i != x->get_number_of_elements(); ++i)
		if ( real(xptr[i]) <= REAL(0) && real(gptr[i]) > 0) gptr[i]=T(0);
}


template<class T> struct cuNDA_hardshrink : public thrust::unary_function<T,T> {

	cuNDA_hardshrink(typename realType<T>::Type gamma_): gamma(gamma_) {};
   __device__ T operator()(const T & x) {
	   return abs(x) < gamma ? T(0) : x;
   }
   typename realType<T>::Type gamma;

};

template<class T> void Gadgetron::hard_shrink(cuNDArray<T>* in_out, typename realType<T>::Type gamma){
	thrust::transform(in_out->begin(),in_out->end(), in_out->begin(),cuNDA_hardshrink<T>(gamma));
		
}



template void Gadgetron::hard_shrink<float>(cuNDArray<float>*, float);
template void Gadgetron::solver_non_negativity_filter<float>(cuNDArray<float>*, cuNDArray<float>*);
template void Gadgetron::solver_non_negativity_filter<float>(hoCuNDArray<float>*, hoCuNDArray<float>*);
