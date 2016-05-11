#include "complext.h"
#include "solver_utils.h"
#include "setup_grid.h"
#include <algorithm>
#include <vector_td_utilities.h>
#define MAX_THREADS_PER_BLOCK 512

using namespace Gadgetron;

static inline
void setup_grid3D(intd3 dims, dim3 *blockDim, dim3 *gridDim) {
	int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
	//int maxGridDim = cudaDeviceManager::Instance()->max_griddim(cur_device);
	int maxBlockDim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
	int maxGridDim = 65535;

	// The default one-dimensional block dimension is...
	*blockDim = dim3(std::min(8,dims[0]),std::min(8,dims[1]),std::min(8,dims[2]));

	*gridDim = dim3((dims[0]+blockDim->x-1)/blockDim->x,
					(dims[1]+blockDim->y-1)/blockDim->y,
					(dims[2]+blockDim->z-1)/blockDim->z);



}

template<class T> __global__ static void huber_kernel(T* image, T* numerator, T* denominator , intd3 size,T alpha){

	const intd3 co = intd3(threadIdx.x+blockDim.x*blockIdx.x,threadIdx.y+blockDim.y*blockIdx.y,threadIdx.z+blockDim.z*blockIdx.z);

	T num =0;
	T den = 0;
	if (co < size){
		const int idx = co[0]+co[1]*size[0]+co[2]*size[0]*size[1];
		T val = image[idx];
		for (int z = co[2]-1;z <=co[2]+1; z++) {
			if (z < 0 || z >= size[2]) continue;

			for (int y = co[1]-1;y <=co[1]+1; y++) {
				if (y < 0 || y >= size[1]) continue;

				for (int x = co[0]-1;x <=co[0]+1; x++) {
					if (x < 0 || x >= size[0]) continue;
					T val2 = x+y*size[0]+z*size[0]*size[1];
					T adiff = abs(val-val2);
					if (adiff < alpha)
						num += adiff*adiff/(2*alpha);
					else
						num += adiff-alpha/2;

					den += 1.0f/max(adiff,alpha);
				}
			}
		}

		numerator[idx] = num;
		denominator[idx] = den;


	}

}

template<class T>  void Gadgetron::huber_norm(cuNDArray<T>* image,cuNDArray<T>* numerator,cuNDArray<T>* denominator,T alpha){
	dim3 dimBlock;
	dim3 dimGrid;

	intd3 size = {image->get_size(0),image->get_size(1),image->get_size(2)};

	setup_grid3D(size,&dimBlock,&dimGrid);

	huber_kernel<<<dimGrid,dimBlock>>>(image->get_data_ptr(),numerator->get_data_ptr(),denominator->get_data_ptr(),size,alpha);

}


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

template<class T> struct cuNDA_shrink_diff : public thrust::binary_function<T,T,T> {

	cuNDA_shrink_diff(typename realType<T>::Type gamma_): gamma(gamma_) {};
   __device__ T operator()(const T & x, const T & y) {
	   T z = x-y;
	   if (z > gamma) return x-gamma;
	   if (z < -gamma) return x+gamma;
	   return y;
   }
   typename realType<T>::Type gamma;

};



template<class T> void Gadgetron::shrink_diff(cuNDArray<T>* in_out, cuNDArray<T> * diff, typename realType<T>::Type gamma){
	thrust::transform(in_out->begin(),in_out->end(),diff->begin(),in_out->begin(),cuNDA_shrink_diff<T>(gamma));
}


template<class T> struct cuNDA_calcC : public thrust::binary_function<T,T,T> {
	__device__ T operator()(const T & l, const T & b){
		if (l > 0){
			T el = exp(-l);
			return 2*b*(1.0f-el-l*el)/(l*l);

		} else
			return b;
	}
};


template<class T> void Gadgetron::calcC(cuNDArray<T>* in_out, cuNDArray<T> * b){
	thrust::transform(in_out->begin(),in_out->end(),b->begin(),in_out->begin(),cuNDA_calcC<T>());
}





template<class T> struct cuNDA_exp_inv : public thrust::unary_function<T,T> {
	__device__ T operator()(const T & a){
			return exp(-a);
	}
};


template<class T> void Gadgetron::exp_inv_inplace(cuNDArray<T>* in_out){
	thrust::transform(in_out->begin(),in_out->end(),in_out->begin(),cuNDA_exp_inv<T>());
}

template void Gadgetron::huber_norm<float>(cuNDArray<float>*,cuNDArray<float>*,cuNDArray<float>*,float);
template void Gadgetron::exp_inv_inplace<float>(cuNDArray<float>*);
template void Gadgetron::calcC<float>(cuNDArray<float>*, cuNDArray<float>*);
template void Gadgetron::hard_shrink<float>(cuNDArray<float>*, float);
template void Gadgetron::shrink_diff<float>(cuNDArray<float>*,cuNDArray<float>*, float);
template void Gadgetron::solver_non_negativity_filter<float>(cuNDArray<float>*, cuNDArray<float>*);
template void Gadgetron::solver_non_negativity_filter<float>(hoCuNDArray<float>*, hoCuNDArray<float>*);
