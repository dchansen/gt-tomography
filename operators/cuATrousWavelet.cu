#include "cuATrousWavelet.h"
#include "complext.h"
//#include "setup_grid.h"
#include "cuNDArray_math.h"
#include "cudaDeviceManager.h"
using namespace Gadgetron;

static inline
  void setup_grid( unsigned int number_of_elements, dim3 *blockDim, dim3* gridDim, unsigned int num_batches = 1 )
  {
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
    //int maxGridDim = cudaDeviceManager::Instance()->max_griddim(cur_device);
    int maxBlockDim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    int maxGridDim = 65535;

    // The default one-dimensional block dimension is...
    *blockDim = dim3(256);
    *gridDim = dim3((number_of_elements+blockDim->x-1)/blockDim->x, num_batches);

    // Extend block/grid dimensions if we exceeded the maximum grid dimension
    if( gridDim->x > maxGridDim){
      blockDim->x = maxBlockDim;
      gridDim->x = (number_of_elements+blockDim->x-1)/blockDim->x;
    }

    if( gridDim->x > maxGridDim ){
      gridDim->x = (unsigned int)std::floor(std::sqrt(float(number_of_elements)/float(blockDim->x)));
      unsigned int num_elements_1d = blockDim->x*gridDim->x;
      gridDim->y *= ((number_of_elements+num_elements_1d-1)/num_elements_1d);
    }

    if( gridDim->x > maxGridDim || gridDim->y > maxGridDim){
      // If this ever becomes an issue, there is an additional grid dimension to explore for compute models >= 2.0.
      throw cuda_error("setup_grid(): too many elements requested.");
    }
  }
static __device__ void atomicAdd(float_complext * ptr, float_complext val){

	atomicAdd((float*) ptr, real(val));
	atomicAdd(((float*)ptr)+1,imag(val));
}
template<class T>  __global__ void aTrous_kernel(const T* __restrict__ image, T* __restrict__ out,  int stepsize, int stride, int dim_length, typename realType<T>::Type * kernel, int kernel_length, int tot_elements){

  const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < tot_elements){
  	T result = 0;

  	const int dim_pos = (idx/stride)%dim_length;
  	const int offset = idx-dim_pos*stride;
  	for (int i = -kernel_length/2; i <= kernel_length/2; i++){
  		int pos = (dim_pos+i*stepsize+dim_length)%dim_length;
  		result += image[pos*stride+offset]*kernel[i+kernel_length/2];
  	}
  	atomicAdd(out+idx,result);
  }
}



template<class T> void Gadgetron::aTrousWavelet(cuNDArray<T>* in, cuNDArray<T>* out, thrust::device_vector<typename realType<T>::Type>* kernel, int stepsize,int dim, bool accumulate){

	dim3 dimGrid,dimBlock;
	setup_grid(in->get_number_of_elements(),&dimBlock,&dimGrid,1);

	if (dim >= in->get_number_of_dimensions())
		throw std::runtime_error("aTrousWavelet: input array has insufficient number of dimensions");

	int max_grid = cudaDeviceManager::Instance()->max_griddim();


	int stride = 1;
	for (int i = 0; i < dim; i++) stride *= in->get_size(i);

	if (!accumulate)
		clear(out);
	aTrous_kernel<<<dimGrid,dimBlock>>>(in->get_data_ptr(), out->get_data_ptr(),stepsize,stride,in->get_size(dim),thrust::raw_pointer_cast(kernel->data()),kernel->size(),in->get_number_of_elements());

}


template void Gadgetron::aTrousWavelet<float>(cuNDArray<float>*, cuNDArray<float>*, thrust::device_vector<float>*, int, int, bool);
template void Gadgetron::aTrousWavelet<float_complext>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, thrust::device_vector<float>*, int, int, bool);
