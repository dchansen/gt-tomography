#include "cuEdgeWavelet.h"
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

__device__ int float_modulo(const int &x, const int &n )

{

float tmp1 ;

int k, r ;



float one_over_n = 1.0f/float(n);



tmp1 = __int2float_rz( x ) ;

tmp1 = tmp1 * one_over_n ;

k = __float2int_rz( tmp1 ) ;

r = x - n*k ;

return r ;

}
template<class T>  __global__ void edgeAtrous_kernel(const T* __restrict__ image, T* __restrict__ out,  int stepsize, int stride, int dim_length, typename realType<T>::Type * kernel, int kernel_length, typename realType<T>::Type sigma2, int tot_elements){

  const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < tot_elements){
  	T result = 0;

	const T val = image[idx];
	typename realType<T>::Type norm = 0;


  	//const int dim_pos = float_modulo((idx/stride),dim_length);
  	const int dim_pos = (idx/stride)%dim_length;
  	const int offset = idx-dim_pos*stride;
  	for (int i = -kernel_length/2; i <= kernel_length/2; i++){
  		//int pos = (dim_pos+i*stepsize+dim_length)%dim_length;
  		int pos = float_modulo(dim_pos+i*stepsize+dim_length,dim_length);
		T img_val = image[pos*stride+offset];
		T k = kernel[i+kernel_length/2];
		T weight = __expf(-(val-img_val)*(val-img_val)/sigma2)*k;
		norm += weight;
  		result += img_val*weight;
  	}
  	atomicAdd(out+idx,result/norm);
  }
}





template<class T> void Gadgetron::EdgeWavelet(cuNDArray<T>* in, cuNDArray<T>* out, thrust::device_vector<typename realType<T>::Type>* kernel, int stepsize,int dim, typename realType<T>::Type sigma, bool accumulate){

	dim3 dimGrid,dimBlock;
	setup_grid(in->get_number_of_elements(),&dimBlock,&dimGrid,1);

	if (dim >= in->get_number_of_dimensions())
		throw std::runtime_error("EdgeWavelet: input array has insufficient number of dimensions");

	int max_grid = cudaDeviceManager::Instance()->max_griddim();


	int stride = 1;
	for (int i = 0; i < dim; i++) stride *= in->get_size(i);

	if (!accumulate)
		clear(out);
	edgeAtrous_kernel<<<dimGrid,dimBlock>>>(in->get_data_ptr(), out->get_data_ptr(),stepsize,stride,in->get_size(dim),thrust::raw_pointer_cast(kernel->data()),kernel->size(),sigma*sigma,in->get_number_of_elements());

}


template void Gadgetron::EdgeWavelet<float>(cuNDArray<float>*, cuNDArray<float>*, thrust::device_vector<float>*, int, int, float,bool);
