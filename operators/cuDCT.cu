#include "cuDCT.h"
#include "math_constants.h"

using namespace Gadgetron;

template<class T, unsigned int S> __inline__ __device__ void dct_row(T * stencil, int k, int col){
	T result = 0;
#pragma unroll
	for (int i = 0; i < S; i++){
		result += 2.0f*stencil[i+col*S]*__cosf(CUDART_PI_F/S*(i+0.5f)*k);
	}
	if (k==0)
		result *= 1.0f/sqrt(2.0f);
		//result = 0;

	__syncthreads();
	stencil[k+col*S] = result*sqrt(1/(2.0f*S));
	__syncthreads();
}


template<class T, unsigned int S> __inline__ __device__ void idct_row(T * stencil, int k, int col){
	T result = stencil[col*S]/sqrt(T(S));
	const T factor = sqrt(2.0f/S);
#pragma unroll
	for (int i = 1; i < S; i++){
		result += factor*stencil[i+col*S]*__cosf(CUDART_PI_F/S*(k+0.5f)*i);
	}
	__syncthreads();
	stencil[k+col*S] = result;
	__syncthreads();
}

template<class T, unsigned int S> __inline__ __device__ void dct_col(T * stencil, int k, int col){
	T result = 0;
#pragma unroll
	for (int i = 0; i < S; i++){
		result += 2.0f*stencil[i*S+col]*__cosf(CUDART_PI_F/S*(i+0.5f)*k);
	}
	if (k==0)
		result *= 1.0f/sqrt(2.0f);
		//result =0;

	__syncthreads();
	stencil[k*S+col] = result*sqrt(1.0f/(2.0f*S));
	__syncthreads();
}


template<class T, unsigned int S> __inline__ __device__ void idct_col(T * stencil, int k, int col){
	T result = stencil[col]/sqrt(T(S));
	const T factor = sqrt(2.0f/S);
#pragma unroll
	for (int i = 1; i < S; i++){
		result += factor*stencil[i*S+col]*__cosf(CUDART_PI_F/S*(k+0.5f)*i);
	}
	__syncthreads();
	stencil[k*S+col] = result;
	__syncthreads();
}



template<class T, unsigned int S>  __global__ void dct2_kernel(T* __restrict__ image, int dimX, int dimY,int offset){


	__shared__ T stencil[S*S];

	const int x = threadIdx.x;
	const int y = threadIdx.y;

	const int imagex = (x+blockIdx.x*S+offset)%dimX;
	const int imagey = (y+blockIdx.y*S+offset)%dimY;

	const int idx = imagex+imagey*dimX+blockIdx.z*dimX*dimY;
	stencil[x+y*S] = image[idx];
	__syncthreads();
	dct_row<T,S>(stencil,x,y);
	dct_col<T,S>(stencil,y,x);
	image[idx] = stencil[x+y*S];




}


template<class T, unsigned int S> __global__ void dct_kernel(T* __restrict__ image, int stride, int dim, int offset){

	__shared__ T stencil[S]; //Do we really get something from using shared memory when it's just 1D?
	const int x = threadIdx.x;

	const int idx = ((x+blockIdx.y*S+offset)%dim)*stride+blockIdx.x+blockIdx.z*dim*stride;
	stencil[x] = image[idx];
	__syncthreads();
	dct_row<T,S>(stencil,x,0);
	image[idx] = stencil[x];

}
template<class T, unsigned int S> __global__ void idct_kernel(T* __restrict__ image, int stride, int dim, int offset){

	__shared__ T stencil[S]; //Do we really get something from using shared memory when it's just 1D?
	const int x = threadIdx.x;

	const int idx = ((x+blockIdx.y*S+offset)%dim)*stride+blockIdx.x+blockIdx.z*dim*stride;
	stencil[x] = image[idx];
	__syncthreads();
	idct_row<T,S>(stencil,x,0);
	image[idx] = stencil[x];

}

template<class T, unsigned int S>  __global__ void idct2_kernel(T* __restrict__ image, int dimX, int dimY, int offset){


	__shared__ T stencil[S*S];

	const int x = threadIdx.x;
	const int y = threadIdx.y;

	const int imagex = (x+blockIdx.x*S+offset)%dimX;
	const int imagey = (y+blockIdx.y*S+offset)%dimY;

	const int idx = imagex+imagey*dimX+blockIdx.z*dimX*dimY;
	stencil[x+y*S] = image[idx];
	idct_row<T,S>(stencil,x,y);
	idct_col<T,S>(stencil,y,x);
	image[idx] = stencil[x+y*S];

}


template<class T, unsigned int S> void Gadgetron::dct2(cuNDArray<T>* in,int offset){

	if ((in->get_size(0)%S != 0) || (in->get_size(1)%S !=0))
		throw std::runtime_error("Image size must be multiple of patchsize");

	dim3 grid,block;
	block.x = S;
	block.y = S;



	grid.x = in->get_size(0)/S;
	grid.y = in->get_size(1)/S;


	grid.z = in->get_number_of_elements()/(in->get_size(0)*in->get_size(1));
	dct2_kernel<T,S><<<grid,block>>>(in->get_data_ptr(),in->get_size(0),in->get_size(1),offset);
	CHECK_FOR_CUDA_ERROR();


}



template<class T, unsigned int S> void Gadgetron::idct2(cuNDArray<T>* in, int offset){

	if ((in->get_size(0)%S != 0) || (in->get_size(1)%S !=0))
		throw std::runtime_error("Image size must be multiple of patchsize");

	dim3 grid,block;
	block.x = S;
	block.y = S;



	grid.x = in->get_size(0)/S;
	grid.y = in->get_size(1)/S;

	grid.z = in->get_number_of_elements()/(in->get_size(0)*in->get_size(1));

	idct2_kernel<T,S><<<grid,block>>>(in->get_data_ptr(),in->get_size(0),in->get_size(1),offset);


}


template<class T, unsigned int S> void Gadgetron::dct(cuNDArray<T>* in, int dim, int offset){
	if (in->get_size(dim)%S != 0) throw std::runtime_error("Image dimension must be multiple of patchsize");

	dim3 grid,block;
	block.x = S;


	size_t stride = 1;
	for (int i =0; i < dim; i++) stride *= in->get_size(i);

	//grid.x = 1;
	grid.x = stride;
	grid.y = in->get_size(dim)/S;
	grid.z = in->get_number_of_elements()/(block.x*block.y*grid.x*grid.y);


	cudaDeviceProp deviceProp;
	int device;
	cudaGetDevice(&device);
	if( cudaGetDeviceProperties( &deviceProp, device ) != cudaSuccess) {
		throw cuda_error("Error: unable to determine device properties.");
	}

	unsigned int repetitions = 1;

	while (grid.z > deviceProp.maxGridSize[2]){
		grid.z /= 2;
		repetitions *= 2;
	}

	size_t chunk_size = block.x*grid.x*grid.y*grid.z;

	for (int i = 0; i < repetitions; i++)
		dct_kernel<T,S><<<grid,block>>>(in->get_data_ptr()+i*chunk_size,int(stride), in->get_size(dim),offset );

	CHECK_FOR_CUDA_ERROR();

}
template<class T, unsigned int S> void Gadgetron::idct(cuNDArray<T>* in, int dim, int offset){
	if (in->get_size(dim)%S != 0) throw std::runtime_error("Image dimension must be multiple of patchsize");

	dim3 grid,block;
	block.x = S;

	size_t stride = 1;
	for (int i =0; i < dim; i++) stride *= in->get_size(i);

	grid.x = stride;
	grid.y = in->get_size(dim)/S;
	grid.z = in->get_number_of_elements()/(block.x*grid.x*grid.y);
	cudaDeviceProp deviceProp;
	int device;
	cudaGetDevice(&device);
	if( cudaGetDeviceProperties( &deviceProp, device ) != cudaSuccess) {
		throw cuda_error("Error: unable to determine device properties.");
	}

	unsigned int repetitions = 1;

	while (grid.z > deviceProp.maxGridSize[2]){
		grid.z /= 2;
		repetitions *= 2;
	}

	size_t chunk_size = block.x*grid.x*grid.y*grid.z;


	for (int i = 0; i < repetitions; i++)
	idct_kernel<T,S><<<grid,block>>>(in->get_data_ptr()+i*chunk_size,stride, in->get_size(dim),offset );

	CHECK_FOR_CUDA_ERROR();

}
template void Gadgetron::dct2<float,16>(cuNDArray<float>* in,int);
template void Gadgetron::idct2<float,16>(cuNDArray<float>* in,int);
template void Gadgetron::dct<float,16>(cuNDArray<float>* ,int, int);
template void Gadgetron::idct<float,16>(cuNDArray<float>* ,int, int);

template void Gadgetron::dct2<float,32>(cuNDArray<float>* in,int);
template void Gadgetron::idct2<float,32>(cuNDArray<float>* in,int);
template void Gadgetron::dct<float,32>(cuNDArray<float>* ,int, int);
template void Gadgetron::idct<float,32>(cuNDArray<float>* ,int, int);
template void Gadgetron::dct<float,10>(cuNDArray<float>* ,int, int);
template void Gadgetron::idct<float,10>(cuNDArray<float>* ,int, int);
template void Gadgetron::dct2<float,8>(cuNDArray<float>* in,int);
template void Gadgetron::idct2<float,8>(cuNDArray<float>* in,int);
template void Gadgetron::dct<float,8>(cuNDArray<float>* ,int, int);
template void Gadgetron::idct<float,8>(cuNDArray<float>* ,int, int);

template void Gadgetron::dct<float,256>(cuNDArray<float>* ,int, int);
template void Gadgetron::idct<float,256>(cuNDArray<float>* ,int, int);

template void Gadgetron::dct<float,192>(cuNDArray<float>* ,int, int);
template void Gadgetron::idct<float,192>(cuNDArray<float>* ,int, int);


template void Gadgetron::dct2<float,4>(cuNDArray<float>* in,int);
template void Gadgetron::idct2<float,4>(cuNDArray<float>* in,int);
template void Gadgetron::dct<float,4>(cuNDArray<float>* ,int, int);
template void Gadgetron::idct<float,4>(cuNDArray<float>* ,int, int);


template void Gadgetron::dct2<float,2>(cuNDArray<float>* in,int);
template void Gadgetron::idct2<float,2>(cuNDArray<float>* in,int);
template void Gadgetron::dct<float,2>(cuNDArray<float>* ,int, int);
template void Gadgetron::idct<float,2>(cuNDArray<float>* ,int, int);