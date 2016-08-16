/*
 * cuframeletOperator.cu
 *
 *  Created on: Jun 6, 2013
 *      Author: u051747
 */

#include "cuFrameletOperator.h"
#include "vector_td_utilities.h"

using namespace Gadgetron;

// Template Power function
template<unsigned int i, unsigned int j>
struct Pow
{
  enum { Value = i*Pow<i,j-1>::Value};
};

template <unsigned int i>
struct Pow<i,1>
{
  enum { Value = i};
};


inline void setup_grid( unsigned int number_of_elements, dim3 *blockDim, dim3* gridDim, unsigned int num_batches = 1 )
{
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
    int maxGridDim = cudaDeviceManager::Instance()->max_griddim(cur_device);
    int maxBlockDim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    // For small arrays we keep the block dimension fairly small
    *blockDim = dim3(256);
    *gridDim = dim3((number_of_elements+blockDim->x-1)/blockDim->x, num_batches);

    // Extend block/grid dimensions for large arrays
    if( gridDim->x > maxGridDim){
        blockDim->x = maxBlockDim;
        gridDim->x = (number_of_elements+blockDim->x-1)/blockDim->x;
    }

    if( gridDim->x > maxGridDim ){
        gridDim->x = ((unsigned int)std::sqrt((float)number_of_elements)+blockDim->x-1)/blockDim->x;
        gridDim->y *= ((number_of_elements+blockDim->x*gridDim->x-1)/(blockDim->x*gridDim->x));
    }

    if( gridDim->x >maxGridDim || gridDim->y >maxGridDim){
        BOOST_THROW_EXCEPTION(cuda_error("Grid dimension larger than supported by device"));
    }
}
template<class T,unsigned int D, unsigned int N> __global__ void frameletKernel(T* in, T* out, vector_td<int,D> dims,int dir, vector_td<T,N> stencil){
	vector_td<T,N> elements;
	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	typename intd<D>::Type co = idx_to_co<D>(idx, dims);
	co[dir] = (co[dir]-N+dims[dir])%dims[dir];

	for (int i = 0; i < N; i++){
		elements[i] = in[co_to_idx<D>(co,dims)];
		co[dir] = (co[dir]+1+dims[dir])%dims[dir];
	}
	out[idx] = dot(elements,stencil);

}


template<class T,unsigned int D, unsigned int N> __global__ void iframeletKernel(T* in, T* out, vector_td<int,D> dims,int dir, vector_td<T,N> stencil){
	vector_td<T,N> elements;
	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	T val = in[idx];
	typename intd<D>::Type co = idx_to_co<D>(idx, dims);
	co[dir] = (co[dir]-N+dims[dir])%dims[dir];

	for (int i = 0; i < N; i++){
		in[co_to_idx<D>(co,dims)] += stencil[i]*val;
		co[dir] = (co[dir]+1+dims[dir])%dims[dir];
	}

}



template<class T, unsigned int D> void cuFrameletOperator<T,D>::mult_M(cuNDArray<T> *in, cuNDArray<T> *out, bool accumulate){

	if (! in->dimensions_equal(this->get_domain_dimensions().get()))
		throw std::runtime_error("cuframeletOperator::mult_M: size of input array does not match operator domain size.");
	if (! out->dimensions_equal(this->get_codomain_dimensions().get()))
			throw std::runtime_error("cuframeletOperator::mult_M: size of output array does not match operator codomain size.");
	boost::shared_ptr<cuNDArray<T> > tmp_in(new cuNDArray<T>(this->get_codomain_dimensions()));

	vector_td<T,3> h0 = vector_td<T,3>(1,2,1)/T(4);
	vector_td<T,3> h1 = vector_td<T,3>(1,0,1)*T(std::sqrt(2.0)/4.0);
	vector_td<T,3> h2 = vector_td<T,3>(-1,2,-1)/T(4);

	std::vector<vector_td<T,3> > stencils;
	stencils.push_back(h0);
	stencils.push_back(h1);
	stencils.push_back(h2);



}


template<class T, unsigned int D> void dispatch(cuNDArray<T>* in_out, std::vector<vector_td<T,3> > &stencils, int dim){
	std::vector<unsigned int> dims = *in_out->get_dimensions();
	dims.pop_back();

	cuNDArray<T> tmp(in_out->get_data_ptr(),&dims);


	unsigned int elements = std::accumulate(dims.begin(),dims.end(), 1, std::multiplies<unsigned int>());

	dim3 blockDim = std::min(elements,cudaDeviceManager::Instance()->max_blockdim());
	dim3 gridDim = (elements+blockDim.x-1)/blockDim;

	typename intd<D>::Type dims = to_intd( from_std_vector<unsigned int,D>(&dims));
	{
		cuNDArray<T> in(tmp);
		for (int i = 0; i < stencils.size(); i++)
			frameletKernel<T,D,3><<<gridDim,blockDim>>>(in->get_data_ptr(),in_out->get_data_ptr()+elements*i*dim,dims,dim,stencils[i]);
	}
	dims.push_back(in_out->get_size(D)/3);

	if (dim > 1)
		for (int i = 0; i < stencils.size(); i++){
			cuNDArray<T> tmp_in_out(in_out->get_data_ptr()+elements*i*dim,dims);
			dispatch<T,D>(&tmp_in_out,stencils,dim-1);
		}
}




template<class T, unsigned int D> void cuFrameletOperator<T,D>::set_domain_dimensions(std::vector<unsigned int>* dims){

	linearOperator<cuNDArray<T> >::set_domain_dimensions(dims);
	std::vector<unsigned int> newdims = *dims;
	newdims.push_back(Pow<3,D>::Value);
	linearOperator<cuNDArray<T> >::set_domain_codimensions(&newdims);
}
