
#include <vector_td_utilities.h>
#include "cuTVPrimalDualOperator.h"
#include "cuPartialDifferenceOperator.h"
#include <cuNDArray_math.h>
#include <cuSolverUtils.h>
#include <cuNDArray_fileio.h>

using namespace Gadgetron;
template<class T> __global__ static void cuTVPrimalKernel(T* __restrict__ in, T* __restrict__ out, vector_td<int,3> dims, T omega,T weight){

    const int elements = prod(dims);
	
    const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	const int iy = blockIdx.y*blockDim.y+threadIdx.y;
	const int iz = blockIdx.z*blockDim.z+threadIdx.z;
	const auto co = vector_td<int,3>(ix,iy,iz);
	vector_td<float,3> result;
    if (co < dims){
		auto val1 = in[co_to_idx(co,dims)];
		auto co2 = co;
        for (int i = 0; i < 3; i++){
			//co[DIM] = (co[DIM]+SKIP+dims[DIM])%dims[DIM];
			
			co2[i] = (co[i]+dims[i]+1)%dims[i];
			result[i] = in[co_to_idx(co2,dims)]-val1;
			co2[i] = co[i];
		}
		result *= weight/(T(1.0)+omega);
		result /= max(T(1),norm(result));
        const int idx = ix+iy*dims[0]+iz*dims[0]*dims[1];
		for (int i =0; i < 3; i++)
			atomicAdd(&out[idx+i*elements],result[i]);
		
    }

};


template<class T> __global__ static void cuTVDualKernel(T* __restrict__ in, T* __restrict__ out, vector_td<int,3> dims,T weight){

    const int elements = prod(dims);

    const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	const int iy = blockIdx.y*blockDim.y+threadIdx.y;
	const int iz = blockIdx.z*blockDim.z+threadIdx.z;
	const auto co = vector_td<int,3>(ix,iy,iz);

    if (co < dims){
		const int idx = ix+iy*dims[0]+iz*dims[0]*dims[1];
		T result = 0;
		auto co2 = co;
        for (int i = 0; i < 3; i++){
			auto val1 = in[idx+i*elements];

			co2[i] = (co[i]+dims[i]-1)%dims[i];
			result += in[co_to_idx(co2,dims)+i*elements]-val1;
			co2[i] = co[i];

		}
		atomicAdd(&out[idx],result*weight);
    }

};


template<class T> void Gadgetron::cuTVPrimalDualOperator<T>::primalDual(cuNDArray<T>* in, cuNDArray<T>* out, T sigma, bool accumulate){

	if (!in->dimensions_equal(out))
	throw std::runtime_error("Input and reference dimensions must agree");

	if (!accumulate) clear(out);

	auto dims3D = std::vector<size_t>{in->get_size(0),in->get_size(1),in->get_size(2)};
	vector_td<int,3> dims = vector_td<int,3>(from_std_vector<size_t,3>(dims3D));


	dim3 threads(8, 8,8);
	dim3 grid((dims[0]+threads.x-1)/threads.x, (dims[1]+threads.y-1)/threads.y,(dims[2]+threads.z-1)/threads.z);

	T* data_in = in->get_data_ptr();
	T* data_out = out->get_data_ptr();

	auto dimsGrad3d = dims3D;
	dimsGrad3d.push_back(3);

	cuNDArray<T> grad3D(dimsGrad3d);

	size_t remaining = 1;
	for (int i = 3; i < in->get_number_of_dimensions(); i++) remaining *= in->get_size(i);

	for (int i = 0; i < remaining; i++){
		clear(&grad3D);

		cuTVPrimalKernel<<<grid,threads>>>(data_in,grad3D.get_data_ptr(),dims,sigma*alpha,this->weight*sigma);
		cudaDeviceSynchronize();

		cuTVDualKernel<<<grid,threads>>>(grad3D.get_data_ptr(),data_out,dims,this->weight);
		data_in += prod(dims);
		data_out += prod(dims);
	}


};





template class Gadgetron::cuTVPrimalDualOperator<float>;
