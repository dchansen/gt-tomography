#include "cuBoxFilterOperator.h"

#include "gpuoperators_export.h"
#include "vector_td_utilities.h"

using namespace Gadgetron;




template<class T,int D> static __global__ void
box_kernel_simpel(T* __restrict__ in,  T* __restrict__ out, vector_td<int,3> dims,int direction){

	const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
	const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
	const int izo = blockDim.z * blockIdx.z + threadIdx.z;


	vector_td<int,3> coord(ixo,iyo,izo);
	vector_td<int,3> coord2(ixo,iyo,izo);
	if (ixo < dims[0] && iyo < dims[1] && izo < dims[2]){


		T res = T(0);


		for (int i = 0; i < 2; i++){
			coord2[D] = coord[D]+ direction*i;
			res += in[co_to_idx<3>((coord2+dims)%dims,dims)];

		}

		//atomicAdd(&out[co_to_idx<3>(coord,dims)],res/norm);
		out[co_to_idx<3>(coord,dims)] = res*0.5f;
	}

}
template<class T, unsigned int D> static void boxFilter(cuNDArray<T>* in,cuNDArray<T>* out, bool accumulate,int direction){


	vector_td<size_t,D> dims = from_std_vector<size_t,D>(*(in->get_dimensions()));

	*out = *in;
	std::vector<size_t> batch_dim = to_std_vector(dims);
	size_t elements = prod(dims);
	for (int batch =0; batch < in->get_number_of_elements()/elements; batch++){


		T* outptr = out->get_data_ptr()+batch*elements;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaExtent extent;
		extent.width = in->get_size(0);
		extent.height = in->get_size(1);
		extent.depth = in->get_size(2);

		dim3 threads(8,8,8);

		dim3 grid((extent.width+threads.x-1)/threads.x, (extent.height+threads.y-1)/threads.y,(extent.depth+threads.z-1)/threads.z);

		for (int d = 0; d < D; d++){
			if (dims[d] ==1) continue;

			cuNDArray<T> tmp(elements);
			cudaMemcpy(tmp.get_data_ptr(),outptr,elements*sizeof(T),cudaMemcpyDeviceToDevice);
			if (d == 0)
				box_kernel_simpel<T,0><<<grid,threads>>>(tmp.get_data_ptr(),outptr,vector_td<int,3>(extent.width,extent.height,extent.depth),direction);
			else if (d == 1)
				box_kernel_simpel<T,1><<<grid,threads>>>(tmp.get_data_ptr(),outptr,vector_td<int,3>(extent.width,extent.height,extent.depth),direction);
			else if (d == 2)
				box_kernel_simpel<T,2><<<grid,threads>>>(tmp.get_data_ptr(),outptr,vector_td<int,3>(extent.width,extent.height,extent.depth),direction);
			else throw std::runtime_error("Unsupported number of dimensions for Gaussian kernel");



		}




		//cudaFreeArray(image_array);
		if (accumulate) *out += *in;


	}



}


template<class T, unsigned int D> void cuBoxFilterOperator<T,D>::mult_M(cuNDArray<T>* in,cuNDArray<T>* out, bool accumulate){
	boxFilter<T,D>(in,out,accumulate,1);
}


template<class T, unsigned int D> void cuBoxFilterOperator<T,D>::mult_MH(cuNDArray<T>* in,cuNDArray<T>* out, bool accumulate){
	boxFilter<T,D>(in,out,accumulate,-1);
}


template EXPORTGPUOPERATORS class cuBoxFilterOperator<float,1>;
template EXPORTGPUOPERATORS class cuBoxFilterOperator<float,2>;
template EXPORTGPUOPERATORS class cuBoxFilterOperator<float,3>;



