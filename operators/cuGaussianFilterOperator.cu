#include "cuGaussianFilterOperator.h"

#include "gpuoperators_export.h"
#include "vector_td_utilities.h"

using namespace Gadgetron;


template<class T,int D> static __global__ void
gauss_kernel(cudaTextureObject_t texObj,  T* out, int width, int height, int depth,T sigma){

	const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
	const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
	const int izo = blockDim.z * blockIdx.z + threadIdx.z;

	const float x = ixo+0.5f;
	const float y = iyo+0.5f;
	const float z = izo+0.5f;

	vector_td<float,3> coord(x,y,z);
	vector_td<float,3> coord2(x,y,z);
	if (ixo < width && iyo < height && izo < depth){

		int steps = max(ceil(sigma*3),3.0f);

		T res = T(0);

		T norm = 0;
		for (int i = -steps; i < steps; i++){
			coord2[D] = coord[D]+ i;
			T weight = expf(-0.5f*T(i*i)/(2*sigma*sigma));
			norm += weight;
			res += weight*tex3D<float>(texObj,coord2[0],coord2[1],coord2[2]);

		}

		atomicAdd(&out[ixo+iyo*width+izo*width*height],res/norm);
		//out[ixo+iyo*width+izo*width*height] = res/norm;
	}

}


template<class T,int D> static __global__ void
gauss_kernel_simpel(T* __restrict__ in,  T* __restrict__ out, vector_td<int,3> dims ,T sigma){

	const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
	const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
	const int izo = blockDim.z * blockIdx.z + threadIdx.z;


	vector_td<int,3> coord(ixo,iyo,izo);
	vector_td<int,3> coord2(ixo,iyo,izo);
	if (ixo < dims[0] && iyo < dims[1] && izo < dims[2]){

		int steps = max(ceil(sigma*3),T(3));

		T res = T(0);
		T norm = 0;
		for (int i = -steps; i < steps; i++){
			coord2[D] = (coord[D]+ i+dims[D])%dims[D];
			T weight = expf(-0.5f*T(i*i)/(2*sigma*sigma));
			norm += weight;
			res += weight*in[co_to_idx<3>(coord2,dims)];

		}

		//atomicAdd(&out[co_to_idx<3>(coord,dims)],res/norm);
		out[co_to_idx<3>(coord,dims)] = res/norm;
	}

}

template<class T> static __global__ void
gauss_kernel3D(cudaTextureObject_t texObj,  T* out, int width, int height, int depth,T sigma){

	const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
	const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
	const int izo = blockDim.z * blockIdx.z + threadIdx.z;

	const float x = ixo+0.5f;
	const float y = iyo+0.5f;
	const float z = izo+0.5f;


	if (ixo < width && iyo < height && izo < depth){


		int steps = max(ceil(sigma*3),1);

		T res = T(0);

		T norm = 0;
		for (int dz = -steps; dz <= steps; dz++) {
			for (int dy = -steps; dy <= steps; dy++) {
				for (int dx = -steps; dx <= steps; dx++) {


					T weight = expf(-0.5f * T(dx*dx+dy*dy+dz*dz) / (2 * sigma * sigma));
					norm += weight;
					res += weight * tex3D<float>(texObj, x+dx,y+dy,z+dz);

				}
			}
		}

		atomicAdd(&out[ixo+iyo*width+izo*width*height],res/norm);
		//out[ixo+iyo*width+izo*width*height] = res/norm;
	}

}








template<class T, unsigned int D> void cuGaussianFilterOperator<T,D>::mult_M(cuNDArray<T>* in,cuNDArray<T>* out, bool accumulate){


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
/*
		cudaMemcpy3DParms cpy_params = {0};
		cpy_params.kind = cudaMemcpyDeviceToDevice;
		cpy_params.extent = extent;

		cudaArray *image_array;
		cudaMalloc3DArray(&image_array, &channelDesc, extent);
		cpy_params.dstArray = image_array;
		cpy_params.srcPtr = make_cudaPitchedPtr
				((void*)outptr, extent.width*sizeof(float), extent.width, extent.height);
		cudaMemcpy3D(&cpy_params);

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = image_array;


		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;
*/
		dim3 threads(8,8,8);

		dim3 grid((extent.width+threads.x-1)/threads.x, (extent.height+threads.y-1)/threads.y,(extent.depth+threads.z-1)/threads.z);

		for (int d = 0; d < D; d++){
			if (dims[d] ==1) continue;

			cuNDArray<T> tmp(elements);
			cudaMemcpy(tmp.get_data_ptr(),outptr,elements*sizeof(T),cudaMemcpyDeviceToDevice);
			if (d == 0)
				gauss_kernel_simpel<T,0><<<grid,threads>>>(tmp.get_data_ptr(),outptr,vector_td<int,3>(extent.width,extent.height,extent.depth),_sigma[d]);
			else if (d == 1)
				gauss_kernel_simpel<T,1><<<grid,threads>>>(tmp.get_data_ptr(),outptr,vector_td<int,3>(extent.width,extent.height,extent.depth),_sigma[d]);
			else if (d == 2)
				gauss_kernel_simpel<T,2><<<grid,threads>>>(tmp.get_data_ptr(),outptr,vector_td<int,3>(extent.width,extent.height,extent.depth),_sigma[d]);
			else throw std::runtime_error("Unsupported number of dimensions for Gaussian kernel");



		}
/*
		for (int d = 0; d < D; d++){
			if (dims[d] ==1) continue;

			cpy_params.dstArray = image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr
					((void*)outptr, extent.width*sizeof(float), extent.width, extent.height);
			cudaMemcpy3D(&cpy_params);
			cudaTextureObject_t texObj = 0;
			cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
			cudaMemset(outptr,0,elements*sizeof(float));
			if (d == 0)
				gauss_kernel<T,0><<<grid,threads>>>(texObj,outptr,extent.width,extent.height,extent.depth,_sigma);
			else if (d == 1)
				gauss_kernel<T,1><<<grid,threads>>>(texObj,outptr,extent.width,extent.height,extent.depth,_sigma);
			else if (d == 2)
				gauss_kernel<T,2><<<grid,threads>>>(texObj,outptr,extent.width,extent.height,extent.depth,_sigma);
			else throw std::runtime_error("Unsupported number of dimensions for Gaussian kernel");
			cudaDestroyTextureObject(texObj);


		}
*/

/*
			cpy_params.dstArray = image_array;
			cpy_params.srcPtr = make_cudaPitchedPtr
					((void*)outptr, extent.width*sizeof(float), extent.width, extent.height);
			cudaMemcpy3D(&cpy_params);
			cudaTextureObject_t texObj = 0;
			cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
			cudaMemset(outptr,0,elements*sizeof(float));

			gauss_kernel3D<T><<<grid,threads>>>(texObj,outptr,extent.width,extent.height,extent.depth,_sigma);

			cudaDestroyTextureObject(texObj);
*/



		//cudaFreeArray(image_array);
		if (accumulate) *out += *in;


	}



}


template<class T, unsigned int D> void cuGaussianFilterOperator<T,D>::mult_MH(cuNDArray<T>* in,cuNDArray<T>* out, bool accumulate){

	this->mult_M(in,out,accumulate);

}


template EXPORTGPUOPERATORS class cuGaussianFilterOperator<float,1>;
template EXPORTGPUOPERATORS class cuGaussianFilterOperator<float,2>;
template EXPORTGPUOPERATORS class cuGaussianFilterOperator<float,3>;



