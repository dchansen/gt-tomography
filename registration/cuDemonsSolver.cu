#include "cuDemonsSolver.h"
#include "setup_grid.h"
#include "gpureg_export.h"
#include "vector_td_utilities.h"
using namespace Gadgetron;

template< class T, unsigned int D> static inline  __device__ void partialDerivs(const T* in, const vector_td<unsigned int,D>& dims, vector_td<unsigned int,D>& co, T * out)
{

	T xi = in[co_to_idx<D>((co+dims)%dims,dims)];
	for (int i = 0; i < D; i++){
		co[i]+=1;
		T dt = in[co_to_idx<D>((co+dims)%dims,dims)];
		out[i] = dt-xi;
		co[i]-=1;
	}
}
/***
 *
 * @param fixed The fixed image
 * @param moving The Moving image
 * @param tot_elemens Total number of elements in fixed (and moving)
 * @param dims Dimensions of the subspace into which the convolution needs to be done
 * @param out Output vector field. Must have same dimensions as fixed and moving + an additional D dimension
 * @param alpha Regularization weight
 * @param beta Small constant added to prevent division by 0.
 */

template<class T, unsigned int D> static __global__ void demons_kernel(T* fixed, T* moving,  size_t tot_elements,const vector_td<unsigned int,D> dims, T* out,T alpha, T beta){

	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;


	if (idx <  tot_elements){

		size_t elements= prod(dims);

		unsigned int batch = idx/elements;
		T * fixed_batch = fixed+elements*batch;
		T * moving_batch = moving+elements*batch;


		T dmov[D];
		T dfix[D];

		vector_td<unsigned int,D> co = idx_to_co<D>(idx, dims);

		partialDerivs(fixed_batch,dims,co,dfix);
		partialDerivs(moving_batch,dims,co,dmov);
		T It = moving_batch[idx]-fixed_batch[idx];

		T gradNorm1 = 0;
		T gradNorm2 = 0;
		for (int i = 0; i < D; i++){
			gradNorm1 += dmov[i]*dmov[i];
			gradNorm2 += dfix[i]*dfix[i];
		}



		for(int i = 0; i < D; i++){
			out[idx+i*tot_elements] = It*(dmov[i]/(gradNorm1+alpha*alpha*It*It+beta)+dfix[i]/(gradNorm2+alpha*alpha*It*It+beta));

		}
	}

}



template<class T, unsigned int D>  boost::shared_ptr<cuNDArray<T> > cuDemonsSolver<T,D>::demonicStep(cuNDArray<T>* fixed,cuNDArray<T>* moving){



	std::vector<size_t> dims = *fixed->get_dimensions();
	dims.push_back(D);

	vector_td<unsigned int,D> idims = vector_td<unsigned int,D>(from_std_vector<size_t,D>(dims));
	boost::shared_ptr<cuNDArray<T> > out(new cuNDArray<T>(&dims));
	clear(out.get());

	dim3 gridDim;
	dim3 blockDim;
	setup_grid(fixed->get_number_of_elements(),&blockDim,&gridDim);

	demons_kernel<T,D><<< gridDim,blockDim>>>(fixed->get_data_ptr(),moving->get_data_ptr(),fixed->get_number_of_elements(),idims,out->get_data_ptr(),alpha,beta);


	return out;
}


// Simple transformation kernel
__global__ static void deform_imageKernel(float* output, float* vector_field, cudaTextureObject_t texObj, int width, int height, int depth){

	const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
	const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
	const int izo = blockDim.z * blockIdx.z + threadIdx.z;


	if (ixo < width && iyo < height && izo < depth){


		const int idx = ixo+iyo*width+izo*width*height;
		const int elements = width*height*depth;
		float ux = vector_field[idx]+0.5f+ixo;
		float uy = vector_field[idx+elements]+0.5f+iyo;
		float uz = vector_field[idx+2*elements]+0.5f+izo;

		output[idx] = tex3D<float>(texObj,ux,uy,uz);


	}




}



cuNDArray<float> deform_image(cuNDArray<float>* image, cuNDArray<float>* vector_field){


	cuNDArray<float> output(image->get_dimensions());
	clear(&output);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = image->get_size(0);
	extent.height = image->get_size(1);
	extent.depth = image->get_size(2);

	cudaMemcpy3DParms cpy_params = {0};
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.extent = extent;

	cudaArray *image_array;
	cudaMalloc3DArray(&image_array, &channelDesc, extent);
	cpy_params.dstArray = image_array;
	cpy_params.srcPtr = make_cudaPitchedPtr
			((void*)image->get_data_ptr(), extent.width*sizeof(float), extent.width, extent.height);
	cudaMemcpy3D(&cpy_params);

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = image_array;


	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	dim3 threads(8,8,8);

	dim3 grid((extent.width+threads.x-1)/threads.x, (extent.height+threads.y-1)/threads.y,(extent.depth+threads.z-1)/threads.z);

	deform_imageKernel<<<grid,threads>>>(output.get_data_ptr(),vector_field->get_data_ptr(),texObj,extent.width,extent.height,extent.depth);

	cudaDestroyTextureObject(texObj);

	// Free device memory
	cudaFreeArray(image_array);

	return output;




}

template class  cuDemonsSolver<float, 1>;
template class  cuDemonsSolver<float, 2>;
template class  cuDemonsSolver<float, 3>;
template class  cuDemonsSolver<float, 4>;

