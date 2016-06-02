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

		//T gradNorm1 = 0;
		//T gradNorm2 = 0;
        T gradNorm = 0;
		for (int i = 0; i < D; i++){
			//gradNorm1 += dmov[i]*dmov[i];
			//gradNorm2 += dfix[i]*dfix[i];
            T grad = 0.5f*(dmov[i]+dfix[i]);
            gradNorm += grad*grad;
		}



        vector_td<T,D> res;
		for(int i = 0; i < D; i++){
			//out[idx+i*tot_elements] = It*(dmov[i]/(gradNorm1+alpha*alpha*It*It+beta)+dfix[i]/(gradNorm2+alpha*alpha*It*It+beta));
            res[i] = -0.5f*It*(dmov[i]+dfix[i])/(gradNorm+alpha*alpha*It*It+beta);

		}
/*
        T length = norm(res);
        if (length > 2.0f)
            res *= 2.0f/length;
*/
        for (int i = 0; i < D; i++){
            out[idx+i*tot_elements] = res[i];
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

	demons_kernel<T,D><<< gridDim,blockDim>>>(fixed->get_data_ptr(),moving->get_data_ptr(),fixed->get_number_of_elements(),idims,out->get_data_ptr(),1.0/alpha,beta);


	return out;
}


template<class T, unsigned int D> void cuDemonsSolver<T,D>::compute( cuNDArray<T> *fixed_image, cuNDArray<T> *moving_image, cuNDArray<T> *stencil_image, boost::shared_ptr<cuNDArray<T> > &result ){


	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = moving_image->get_size(0);
	extent.height = moving_image->get_size(1);
	extent.depth = moving_image->get_size(2);

	cudaMemcpy3DParms cpy_params = {0};
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.extent = extent;

	cudaArray *image_array;
	cudaMalloc3DArray(&image_array, &channelDesc, extent);
	cpy_params.dstArray = image_array;
	cpy_params.srcPtr = make_cudaPitchedPtr
			((void*)moving_image->get_data_ptr(), extent.width*sizeof(float), extent.width, extent.height);
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
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	cuNDArray<T> def_moving(*moving_image);

	cuGaussianFilterOperator<T,D> gaussDiff;
	gaussDiff.set_sigma(sigmaDiff);

	cuGaussianFilterOperator<T,D> gaussFluid;
	gaussFluid.set_sigma(sigmaFluid);

	std::vector<size_t> image_dims = *moving_image->get_dimensions();
	std::vector<size_t> dims = *moving_image->get_dimensions();

	dims.push_back(D);

	if (!result.get()){
		result = boost::shared_ptr<cuNDArray<T> >(new cuNDArray<T>(&dims));
		clear(result.get());
	}

	for (int i = 0; i < this->max_num_iterations_per_level_; i++){
		//Calculate the gradients

		boost::shared_ptr<cuNDArray<T> > update = demonicStep(fixed_image,&def_moving);
		if (sigmaFluid > 0){
			cuNDArray<T> blurred_update(update->get_dimensions());
			gaussFluid.mult_M(update.get(),&blurred_update);
			//blurred_update = *update;
			std::cout << "Update step: " << nrm2(&blurred_update) << std::endl;

			deform_vfield(result.get(),&blurred_update);
			*result += blurred_update;




		} else {
			deform_vfield(result.get(),update.get());
			*result += *update;
		}

		if (sigmaDiff > 0){
			cuNDArray<T> blurred_result(*result);
			gaussDiff.mult_M(&blurred_result,result.get());
		}




		def_moving = deform_image(texObj,image_dims,result.get());

	}


	cudaDestroyTextureObject(texObj);

	// Free device memory
	cudaFreeArray(image_array);






}


// Simple transformation kernel
__global__ static void deform_vfieldKernel(float* output, float* vector_field, cudaTextureObject_t texX,cudaTextureObject_t texY,cudaTextureObject_t texZ, int width, int height, int depth){

	const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
	const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
	const int izo = blockDim.z * blockIdx.z + threadIdx.z;


	if (ixo < width && iyo < height && izo < depth){


		const int idx = ixo+iyo*width+izo*width*height;
		const int elements = width*height*depth;
		float ux = vector_field[idx]+0.5f+ixo;
		float uy = vector_field[idx+elements]+0.5f+iyo;
		float uz = vector_field[idx+2*elements]+0.5f+izo;

		output[idx] = tex3D<float>(texX,ux,uy,uz);

		output[idx+elements] = tex3D<float>(texY,ux,uy,uz);
		output[idx+2*elements] = tex3D<float>(texZ,ux,uy,uz);


	}




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



cuNDArray<float> Gadgetron::deform_image(cudaTextureObject_t  texObj,std::vector<size_t> dimensions, cuNDArray<float>* vector_field){


	cuNDArray<float> output(dimensions);
	clear(&output);

	dim3 threads(8,8,8);

	dim3 grid((dimensions[0]+threads.x-1)/threads.x, (dimensions[1]+threads.y-1)/threads.y,(dimensions[2]+threads.z-1)/threads.z);

	deform_imageKernel<<<grid,threads>>>(output.get_data_ptr(),vector_field->get_data_ptr(),texObj,dimensions[0],dimensions[1],dimensions[2]);


	return output;

}


void Gadgetron::deform_vfield(cuNDArray<float>* vfield1, cuNDArray<float>* vector_field){




	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = vfield1->get_size(0);
	extent.height = vfield1->get_size(1);
	extent.depth = vfield1->get_size(2);

	size_t elements = extent.height*extent.depth*extent.width;

	cudaMemcpy3DParms cpy_params = {0};
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.extent = extent;

	cudaArray *x_array;
	cudaArray *y_array;
	cudaArray *z_array;
	cudaMalloc3DArray(&x_array, &channelDesc, extent);
	cudaMalloc3DArray(&y_array, &channelDesc, extent);
	cudaMalloc3DArray(&z_array, &channelDesc, extent);


	//Copy x, y and z coordinates into their own textures.
	cpy_params.dstArray = x_array;
	cpy_params.srcPtr = make_cudaPitchedPtr
			((void*)vfield1->get_data_ptr(), extent.width*sizeof(float), extent.width, extent.height);
	cudaMemcpy3D(&cpy_params);
	cpy_params.dstArray = y_array;
	cpy_params.srcPtr = make_cudaPitchedPtr
			((void*)vfield1->get_data_ptr()+elements, extent.width*sizeof(float), extent.width, extent.height);
	cudaMemcpy3D(&cpy_params);
	cpy_params.dstArray = y_array;
	cpy_params.srcPtr = make_cudaPitchedPtr
			((void*)vfield1->get_data_ptr()+2*elements, extent.width*sizeof(float), extent.width, extent.height);
	cudaMemcpy3D(&cpy_params);

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = x_array;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaTextureObject_t texX = 0;
	cudaCreateTextureObject(&texX, &resDesc, &texDesc, NULL);




	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = y_array;


	cudaTextureObject_t texY = 0;
	cudaCreateTextureObject(&texY, &resDesc, &texDesc, NULL);


	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = z_array;


	cudaTextureObject_t texZ = 0;
	cudaCreateTextureObject(&texZ, &resDesc, &texDesc, NULL);

	dim3 threads(8,8,8);

	dim3 grid((extent.width+threads.x-1)/threads.x, (extent.height+threads.y-1)/threads.y,(extent.depth+threads.z-1)/threads.z);

	deform_vfieldKernel<<<grid,threads>>>(input.get_data_ptr(),vector_field->get_data_ptr(),texX,texY,texZ,extent.width,extent.height,extent.depth);

	cudaDestroyTextureObject(texX);
	cudaDestroyTextureObject(texY);
	cudaDestroyTextureObject(texZ);

	// Free device memory
	cudaFreeArray(x_array);
	cudaFreeArray(y_array);
	cudaFreeArray(z_array);



}

cuNDArray<float> Gadgetron::deform_image(cuNDArray<float>* image, cuNDArray<float>* vector_field){


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
	texDesc.addressMode[2] = cudaAddressModeClamp;
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


