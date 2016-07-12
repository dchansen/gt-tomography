#include <cuPartialDerivativeOperator.h>
#include <boost/make_shared.hpp>
#include <cuNDArray_fileio.h>
#include "cuDemonsSolver.h"
#include "setup_grid.h"
#include "gpureg_export.h"
#include "vector_td_utilities.h"
#include <thrust/extrema.h>
#include "morphon.h"
using namespace Gadgetron;



struct vsquarefunctor : public thrust::unary_function<thrust::tuple<float,float,float>,float> {
	__host__ __device__ float operator()(thrust::tuple<float,float,float> tup){
		float x = thrust::get<0>(tup);
		float y = thrust::get<1>(tup);
		float z = thrust::get<2>(tup);
		return x*x+y*y+z*z;
	}
};

static void vfield_exponential(cuNDArray<float>* vfield){

	auto dims3D = *vfield->get_dimensions();
	dims3D.pop_back();

	size_t elements = dims3D[0]*dims3D[1]*dims3D[2];

	cuNDArray<float> xview(dims3D,vfield->get_data_ptr());
	cuNDArray<float> yview(dims3D,vfield->get_data_ptr()+elements);
	cuNDArray<float> zview(dims3D,vfield->get_data_ptr()+elements*2);

	auto iter = thrust::make_zip_iterator(thrust::make_tuple(xview.begin(),yview.begin(),zview.begin()));
	auto iter_end = thrust::make_zip_iterator(thrust::make_tuple(xview.end(),yview.end(),zview.end()));
	auto msquare = thrust::max_element(thrust::make_transform_iterator(iter,vsquarefunctor()),
								  thrust::make_transform_iterator(iter_end,vsquarefunctor()));



	int n = ceil(2+log2(sqrt(*msquare)/0.5));
	n = std::max(n,0);
	std::cout << " N " << n << " " << float(std::pow(2,-float(n))) << " " << sqrt(*msquare) << std::endl;
	*vfield *= float(std::pow(2,-float(n)));

	for (int i =0; i < n; i++) {
		cuNDArray<float> vfield_copy(*vfield);
		deform_vfield(vfield,&vfield_copy);
		*vfield += vfield_copy;
	}



}

template< class T, unsigned int D> static inline  __device__ void partialDerivs(const T* in, const vector_td<int,D>& dims, vector_td<int,D>& co, vector_td<T,D>& out)
{

	vector_td<int,D> co2 = co;
	//T xi = in[co_to_idx<D>(co,dims)];
	for (int i = 0; i < D; i++){
		co2[i] = min(co[i]+1,dims[i]-1);
		T dt = in[co_to_idx<D>(co2,dims)];
		co2[i] = max(co[i]-1,0);
		T xi = in[co_to_idx<D>(co2,dims)];
		out[i] = (dt-xi)*0.5f;
		co2[i] = co[i];
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

template<class T, unsigned int D> static __global__ void demons_kernel(T* fixed, T* moving,  size_t tot_elements,const vector_td<int,D> dims, T* out,T alpha, T beta){

	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;


	if (idx <  tot_elements){

		size_t elements= prod(dims);

		int batch = idx/elements;
		T * fixed_batch = fixed+elements*batch;
		T * moving_batch = moving+elements*batch;


		vector_td<T,D> dmov;
		vector_td<T,D> dfix;

		vector_td<int,D> co = idx_to_co<D>(idx, dims);

		partialDerivs(fixed_batch,dims,co,dfix);
		partialDerivs(moving_batch,dims,co,dmov);
		T It = fixed_batch[idx]-moving_batch[idx];
		//T It = moving_batch[idx]-fixed_batch[idx];

		//T gradNorm1 = 0;
		//T gradNorm2 = 0;


		dmov += dfix;
		dmov *= 0.5f;
        T gradNorm = norm_squared(dmov);




        vector_td<T,D> res;
		for(int i = 0; i < D; i++){
			//out[idx+i*tot_elements] = It*(dmov[i]/(gradNorm1+alpha*alpha*It*It+beta)+dfix[i]/(gradNorm2+alpha*alpha*It*It+beta));
            res[i] = It*(dmov[i])/(gradNorm+alpha*alpha*It*It+beta);
			//res[i] = 0.5f*It*(dmov[i]/(norm_squared(dmov)+alpha*alpha*It*It+beta)+dfix[i]/(norm_squared(dfix)+alpha*alpha*It*It+beta));
			//res[i] = It*dfix[i]/(norm_squared(dfix)+alpha*alpha*It*It+beta);

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

	vector_td<int,D> idims = vector_td<int,D>(from_std_vector<size_t,D>(dims));
	boost::shared_ptr<cuNDArray<T> > out(new cuNDArray<T>(&dims));
	clear(out.get());

	dim3 gridDim;
	dim3 blockDim;
	setup_grid(fixed->get_number_of_elements(),&blockDim,&gridDim);

	demons_kernel<T,D><<< gridDim,blockDim>>>(fixed->get_data_ptr(),moving->get_data_ptr(),fixed->get_number_of_elements(),idims,out->get_data_ptr(),1.0/alpha,beta);


	return out;
}


template<class T, unsigned int D> static __global__ void NGF_kernel(T* image,   size_t tot_elements,const vector_td<int,D> dims, T* out,T eps){

    const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;


    if (idx <  tot_elements){
        vector_td<T,D> diff;
        vector_td<int,D> co = idx_to_co<D>(idx, dims);

        partialDerivs(image,dims,co,diff);

        T dnorm = sqrt(norm_squared(diff)+eps);
/*
        T length = norm(res);
        if (length > 2.0f)
            res *= 2.0f/length;
*/
        for (int i = 0; i < D; i++){
            out[idx+i*tot_elements] = diff[i]/dnorm;
        }
    }

}


template<class T, unsigned int D> static boost::shared_ptr<cuNDArray<T> > normalized_gradient_field(cuNDArray<T>* image,T eps){



    std::vector<size_t> dims = *image->get_dimensions();
    dims.push_back(D);

    vector_td<int,D> idims = vector_td<int,D>(from_std_vector<size_t,D>(dims));
    boost::shared_ptr<cuNDArray<T> > out(new cuNDArray<T>(&dims));
    clear(out.get());

    dim3 gridDim;
    dim3 blockDim;
    setup_grid(image->get_number_of_elements(),&blockDim,&gridDim);

    NGF_kernel<T,D><<< gridDim,blockDim>>>(image->get_data_ptr(),image->get_number_of_elements(),idims,out->get_data_ptr(),eps);


    return out;
}


template<class T, unsigned int D> boost::shared_ptr<cuNDArray<T>> cuDemonsSolver<T,D>::registration( cuNDArray<T> *fixed_image, cuNDArray<T> *moving_image){


	auto vdims = *fixed_image->get_dimensions();
	vdims.push_back(D);
	auto result = boost::make_shared<cuNDArray<T>>(vdims);
	clear(result.get());
	single_level_reg(fixed_image,moving_image,result.get());

	return result;



}


template<class T, unsigned int D> boost::shared_ptr<cuNDArray<T>> cuDemonsSolver<T,D>::multi_level_reg( cuNDArray<T> *fixed_image, cuNDArray<T> *moving_image,int levels, float scale){

	std::cout << "Level " << levels << std::endl;
	auto vdims = *fixed_image->get_dimensions();
	vdims.push_back(D);
	auto result = boost::make_shared<cuNDArray<T>>(vdims);
	clear(result.get());
	if (levels <= 1) {
		single_level_reg(fixed_image, moving_image, result.get(),scale);
	} else {

		auto d_fixed = downsample<T,D>(fixed_image);
		auto d_moving = downsample<T,D>(moving_image);
		auto tmp_res = multi_level_reg(d_fixed.get(),d_moving.get(),levels-1,scale/2);

		auto dims = *tmp_res->get_dimensions();
		for (auto d : dims)
			std::cout << d << " ";
		std::cout << std::endl;


		upscale_vfield(tmp_res.get(),result.get());
		*result *= T(2);

		single_level_reg(fixed_image,moving_image,result.get(),scale);

	}

	return result;




}


template<class T, unsigned int D> void cuDemonsSolver<T,D>::single_level_reg( cuNDArray<T> *fixed_image, cuNDArray<T> *moving_image,cuNDArray<T>* result,float scale){


	auto vdims = *fixed_image->get_dimensions();
	vdims.push_back(D);

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



	cuGaussianFilterOperator<T,D> gaussDiff;
	gaussDiff.set_sigma(sigmaDiff*T(scale));

	cuGaussianFilterOperator<T,D> gaussFluid;
	gaussFluid.set_sigma(sigmaFluid*scale);



	std::vector<size_t> image_dims = *moving_image->get_dimensions();
	std::vector<size_t> dims = *moving_image->get_dimensions();

	auto def_moving = deform_image(texObj,image_dims,result);

	dims.push_back(D);



	for (int i = 0; i < iterations; i++){
		//Calculate the gradients
		boost::shared_ptr<cuNDArray<T> > update;
		if (epsilonNGF > 0) {
			auto diff_fixed = normalized_gradient_field<T,D>(fixed_image,epsilonNGF);
			auto diff_moving = normalized_gradient_field<T,D>(&def_moving,epsilonNGF);
			//write_nd_array(diff_fixed.get(),"diff_fixed.real");
			//write_nd_array(diff_m.get(),"diff_fixed.real");
			update = boost::make_shared<cuNDArray<T>>(diff_fixed->get_dimensions());
			clear(update.get());
			size_t elements = fixed_image->get_number_of_elements();
			auto dims = *fixed_image->get_dimensions();

			for (int d = 0; d < D; d++){
				cuNDArray<T> f_view(dims,diff_fixed->get_data_ptr()+d*elements);
				cuNDArray<T> m_view(dims,diff_moving->get_data_ptr()+d*elements);
				*update += *demonicStep(&f_view, &m_view);
			}
			*update /= T(3);

		} else {
			update = demonicStep(fixed_image, &def_moving);
			//update = morphon(&def_moving,fixed_image);
			std::cout << "Updated norm " << nrm2(update.get()) << std::endl;
		}
		if (sigmaFluid > 0){
			cuNDArray<T> blurred_update(update->get_dimensions());
			gaussFluid.mult_M(update.get(),&blurred_update);
			//blurred_update = *update;


			if (exponential) vfield_exponential(&blurred_update);
			if (compositive) deform_vfield(result,&blurred_update);
			*result += blurred_update;




		} else {

			if (exponential) vfield_exponential(update.get());
			if (compositive) deform_vfield(result,update.get());
			*result += *update;
		}


		if (sigmaDiff > vector_td<T,D>(0)){
			if (sigmaInt >0 || sigmaVDiff > 0)
				bilateral_vfield(result,&def_moving,sigmaDiff*T(scale),sigmaInt,sigmaVDiff);
			else {
				cuNDArray<T> blurred_result(*result);

				gaussDiff.mult_M(&blurred_result, result);
			}
		}




		def_moving = deform_image(texObj,image_dims,result);
		{
			cuNDArray<T> tmp = *fixed_image;
			tmp -= def_moving;
			std::cout << "Diff " << nrm2(&tmp) << std::endl;
		}

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
			((void*)(vfield1->get_data_ptr()+elements), extent.width*sizeof(float), extent.width, extent.height);
	cudaMemcpy3D(&cpy_params);
	cpy_params.dstArray = z_array;
	cpy_params.srcPtr = make_cudaPitchedPtr
			((void*)(vfield1->get_data_ptr()+2*elements), extent.width*sizeof(float), extent.width, extent.height);
	cudaMemcpy3D(&cpy_params);

	struct cudaResourceDesc resDescX;
	memset(&resDescX, 0, sizeof(resDescX));
	resDescX.resType = cudaResourceTypeArray;
	resDescX.res.array.array = x_array;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaTextureObject_t texX = 0;
	cudaCreateTextureObject(&texX, &resDescX, &texDesc, NULL);





	struct cudaResourceDesc resDescY;
	memset(&resDescY, 0, sizeof(resDescY));
	resDescY.resType = cudaResourceTypeArray;
	resDescY.res.array.array = y_array;


	cudaTextureObject_t texY = 0;
	cudaCreateTextureObject(&texY, &resDescY, &texDesc, NULL);


	struct cudaResourceDesc resDescZ;
	memset(&resDescZ, 0, sizeof(resDescZ));
	resDescZ.resType = cudaResourceTypeArray;
	resDescZ.res.array.array = z_array;



	cudaTextureObject_t texZ = 0;
	cudaCreateTextureObject(&texZ, &resDescZ, &texDesc, NULL);


	cudaDeviceSynchronize();
	dim3 threads(8,8,8);

	dim3 grid((extent.width+threads.x-1)/threads.x, (extent.height+threads.y-1)/threads.y,(extent.depth+threads.z-1)/threads.z);

	deform_vfieldKernel<<<grid,threads>>>(vfield1->get_data_ptr(),vector_field->get_data_ptr(),texX,texY,texZ,extent.width,extent.height,extent.depth);

	cudaDeviceSynchronize();
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


// Simple transformation kernel
__global__ static void upscale_vfieldKernel(float* output, cudaTextureObject_t texObj, int width, int height, int depth){

	const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
	const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
	const int izo = blockDim.z * blockIdx.z + threadIdx.z;


	if (ixo < width && iyo < height && izo < depth){


		const int idx = ixo+iyo*width+izo*width*height;
		float ux = 0.5f+ixo;
		float uy = 0.5f+iyo;
		float uz = 0.5f+izo;

		output[idx] = tex3D<float>(texObj,ux/width,uy/height,uz/depth);


	}




}

template<class T, unsigned int D> void cuDemonsSolver<T,D>::upscale_vfield(cuNDArray<T> *in, cuNDArray<T> *out){




	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = in->get_size(0);
	extent.height = in->get_size(1);
	extent.depth = in->get_size(2);


	size_t elements = extent.depth*extent.height*extent.height;
	cudaMemcpy3DParms cpy_params = {0};
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.extent = extent;

	cudaArray *image_array;
	cudaMalloc3DArray(&image_array, &channelDesc, extent);

	for (int i = 0; i < in->get_size(3); i++) {
		cpy_params.dstArray = image_array;
		cpy_params.srcPtr = make_cudaPitchedPtr
				((void *) (in->get_data_ptr()+i*elements), extent.width * sizeof(float), extent.width, extent.height);
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
		texDesc.normalizedCoords = 1;
		cudaTextureObject_t texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

		dim3 threads(8, 8, 8);

		dim3 grid((out->get_size(0) + threads.x - 1) / threads.x, (out->get_size(1) + threads.y - 1) / threads.y,
				  (out->get_size(2) + threads.z - 1) / threads.z);

		upscale_vfieldKernel<< < grid, threads >> >
									  (out->get_data_ptr()+i*elements, texObj, out->get_size(0),out->get_size(1),out->get_size(2));

		cudaDestroyTextureObject(texObj);

	}
	cudaFreeArray(image_array);
	// Free device memory




}



static __global__ void jacobian_kernel(float* __restrict__ image,  const vector_td<int,3> dims, float * __restrict__ out){

	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	int elements= prod(dims);

	if (idx <  elements){

		vector_td<float,3> dX;
		vector_td<float,3> dY;
		vector_td<float,3> dZ;


		vector_td<int,3> co = idx_to_co<3>(idx, dims);

		partialDerivs(image,dims,co,dX);
		partialDerivs(image+elements,dims,co,dY);
		partialDerivs(image+elements*2,dims,co,dZ);

		out[idx] = (1+dX[0])*(1+dY[1])*(1+dZ[2]) +
				(1+dX[0])*dY[2]*dZ[1] +
				dX[1]*dY[0]*(1+dZ[2]) +
				dX[2]*dY[0]*dZ[1] +
				dX[1]*dY[2]*dZ[0] +
				dX[2]*(1+dY[1])*dZ[0];
	}

}



cuNDArray<float> Gadgetron::Jacobian(cuNDArray<float>* vfield){



	std::vector<size_t> dims = *vfield->get_dimensions();
	dims.pop_back();

	vector_td<int,3> idims = vector_td<int,3>(from_std_vector<size_t,3>(dims));
	cuNDArray<float> out(dims);


	dim3 gridDim;
	dim3 blockDim;
	setup_grid(out.get_number_of_elements(),&blockDim,&gridDim);

	jacobian_kernel<<< gridDim,blockDim>>>(vfield->get_data_ptr(),idims,out.get_data_ptr());


	return out;
}


static __global__ void
bilateral_kernel3D(float* __restrict__ out, const float* __restrict__ vfield, const float* __restrict__ image, int width, int height, int depth,floatd3 sigma_spatial,float sigma_int,float sigma_diff){

	const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
	const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
	const int izo = blockDim.z * blockIdx.z + threadIdx.z;

	const int idx = ixo+iyo*width+izo*width*height;
	int elements= width*height*depth;



	if (ixo < width && iyo < height && izo < depth){


		int steps = 8;

		vector_td<float,3> vec(vfield[idx],vfield[idx+elements],vfield[idx+elements*2]);

		vector_td<float,3> res(0);
		float image_value = image[idx];
		float norm =0;
		for (int dz = -steps; dz <= steps; dz++) {
			int z = (izo+dz+depth)%depth;

			for (int dy = -steps; dy <= steps; dy++) {
				int y = (iyo+dy+height)%height;
				for (int dx = -steps; dx <= steps; dx++) {
					int x = (ixo+dx+width)%width;


					const int idx2 = x+y*width+z*width*height;

					vector_td<float,3> vec2 (vfield[idx2],vfield[idx2+elements],vfield[idx2+elements*2]);

					float image_diff = image_value-image[idx2];
					float vdiff = (vec2[0]-vec[0])*(vec2[0]-vec[0])+(vec2[1]-vec[1])*(vec2[1]-vec[1])+(vec2[2]-vec[2])*(vec2[2]-vec[2]);
					float weight = expf(-float(dx*dx) / (2 * sigma_spatial[0] * sigma_spatial[0])
										-float(dy*dy) / (2 * sigma_spatial[1] * sigma_spatial[1])
										 -float(dz*dz) / (2 * sigma_spatial[2] * sigma_spatial[2])
									-image_diff*image_diff/(2*sigma_int*sigma_int)
									- vdiff/(2*sigma_diff*sigma_diff));
					norm += weight;
					res[0] += weight * vec2[0];
					res[1] += weight * vec2[1];
					res[2] += weight * vec2[2];

				}
			}
		}

		int idx = ixo+iyo*width+izo*width*height;
		int elements = height*width*depth;
		out[idx] = res[0]/norm;
		out[idx+elements] = res[1]/norm;
		out[idx+2*elements] = res[2]/norm;
	}

}


template<int D> static __global__ void
bilateral_kernel1D(float* __restrict__ out, const float* __restrict__ vfield, const float* __restrict__ image, vector_td<int,3> dims,float sigma_spatial,float sigma_int,float sigma_diff){

	const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
	const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
	const int izo = blockDim.z * blockIdx.z + threadIdx.z;



	if (ixo < dims[0] && iyo < dims[1] && izo < dims[2]){
		vector_td<int,3> coord(ixo,iyo,izo);
		vector_td<int,3> coord2(ixo,iyo,izo);
		int elements = prod(dims);
		int steps = max(ceil(sigma_spatial*4),1.0f);

		int idx = co_to_idx<3>(coord,dims);
		vector_td<float,3> res(0);
		float image_value = image[idx];
		vector_td<float,3> vec(vfield[idx],vfield[idx+elements],vfield[idx+2*elements]);

		float norm = 0;

		for (int i = -steps; i < steps; i++){
			coord2[D] = coord[D]+ i;

			int idx2 = co_to_idx<3>((coord2+dims)%dims,dims);
			vector_td<float,3> vec2(vfield[idx2],vfield[idx2+elements],vfield[idx2+2*elements]);
			float image_diff = image_value-image[idx2];
			float vdiff = (vec2[0]-vec[0])*(vec2[0]-vec[0])+(vec2[1]-vec[1])*(vec2[1]-vec[1])+(vec2[2]-vec[2])*(vec2[2]-vec[2]);
			float weight = expf(-0.5f*float(i*i)/(2*sigma_spatial*sigma_spatial)
								-image_diff*image_diff/(2*sigma_int*sigma_int)
								- vdiff/(2*sigma_diff*sigma_diff));
			norm += weight;



			res += weight*vec2;

		}

		//atomicAdd(&out[co_to_idx<3>(coord,dims)],res/norm);



		out[idx] = res[0]/norm;
		out[idx+elements] = res[1]/norm;
		out[idx+2*elements] = res[2]/norm;
	}

}




void Gadgetron::bilateral_vfield(cuNDArray<float>* vfield1, cuNDArray<float>* image, floatd3  sigma_spatial,float sigma_int, float sigma_diff){
	cudaExtent extent;
	extent.width = vfield1->get_size(0);
	extent.height = vfield1->get_size(1);
	extent.depth = vfield1->get_size(2);
/*
	cudaExtent extent;
	extent.width = vfield1->get_size(0);
	extent.height = vfield1->get_size(1);
	extent.depth = vfield1->get_size(2);


	cudaDeviceSynchronize();
	dim3 threads(8,8,8);

	auto vfield_copy = *vfield1;
	dim3 grid((extent.width+threads.x-1)/threads.x, (extent.height+threads.y-1)/threads.y,(extent.depth+threads.z-1)/threads.z);

	bilateral_kernel3D<<<grid,threads>>>(vfield1->get_data_ptr(),vfield_copy.get_data_ptr(),image->get_data_ptr(),extent.width,extent.height,extent.depth,sigma_spatial,sigma_int,sigma_diff);
*/

	/*
	dim3 threads;
	dim3 grid;
	setup_grid(image->get_number_of_elements(),&threads,&grid);
*/
	dim3 threads(8,8,8);
	dim3 grid((extent.width+threads.x-1)/threads.x, (extent.height+threads.y-1)/threads.y,(extent.depth+threads.z-1)/threads.z);
	auto vfield_copy = *vfield1;

	vector_td<int, 3> image_dims(image->get_size(0),image->get_size(1), image->get_size(2));



	bilateral_kernel1D<0><<<grid,threads>>>(vfield1->get_data_ptr(), vfield_copy.get_data_ptr(), image->get_data_ptr(), image_dims, sigma_spatial[0], sigma_int, sigma_diff);
	vfield_copy = *vfield1;
	bilateral_kernel1D<1><<<grid,threads>>>(vfield1->get_data_ptr(), vfield_copy.get_data_ptr(), image->get_data_ptr(), image_dims, sigma_spatial[1], sigma_int, sigma_diff);
	vfield_copy = *vfield1;
	bilateral_kernel1D<2><<<grid,threads>>>(vfield1->get_data_ptr(), vfield_copy.get_data_ptr(), image->get_data_ptr(), image_dims, sigma_spatial[2], sigma_int, sigma_diff);
	/*
	cudaDeviceSynchronize();
	bilateral_kernel1D<1><<<grid,threads>>>(vfield_copy.get_data_ptr(),vfield1->get_data_ptr(), image->get_data_ptr(), image_dims, sigma_spatial[1], sigma_int, sigma_diff);
	cudaDeviceSynchronize();
	bilateral_kernel1D<2><<<grid,threads>>>(vfield1->get_data_ptr(),vfield_copy.get_data_ptr(), image->get_data_ptr(), image_dims, sigma_spatial[2], sigma_int, sigma_diff);
	cudaDeviceSynchronize();
*/








}



template class  cuDemonsSolver<float, 3>;


