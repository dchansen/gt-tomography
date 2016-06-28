#include <cuPartialDerivativeOperator.h>
#include <boost/make_shared.hpp>
#include <cuNDArray_fileio.h>
#include "cuDemonsSolver.h"
#include "setup_grid.h"
#include "gpureg_export.h"
#include "vector_td_utilities.h"
#include <thrust/extrema.h>
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

	int n = ceil(log2(sqrt(*msquare)/0.5));
	n = std::max(n,0);
	std::cout << " N " << n << std::endl;
	*vfield *= float(std::pow(2,-float(n)));

	for (int i =0; i < n; i++) {
		cuNDArray<float> vfield_copy(*vfield);
//deform_vfield(&vfield_copy, vfield);
//		*vfield += vfield_copy;
		deform_vfield(vfield,&vfield_copy);
	}



}

template< class T, unsigned int D> static inline  __device__ void partialDerivs(const T* in, const vector_td<int,D>& dims, vector_td<int,D>& co, T * out)
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

template<class T, unsigned int D> static __global__ void demons_kernel(T* fixed, T* moving,  size_t tot_elements,const vector_td<int,D> dims, T* out,T alpha, T beta){

	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;


	if (idx <  tot_elements){

		size_t elements= prod(dims);

		int batch = idx/elements;
		T * fixed_batch = fixed+elements*batch;
		T * moving_batch = moving+elements*batch;


		T dmov[D];
		T dfix[D];

		vector_td<int,D> co = idx_to_co<D>(idx, dims);

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

        partialDerivs(image,dims,co,&diff[0]);

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
			std::cout << "Updated norm " << nrm2(update.get()) << " " << nrm2(&def_moving) << " " << nrm2(fixed_image) << std::endl;
        }
		if (sigmaFluid > 0){
			cuNDArray<T> blurred_update(update->get_dimensions());
			gaussFluid.mult_M(update.get(),&blurred_update);
			//blurred_update = *update;
			std::cout << "Update step: " << nrm2(&blurred_update) << std::endl;

			if (exponential) vfield_exponential(&blurred_update);
			if (compositive) deform_vfield(result.get(),&blurred_update);
			*result += blurred_update;




		} else {
			std::cout << "Update step: " << nrm2(update.get()) << std::endl;
			if (exponential) vfield_exponential(update.get());
			if (compositive) deform_vfield(result.get(),update.get());
			*result += *update;
		}

		if (sigmaDiff > 0){
			if (sigmaInt >0 || sigmaVDiff > 0)
				bilateral_vfield(result.get(),&def_moving,sigmaDiff,sigmaInt,sigmaVDiff);
			else {
				cuNDArray<T> blurred_result(*result);

				gaussDiff.mult_M(&blurred_result, result.get());
			}
		}




		def_moving = deform_image(texObj,image_dims,result.get());

	}



	cudaDestroyTextureObject(texObj);

	// Free device memory
	cudaFreeArray(image_array);


	return result;



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


static __global__ void jacobian_kernel(float* __restrict__ image,  const vector_td<int,3> dims, float * __restrict__ out){

	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	int elements= prod(dims);

	if (idx <  elements){

		float dX[3];
		float dY[3];
		float dZ[3];


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
bilateral_kernel3D(float* out,cudaTextureObject_t texX,cudaTextureObject_t texY,cudaTextureObject_t texZ,cudaTextureObject_t image, int width, int height, int depth,float sigma_spatial,float sigma_int,float sigma_diff){

	const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
	const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
	const int izo = blockDim.z * blockIdx.z + threadIdx.z;

	const float x = ixo+0.5f;
	const float y = iyo+0.5f;
	const float z = izo+0.5f;


	if (ixo < width && iyo < height && izo < depth){


		int steps = 3;

		float vec[3];
		vec[0] = tex3D<float>(texX,x,y,z);
		vec[1] = tex3D<float>(texY,x,y,z);
		vec[2] = tex3D<float>(texZ,x,y,z);
		float image_value = tex3D<float>(image,x,y,z);
		float res[3];
		res[0] = 0;
		res[1] = 0;
		res[2] = 0;

		float norm = 0;
		for (int dz = -steps; dz <= steps; dz++) {
			const float z2 = z+dz;
			for (int dy = -steps; dy <= steps; dy++) {
				const float y2 = y+dy;
				for (int dx = -steps; dx <= steps; dx++) {

					const float x2 = x+dx;



					float vec2[3];
					vec2[0] = tex3D<float>(texX,x2,y2,z2);
					vec2[1] = tex3D<float>(texY,x2,y2,z2);
					vec2[2] = tex3D<float>(texZ,x2,y2,z2);
					float image_diff = image_value-tex3D<float>(image,x2,y2,z2);
					float vdiff = (vec2[0]-vec[0])*(vec2[0]-vec[0])+(vec2[1]-vec[1])*(vec2[1]-vec[1])+(vec2[2]-vec[2])*(vec2[2]-vec[2]);
					float weight = expf(-float(dx*dx+dy*dy+dz*dz) / (2 * sigma_spatial * sigma_spatial)
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




void Gadgetron::bilateral_vfield(cuNDArray<float>* vfield1, cuNDArray<float>* image, float sigma_spatial,float sigma_int, float sigma_diff){




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
	cudaArray *image_array;
	cudaMalloc3DArray(&x_array, &channelDesc, extent);
	cudaMalloc3DArray(&y_array, &channelDesc, extent);
	cudaMalloc3DArray(&z_array, &channelDesc, extent);
	cudaMalloc3DArray(&image_array, &channelDesc, extent);


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

	cpy_params.dstArray = image_array;
	cpy_params.srcPtr = make_cudaPitchedPtr
			((void*)image->get_data_ptr(), extent.width*sizeof(float), extent.width, extent.height);
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


	struct cudaResourceDesc resDescImg;
	memset(&resDescImg, 0, sizeof(resDescImg));
	resDescImg.resType = cudaResourceTypeArray;
	resDescImg.res.array.array = image_array;

	cudaTextureObject_t texImage = 0;
	cudaCreateTextureObject(&texImage, &resDescImg, &texDesc, NULL);


	cudaDeviceSynchronize();
	dim3 threads(8,8,8);

	dim3 grid((extent.width+threads.x-1)/threads.x, (extent.height+threads.y-1)/threads.y,(extent.depth+threads.z-1)/threads.z);

	bilateral_kernel3D<<<grid,threads>>>(vfield1->get_data_ptr(),texX,texY,texZ,texImage,extent.width,extent.height,extent.depth,sigma_spatial,sigma_int,sigma_diff);

	cudaDeviceSynchronize();
	cudaDestroyTextureObject(texX);
	cudaDestroyTextureObject(texY);
	cudaDestroyTextureObject(texZ);
	cudaDestroyTextureObject(texImage);

	// Free device memory
	cudaFreeArray(x_array);
	cudaFreeArray(y_array);
	cudaFreeArray(z_array);
	cudaFreeArray(image_array);



}


template class  cuDemonsSolver<float, 1>;
template class  cuDemonsSolver<float, 2>;
template class  cuDemonsSolver<float, 3>;


