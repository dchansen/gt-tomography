/*
 * hoCuParallelProjection.cu
 *
 *  Created on: May 28, 2014
 *      Author: dch
 */

#include "hoCuParallelProjection.h"
#include "float3x3.h"
#include "vector_td.h"
#include "vector_td_operators.h"
#include "setup_grid.h"
#include <cuda_runtime_api.h>
#include <math_constants.h>

static texture<float, 3, cudaReadModeElementType>
projection_tex( 1, cudaFilterModeLinear, cudaAddressModeBorder );

using namespace Gadgetron;

static __device__ float sinc(float x ){
	if (x == 0)
		return 1;

	float x2 = CUDART_PI_F*x;
	return sin(x2)/x2;
}


static __device__ void interpolate_along(float & val, float & weight, const intd3 co, int dim, const intd3 image_size, const float * __restrict__ image){
	const int width =1;
	//const float kernel[9] = {9.19870514e-01,   7.13804116e-01, 4.62716498e-01,   2.45964865e-01,   1.03825822e-01, 3.28855260e-02,   6.99601802e-03,   7.58708130e-04,   7.72686684e-06};
	//const float kernel[5] = { 7.61509399e-01,   3.24874218e-01,   6.81537432e-02,  4.80567914e-03,   7.72686684e-06};
	//const float kernel[5] = {7.02354993e-01,   2.32592792e-01,   3.06056304e-02,       9.68690088e-04,   1.60812750e-07};
	//const float kernel[2] = {1.26138724e-01,   1.11926154e-06};
	//const float kernel[5] = {1.33964726e-01,   2.47331088e-04,   2.30516831e-09,
	//         5.48923649e-18,   9.31314002e-43};
	//const float kernel[5] = {6.09699226e-01,   1.29631779e-01,   7.54313367e-03, 5.88182195e-05,   1.73173349e-10};
	//const float kernel[1] = {1};


	//const float kernel[9] = {0.97273069,  0.89448611,  0.7753221 ,  0.63001629,  0.47552746,
	//        0.32820196,  0.20127873,  0.10315762,  0.03671089};
	intd3 co2 = co;
	for (int i = 1; i <= width; i++){
		co2[dim]=co[dim]-i;
		float img = image[co_to_idx((co2+image_size)%image_size,image_size)];
		if (img > 0){
			//float w = sinc(float(i));
			float w =  1;
			//float w = kernel[i-1];
			val += img*w;
			weight += w;
		}
	}



	for (int i = 1; i <= width; i++){
		co2[dim]=co[dim]+i;
		float img = image[co_to_idx((co2+image_size)%image_size,image_size)];
		if (img > 0){

			//float w = sinc(float(i));

			//float w = kernel[i-1];
			float w = 1;
			val += img*w;
			weight += w;
		}
	}

}
static __device__ void add_to_interpolation(float & val, int & n, intd3& co, const intd3 image_size, const float * __restrict__ image){
	float w = image[co_to_idx((co+image_size)%image_size,image_size)];
	if (w > 0){
		val += w;
		n++;
	}
}
static __global__ void interpolate_projection(float* __restrict__ image_out, const float * __restrict__ image, const intd3 image_size){

	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
	const int num_elements = prod(image_size);

	if (idx < num_elements){

		if (image[idx] <= 0){


			intd3 co = idx_to_co<3>(idx,image_size);
/*
			const int steps = 20;
			int start = (co[0]-steps < 0) ? 0 : co[0]-steps;
			int end = (co[0]+steps <= image_size[0]) ? co[0]+steps : image_size[0];

			float mask_val = 0;
			float val = 0;
			float tot = 0;
			for (int i = start; i < end; i++){
				float weight = sinc(float(i));
				tot += weight;
				float loc_mask = mask[idx+i];
				if (loc_mask > 0){
					mask_val += weight*loc_mask;
					val += weight*image[idx+i];
				}
			}
			image[idx] = val;
			out_mask[idx] = mask_val;
*/
/*

			float val = 0;
			int n = 0;
			co[0]+=1;
			add_to_interpolation(val,n,co,image_size,image);
			co[0] -= 2;
			add_to_interpolation(val,n,co,image_size,image);
			co[0] += 1;
			co[1] += 1;
			add_to_interpolation(val,n,co,image_size,image);
			co[1] -= 2;
			add_to_interpolation(val,n,co,image_size,image);
			if (n > 0)
				image_out[idx] = val/n;
*/
/*
			if (co[0] == 0){
				image_out[idx] = image[idx+1];
			} else if (co[0] == (image_size[0]-1)){
				image_out[idx] = image[idx-1];
			} else {
			image_out[idx] = (image[idx-1]+image[idx+1])/2;
			}
*/


			float val = 0;

			float w = 0;
			interpolate_along(val,w,co,0,image_size,image);
			interpolate_along(val,w,co,1,image_size,image);
			if (w > 0)
				image_out[idx] = val/w;

		}
	}
}


void Gadgetron::interpolate_missing( cuNDArray<float>* image){
	dim3 dimBlock, dimGrid;
	setup_grid( image->get_number_of_elements(), &dimBlock, &dimGrid );

	cuNDArray<float> image_copy(image);
	std::cout << "Interpolating missing data" << std::endl;
	interpolate_projection<<<dimGrid,dimBlock>>>(image->get_data_ptr(),image_copy.get_data_ptr(),intd3(from_std_vector<size_t,3>(*image->get_dimensions())));

}

static __global__ void parallel_backprojection_kernel(float * __restrict__ image, floatd3 image_dims, intd3 image_size, floatd3 projection_dims, float angle){

	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
	const int num_elements = prod(image_size);

	if (idx < num_elements){
		//const float pixel_width = image_dims[0]/image_size[0];
		const intd3 co = idx_to_co<3>(idx, image_size);


		floatd3 pos = co*image_dims/image_size-image_dims/2;

		const float3x3 inverseRotation = calcRotationMatrixAroundZ(-angle);

		// Rotated image coordinate (local to the projection's coordinate system)
		//
		const floatd3 pos_proj_norm = mul(inverseRotation, pos)/projection_dims+0.5f;
		//image[idx] += tex3D(projection_tex,pos_proj_norm[0],pos_proj_norm[1],pos_proj_norm[2])*pixel_width;
		image[idx] += tex3D(projection_tex,pos_proj_norm[0],pos_proj_norm[1],pos_proj_norm[2]);
	}
}



void Gadgetron::parallel_backprojection(cuNDArray<float>* projection, cuNDArray<float>* image, float  angle, floatd3 image_dims, floatd3 projection_dims){

	//cudaFuncSetCacheConfig(parallel_backprojection_kernel, cudaFuncCachePreferL1);
	std::vector<size_t> proj_dims = *projection->get_dimensions();

	size_t proj_elements = projection->get_number_of_elements();




	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = proj_dims[0];
	extent.height = proj_dims[1];
	extent.depth = proj_dims[2];

	cudaArray *projection_array;
	cudaMalloc3DArray( &projection_array, &channelDesc, extent );
	CHECK_FOR_CUDA_ERROR();

	cudaMemcpy3DParms cpy_params = {0};
	cpy_params.extent = extent;
	cpy_params.dstArray = projection_array;
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.srcPtr =
			make_cudaPitchedPtr( (void*)projection->get_data_ptr(), proj_dims[0]*sizeof(float),
					proj_dims[0],proj_dims[1] );
	cudaMemcpy3D( &cpy_params);
	cudaBindTextureToArray( projection_tex, projection_array, channelDesc );
	dim3 dimBlock, dimGrid;
	setup_grid( image->get_number_of_elements(), &dimBlock, &dimGrid );

	parallel_backprojection_kernel<<<dimGrid,dimBlock>>>(image->get_data_ptr(),image_dims,intd3(from_std_vector<size_t,3>(*image->get_dimensions())),projection_dims, angle);

	cudaUnbindTexture(projection_tex);
	cudaFreeArray(projection_array);

}



void Gadgetron::parallel_backprojection(hoCuNDArray<float>* projection, cuNDArray<float>* image, float  angle, floatd3 image_dims, floatd3 projection_dims){

	//cudaFuncSetCacheConfig(parallel_backprojection_kernel, cudaFuncCachePreferL1);
	std::vector<size_t> proj_dims = *projection->get_dimensions();

	size_t proj_elements = projection->get_number_of_elements();




	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = proj_dims[0];
	extent.height = proj_dims[1];
	extent.depth = proj_dims[2];

	cudaArray *projection_array;
	cudaMalloc3DArray( &projection_array, &channelDesc, extent );
	CHECK_FOR_CUDA_ERROR();

	cudaMemcpy3DParms cpy_params = {0};
	cpy_params.extent = extent;
	cpy_params.dstArray = projection_array;
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.srcPtr =
			make_cudaPitchedPtr( (void*)projection->get_data_ptr(), proj_dims[0]*sizeof(float),
					proj_dims[0],proj_dims[1] );
	cudaMemcpy3D( &cpy_params);
	cudaBindTextureToArray( projection_tex, projection_array, channelDesc );
	dim3 dimBlock, dimGrid;
	setup_grid( image->get_number_of_elements(), &dimBlock, &dimGrid );

	parallel_backprojection_kernel<<<dimGrid,dimBlock>>>(image->get_data_ptr(),image_dims,intd3(from_std_vector<size_t,3>(*image->get_dimensions())),projection_dims, angle);

	cudaUnbindTexture(projection_tex);
	cudaFreeArray(projection_array);

}

