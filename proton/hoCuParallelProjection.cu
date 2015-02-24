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

static __global__ void interpolate_projection(float * __restrict__ image, const float * __restrict__ mask, float * __restrict__ out_mask, intd3 image_size){

	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
	const int num_elements = prod(image_size);

	if (idx < num_elements){

		if (mask[idx] <= 0){

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

			if (co[0] == 0){
				image[idx] = image[idx+1];
				out_mask[idx] = mask[idx+1];

			} else if (co[0] == (image_size[0]-1)){
				image[idx] = image[idx-1];
				out_mask[idx] = mask[idx-1];
			} else {


			image[idx] = (image[idx-1]+image[idx+1])/2;
			out_mask[idx] = (mask[idx-1]+mask[idx+1])/2;
			}
		}
	}
}


void Gadgetron::interpolate_missing( cuNDArray<float>* image, cuNDArray<float>* mask){
	dim3 dimBlock, dimGrid;
	setup_grid( image->get_number_of_elements(), &dimBlock, &dimGrid );

	cuNDArray<float> mask_copy(*mask);
std::cout << "Interpolating missing data" << std::endl;
	interpolate_projection<<<dimGrid,dimBlock>>>(image->get_data_ptr(),mask->get_data_ptr(),mask_copy.get_data_ptr(), intd3(from_std_vector<size_t,3>(*image->get_dimensions())));
	*mask = mask_copy;

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

