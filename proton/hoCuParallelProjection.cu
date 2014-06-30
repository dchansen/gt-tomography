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


static texture<float, 3, cudaReadModeElementType>
projection_tex( 1, cudaFilterModeLinear, cudaAddressModeBorder );

using namespace Gadgetron;


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

