//
// This code performs 3D cone beam CT forwards and backwards projection
//

#include "conebeam_projection.h"
#include "float3x3.h"
#include "hoCuNDArray_elemwise.h"
#include "vector_td.h"
#include "cuNDArray_math.h"
#include "cuNDArray_utils.h"
#include "cuNFFT.h"
#include "check_CUDA.h"
#include "GPUTimer.h"
#include "cudaDeviceManager.h"
#include "hoNDArray_fileio.h"

#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <cufft.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#define PS_ORIGIN_CENTERING
#define IS_ORIGIN_CENTERING
//#define FLIP_Z_AXIS

// Read the projection/image data respectively as a texture (for input)
// - taking advantage of the cache and hardware interpolation
//

#define NORMALIZED_TC 1

static texture<float, 3, cudaReadModeElementType>
image_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeBorder );

static texture<float, cudaTextureType2DLayered, cudaReadModeElementType>
projections_tex( NORMALIZED_TC, cudaFilterModeLinear, cudaAddressModeBorder );

static texture<float, cudaTextureType2DLayered, cudaReadModeElementType>
projections_mask_tex( NORMALIZED_TC, cudaFilterModePoint, cudaAddressModeBorder );

namespace Gadgetron
{

static inline
  void setup_grid( unsigned int number_of_elements, dim3 *blockDim, dim3* gridDim, unsigned int num_batches = 1 )
  {
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
    //int maxGridDim = cudaDeviceManager::Instance()->max_griddim(cur_device);
    int maxBlockDim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    int maxGridDim = 65535;

    // The default one-dimensional block dimension is...
    *blockDim = dim3(256);
    *gridDim = dim3((number_of_elements+blockDim->x-1)/blockDim->x, num_batches);

    // Extend block/grid dimensions if we exceeded the maximum grid dimension
    if( gridDim->x > maxGridDim){
      blockDim->x = maxBlockDim;
      gridDim->x = (number_of_elements+blockDim->x-1)/blockDim->x;
    }

    if( gridDim->x > maxGridDim ){
      gridDim->x = (unsigned int)std::floor(std::sqrt(float(number_of_elements)/float(blockDim->x)));
      unsigned int num_elements_1d = blockDim->x*gridDim->x;
      gridDim->y *= ((number_of_elements+num_elements_1d-1)/num_elements_1d);
    }

    if( gridDim->x > maxGridDim || gridDim->y > maxGridDim){
      // If this ever becomes an issue, there is an additional grid dimension to explore for compute models >= 2.0.
      throw cuda_error("setup_grid(): too many elements requested.");
    }
  }
// Utility to convert from degrees to radians
//

static inline __host__ __device__
float degrees2radians(float degree) {
	return degree * (CUDART_PI_F/180.0f);
}


static inline __device__
floatd3 calculate_endpoint(const floatd3 & det_focal_cyl, const float & y_offset const intd2 & elements, const floatd2 & spacing,const floatd2 & central_element, const int3d & co){

        float phi = spacing[0]*elements[0]*(co[0]-central_element[0])/(det_focal_cyl[1]+y_offset);
        float x = -rho*sin(phi);
        float y = rho*cos(phi)-y_offset;
        float z = spacing[1]*elements[1]*(co[1]-central_element[1])+det_focal_cyl[2];



        return floatd3(cos(det_focal_cyl[0])*x+sin(det_focal_cyl[0])*y,-sin(det_focal_cyl[0])*x+cos(det_focal_cyl[0])*y,z);
    }


//
// Forwards projection
//

__global__ void
ct_forwards_projection_kernel( float * __restrict__ projections,
		const floatd3 * __restrict__ detector_focal_cyls,
		const floatd3 * __restrict__ focal_offset_cyls,
                               const floatd2 * __restrict__ centralElements,
		floatd3 is_dims_in_pixels,
		floatd3 is_dims_in_mm,
		intd2 ps_dims_in_pixels_int,
                               floatd2 ps_spacing,
		int num_projections,
		float SAD,
		int num_samples_per_ray, bool accumulate )
{
	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
	const int num_elements = prod(ps_dims_in_pixels_int)*num_projections;

	if( idx < num_elements){

		const intd3 co = idx_to_co<3>( idx, intd3(ps_dims_in_pixels_int[0], ps_dims_in_pixels_int[1], num_projections) );

		// Projection space dimensions and spacing
		//


		// Determine projection angle and rotation matrix
		//

        floatd3 detector_focal_cyl = detector_focal_cyls[co[2]];
        floatd3 focal_cyl = detector_focal_cyl + focal_offset_cyls[co[2]];
        const float centralElement = centralElements[co[2]];
		// Find start and end point for the line integral (image space)
		//
		floatd3 startPoint = floatd3(-focal_cyl[1]*sin(focal_cyl[0]),focal_cyl[1]*cos(focal_cyl[0]),focal_cyl[2]);


		// Projection plate indices
		//


    	// Find direction vector of the line integral
		//
        floatd3 endPoint = calculate_endpoint(detector_focal_cyl,SAD,ps_dims_in_pixels_int,ps_spacing,centralElement,co);

		floatd3 dir = endPoint-startPoint;

		// Perform integration only inside the bounding cylinder of the image volume
		//

		const floatd3 vec_over_dir = (is_dims_in_mm-startPoint)/dir;
		const floatd3 vecdiff_over_dir = (-is_dims_in_mm-startPoint)/dir;
		const floatd3 start = amin(vecdiff_over_dir, vec_over_dir);
		const floatd3 end   = amax(vecdiff_over_dir, vec_over_dir);

		float a1 = fmax(max(start),0.0f);
		float aend = fmin(min(end),1.0f);
		startPoint += a1*dir;

		const float sampling_distance = norm((aend-a1)*dir)/num_samples_per_ray;

		// Now perform conversion of the line integral start/end into voxel coordinates
		//

		startPoint /= is_dims_in_mm;

		startPoint += 0.5f;
		dir /= is_dims_in_mm;
		dir /= float(num_samples_per_ray); // now in step size units

		//
		// Perform line integration
		//

		float result = 0.0f;

		for ( int sampleIndex = 0; sampleIndex<num_samples_per_ray; sampleIndex++) {


			floatd3 samplePoint = startPoint+dir*float(sampleIndex);

			// Accumulate result
			//

			result += tex3D( image_tex, samplePoint[0], samplePoint[1], samplePoint[2] );
		}

		// Output (normalized to the length of the ray)
		//
		if (accumulate)
			projections[idx] += result*sampling_distance;
		else
			projections[idx] = result*sampling_distance;
	}
}

//
// Forwards projection of a 3D volume onto a set of (binned) projections
//

void
ct_forwards_projection( hoCuNDArray<float> *projections,
		hoCuNDArray<float> *image,
		std::vector<float> angles,
		std::vector<floatd2> offsets,
		std::vector<unsigned int> indices,
		int projections_per_batch,
		float samples_per_pixel,
		floatd3 is_dims_in_mm,
		floatd2 ps_dims_in_mm,
		float SDD,
		float SAD)
{

	//
	// Validate the input
	//

	if( projections == 0x0 || image == 0x0 ){
		throw std::runtime_error("Error: conebeam_forwards_projection: illegal array pointer provided");
	}

	if( projections->get_number_of_dimensions() != 3 ){
		throw std::runtime_error("Error: conebeam_forwards_projection: projections array must be three-dimensional");
	}

	if( image->get_number_of_dimensions() != 3 ){
		throw std::runtime_error("Error: conebeam_forwards_projection: image array must be three-dimensional");
	}

	if( projections->get_size(2) != angles.size() || projections->get_size(2) != offsets.size() ) {
		throw std::runtime_error("Error: conebeam_forwards_projection: inconsistent sizes of input arrays/vectors");
	}

	if (indices.size() == 0){
		return;
	}

	int projection_res_x = projections->get_size(0);
	int projection_res_y = projections->get_size(1);

	int num_projections_in_bin = indices.size();
	int num_projections_in_all_bins = projections->get_size(2);

	int matrix_size_x = image->get_size(0);
	int matrix_size_y = image->get_size(1);
	int matrix_size_z = image->get_size(2);

	hoCuNDArray<float> *int_projections = projections;
	if( projections_per_batch > num_projections_in_bin )
		projections_per_batch = num_projections_in_bin;

	int num_batches = (num_projections_in_bin+projections_per_batch-1) / projections_per_batch;

	// Build texture from input image
	//

	cudaFuncSetCacheConfig(conebeam_forwards_projection_kernel, cudaFuncCachePreferL1);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = matrix_size_x;
	extent.height = matrix_size_y;
	extent.depth = matrix_size_z;

	cudaMemcpy3DParms cpy_params = {0};
	cpy_params.kind = cudaMemcpyHostToDevice;
	cpy_params.extent = extent;

	cudaArray *image_array;
	cudaMalloc3DArray(&image_array, &channelDesc, extent);
	CHECK_FOR_CUDA_ERROR();

	cpy_params.dstArray = image_array;
	cpy_params.srcPtr = make_cudaPitchedPtr
			((void*)image->get_data_ptr(), extent.width*sizeof(float), extent.width, extent.height);
	cudaMemcpy3D(&cpy_params);
	CHECK_FOR_CUDA_ERROR();

	cudaBindTextureToArray(image_tex, image_array, channelDesc);
	CHECK_FOR_CUDA_ERROR();

	// Allocate the angles, offsets and projections in device memory
	//

	float *projections_DevPtr, *projections_DevPtr2;
	cudaMalloc( (void**) &projections_DevPtr, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));
	cudaMalloc( (void**) &projections_DevPtr2, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));

	cudaStream_t mainStream, indyStream;
	cudaStreamCreate(&mainStream);
	cudaStreamCreate(&indyStream);

	std::vector<float> angles_vec;
	std::vector<floatd2> offsets_vec;

	for( int p=0; p<indices.size(); p++ ){

		int from_id = indices[p];

		if( from_id >= num_projections_in_all_bins ) {
			throw std::runtime_error("Error: conebeam_forwards_projection: illegal index in bin");
		}

		angles_vec.push_back(angles[from_id]);
		offsets_vec.push_back(offsets[from_id]);
	}

	thrust::device_vector<float> angles_devVec(angles_vec);
	thrust::device_vector<floatd2> offsets_devVec(offsets_vec);

	//
	// Iterate over the batches
	//

	for (unsigned int batch=0; batch<num_batches; batch++ ){

		int from_projection = batch * projections_per_batch;
		int to_projection = (batch+1) * projections_per_batch;

		if (to_projection > num_projections_in_bin)
			to_projection = num_projections_in_bin;

		int projections_in_batch = to_projection-from_projection;

		// Block/grid configuration
		//

		dim3 dimBlock, dimGrid;
		setup_grid( projection_res_x*projection_res_y*projections_in_batch, &dimBlock, &dimGrid );

		// Launch kernel
		//

		floatd3 is_dims_in_pixels(matrix_size_x, matrix_size_y, matrix_size_z);
		intd2 ps_dims_in_pixels(projection_res_x, projection_res_y);

		float* raw_angles = thrust::raw_pointer_cast(&angles_devVec[from_projection]);
		floatd2* raw_offsets = thrust::raw_pointer_cast(&offsets_devVec[from_projection]);

		conebeam_forwards_projection_kernel<<< dimGrid, dimBlock, 0, mainStream >>>
				( projections_DevPtr, raw_angles, raw_offsets,
						is_dims_in_pixels, is_dims_in_mm, ps_dims_in_pixels, ps_dims_in_mm,
						projections_in_batch, SDD, SAD, samples_per_pixel*float(matrix_size_x), false );

		// If not initial batch, start copying the old stuff
		//

		int p = from_projection;
		while( p<to_projection) {

			int num_sequential_projections = 1;
			while( p+num_sequential_projections < to_projection &&
					indices[p+num_sequential_projections]==(indices[p+num_sequential_projections-1]+1) ){
				num_sequential_projections++;
			}

			int to_id = indices[p];
			int size = projection_res_x*projection_res_y;

			cudaMemcpyAsync( int_projections->get_data_ptr()+to_id*size,
					projections_DevPtr+(p-from_projection)*size,
					size*num_sequential_projections*sizeof(float),
					cudaMemcpyDeviceToHost, mainStream);

			p += num_sequential_projections;
		}

		std::swap(projections_DevPtr, projections_DevPtr2);
		std::swap(mainStream, indyStream);
	}

	cudaFree(projections_DevPtr2);
	cudaFree(projections_DevPtr);
	cudaFreeArray(image_array);

	CUDA_CALL(cudaStreamDestroy(indyStream));
	CUDA_CALL(cudaStreamDestroy(mainStream));
	CHECK_FOR_CUDA_ERROR();

}

/**
 *
 * Assumption - projections are sorted by z. Each slice
 **/
 __global__ void
ct_backwards_projection_kernel( float * __restrict__ image, // Image of size [nx,ny,nz]
       	const floatd3 * __restrict__ detector_focal_cyls, //phi,rho,z of the detector focal spot in units of rad, mm and mm
		const floatd3 * __restrict__ focal_offset_cyls, // phi,rho,z offset of the source focal spot compared to detector focal spot
        const floatd2 * __restrict__ centralElements, // Central element on the detector
        const intd2 * __restrict__ proj_indices, // Array of size nz containing first to last projection for slice
		intd3 is_dims_in_pixels_int, //nx, ny, nz
		floatd3 is_dims_in_mm, // Image size in mm
		floatd2 ps_dims_in_pixels, //Projection size
		floatd2 ps_spacing,  //Size of each projection element in mm
		float SDD, //focalpoint - detector disance
        int offset,
		bool accumulate )
{
	// Image voxel to backproject into (pixel coordinate and index)
	//

	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
	const int num_elements = prod(is_dims_in_pixels_int);

	if( idx < num_elements ){

		const intd3 co = idx_to_co<3>(idx, is_dims_in_pixels_int);

		const floatd3 is_pc = floatd3(co) + floatd3(0.5);


		// Normalized image space coordinate [-0.5, 0.5[
		//

		const floatd3 is_dims_in_pixels(is_dims_in_pixels_int);


		const floatd3 is_nc = is_pc / is_dims_in_pixels - floatd3(0.5f);

		// Image space coordinate in metric units
		//

		const floatd3 pos = is_nc * is_dims_in_mm;

		// Read the existing output value for accumulation at this point.
		// The cost of this fetch is hidden by the loop

		const float incoming = (accumulate) ? image[idx] : 0.0f;

		// Backprojection loop
		//

		float result = 0.0f;

		for( int projection = proj_indices[co[2]][0]-offset; projection < (proj_indices[co[2]][1]-offset); projection++ ) {

			// Projection angle
			//
            const floatd3 detector_focal_cyl = detector_focal_cyls[projection];
            const floatd3 focal_cyl = focal_offset_cyls[projection]+detector_focal_cyl;

            const floatd3 startPoint = floatd3(-focal_cyl[1]*sin(focal_cyl[0]),focal_cyl[0]*cos(focal_cyl[0]),focal_cyl[2]);
            const floatd3 focal_point = floatd3(-detector_focal_cyl[1]*sin(detector_focal_cyl[0]),detector_focal_cyl[0]*cos(detector_focal_cyl[0]),detector_focal_cyl[2]);
            const floatd3 dir = pos-startPoint;


            const float a = (dir[0]*dir[0]+dir[1]*dir[1]);
            const float b = (pos[0]-focal_point[0])*dir[0]+(pos[1]-focal_point[1])*dir[1];
            const float c= -SDD*SDD+(pos[0]-focal_point[0])*(pos[0]-focal_point[0])+(pos[1]-focal_point[1])*(pos[1]-focal_point[1]);
            float t = (-b+sqrt(b*b-4*a*c))/(2*a);

            const floatd3 detectorPoint = startPoint+dir*t;
            const floatd2 element_rad = floatd2((atan2(detectorPoint[0],detectorPoint[1])-detector_focal_cyl[0])*SDD/ps_spacing[0],
                                                (detectorPoint[2]-detector_focal_cyl[2])/ps_spacing[1])-centralElements[projection]+0.5f;
			// Convert metric projection coordinates into pixel coordinates
			//

			// Read the projection data (bilinear interpolation enabled) and accumulate
			//

			result += tex2DLayered( projections_tex, element_rad[0], element_rad[1], projection );
		}

		// Output normalized image
		//

		image[idx] = incoming + result  ;
	}
}

//
// Backprojection
//

void ct_backwards_projection( cuNDArray<float> *projections,
		cuNDArray<float> *image,
		std::vector<float> angles,
		std::vector<floatd2> offsets,
		intd3 is_dims_in_pixels,
		floatd3 is_dims_in_mm,
		floatd2 ps_dims_in_mm,
		float SDD,
		float SAD,
		bool use_offset_correction,
		bool accumulate
)
{
	//
	// Validate the input
	//

	if( projections == 0x0 || image == 0x0 ){
		throw std::runtime_error("Error: conebeam_backwards_projection: illegal array pointer provided");
	}

	if( projections->get_number_of_dimensions() != 3 ){
		throw std::runtime_error("Error: conebeam_backwards_projection: projections array must be three-dimensional");
	}

	if( image->get_number_of_dimensions() != 3 ){
		throw std::runtime_error("Error: conebeam_backwards_projection: image array must be three-dimensional");
	}

	if( projections->get_size(2) != angles.size() || projections->get_size(2) != offsets.size() ) {
		throw std::runtime_error("Error: conebeam_backwards_projection: inconsistent sizes of input arrays/vectors");
	}

	// Some utility variables
	//

	int matrix_size_x = image->get_size(0);
	int matrix_size_y = image->get_size(1);
	int matrix_size_z = image->get_size(2);

	floatd3 is_dims(matrix_size_x, matrix_size_y, matrix_size_z);

	int projection_res_x = projections->get_size(0);
	int projection_res_y = projections->get_size(1);
	int num_projections = projections->get_size(2);

	floatd2 ps_dims_in_pixels(projection_res_x, projection_res_y);

	// Allocate device memory for the backprojection result
	//
	// Allocate the angles, offsets and projections in device memory
	//

	thrust::device_vector<float> angles_devVec(angles);
	thrust::device_vector<floatd2> offsets_devVec(offsets);

	std::vector<size_t> dims;
	dims.push_back(projection_res_x);
	dims.push_back(projection_res_y);
	dims.push_back(num_projections);


	//
	// Iterate over batches
	//

	float* raw_angles = thrust::raw_pointer_cast(&angles_devVec[0]);
	floatd2* raw_offsets = thrust::raw_pointer_cast(&offsets_devVec[0]);


	if (use_offset_correction)
		offset_correct_sqrt( projections, raw_offsets, ps_dims_in_mm, SAD, SDD );

	// Build array for input texture
	//

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent extent;
	extent.width = projection_res_x;
	extent.height = projection_res_y;
	extent.depth = num_projections;

	cudaArray *projections_array;
	cudaMalloc3DArray( &projections_array, &channelDesc, extent, cudaArrayLayered );
	CHECK_FOR_CUDA_ERROR();

	cudaMemcpy3DParms cpy_params = {0};
	cpy_params.extent = extent;
	cpy_params.dstArray = projections_array;
	cpy_params.kind = cudaMemcpyDeviceToDevice;
	cpy_params.srcPtr =
			make_cudaPitchedPtr( (void*)projections->get_data_ptr(), projection_res_x*sizeof(float),
					projection_res_x, projection_res_y );
	cudaMemcpy3D( &cpy_params );
	CHECK_FOR_CUDA_ERROR();

	cudaBindTextureToArray( projections_tex, projections_array, channelDesc );
	CHECK_FOR_CUDA_ERROR();

	// Upload projections for the next batch
	// - to enable streaming
	//
	// Define dimensions of grid/blocks.
	//

	dim3 dimBlock, dimGrid;
	setup_grid( matrix_size_x*matrix_size_y*matrix_size_z, &dimBlock, &dimGrid );

	// Invoke kernel
	//

	cudaFuncSetCacheConfig(conebeam_backwards_projection_kernel<false>, cudaFuncCachePreferL1);

	conebeam_backwards_projection_kernel<false	><<< dimGrid, dimBlock >>>
			( image->get_data_ptr(), raw_angles, raw_offsets,
					is_dims_in_pixels, is_dims_in_mm, ps_dims_in_pixels, ps_dims_in_mm,
					num_projections,  SDD, SAD, accumulate );

	CHECK_FOR_CUDA_ERROR();

	// Cleanup
	//

	cudaUnbindTexture(projections_tex);
	cudaFreeArray(projections_array);
	CHECK_FOR_CUDA_ERROR();
}
template <bool FBP>
void conebeam_backwards_projection( hoCuNDArray<float> *projections,
		hoCuNDArray<float> *image,
		std::vector<float> angles,
		std::vector<floatd2> offsets,
		std::vector<unsigned int> indices,
		int projections_per_batch,
		intd3 is_dims_in_pixels,
		floatd3 is_dims_in_mm,
		floatd2 ps_dims_in_mm,
		float SDD,
		float SAD,
		bool short_scan,
		bool use_offset_correction,
		bool accumulate,
		cuNDArray<float> *cosine_weights,
		cuNDArray<float> *frequency_filter
)
{
	//
	// Validate the input
	//

	if( projections == 0x0 || image == 0x0 ){
		throw std::runtime_error("Error: conebeam_backwards_projection: illegal array pointer provided");
	}

	if( projections->get_number_of_dimensions() != 3 ){
		throw std::runtime_error("Error: conebeam_backwards_projection: projections array must be three-dimensional");
	}

	if( image->get_number_of_dimensions() != 3 ){
		throw std::runtime_error("Error: conebeam_backwards_projection: image array must be three-dimensional");
	}

	if( projections->get_size(2) != angles.size() || projections->get_size(2) != offsets.size() ) {
		throw std::runtime_error("Error: conebeam_backwards_projection: inconsistent sizes of input arrays/vectors");
	}

	if( FBP && !(cosine_weights && frequency_filter) ){
		throw std::runtime_error("Error: conebeam_backwards_projection: for _filtered_ backprojection both cosine weights and a filter must be provided");
	}

	if (indices.size() == 0){
		if (accumulate)
			return;
		else {
			clear(image);
			return;
		}
	}

	// Some utility variables
	//

	int matrix_size_x = image->get_size(0);
	int matrix_size_y = image->get_size(1);
	int matrix_size_z = image->get_size(2);

	floatd3 is_dims(matrix_size_x, matrix_size_y, matrix_size_z);
	int num_image_elements = matrix_size_x*matrix_size_y*matrix_size_z;

	int projection_res_x = projections->get_size(0);
	int projection_res_y = projections->get_size(1);

	floatd2 ps_dims_in_pixels(projection_res_x, projection_res_y);

	int num_projections_in_all_bins = projections->get_size(2);
	int num_projections_in_bin = indices.size();

	if( projections_per_batch > num_projections_in_bin )
		projections_per_batch = num_projections_in_bin;

	int num_batches = (num_projections_in_bin+projections_per_batch-1) / projections_per_batch;

	// Allocate device memory for the backprojection result
	//

	boost::shared_ptr< cuNDArray<float> > image_device;

	if( accumulate ){
		image_device = boost::shared_ptr< cuNDArray<float> >(new cuNDArray<float>(image));
	}
	else{
		image_device = boost::shared_ptr< cuNDArray<float> >(new cuNDArray<float>(image->get_dimensions().get()));
	}

	// Allocate the angles, offsets and projections in device memory
	//

	float *projections_DevPtr, *projections_DevPtr2;
	cudaMalloc( (void**) &projections_DevPtr, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));
	cudaMalloc( (void**) &projections_DevPtr2, projection_res_x*projection_res_y*projections_per_batch*sizeof(float));

	cudaStream_t mainStream, indyStream;
	cudaStreamCreate(&mainStream);
	cudaStreamCreate(&indyStream);

	std::vector<float> angles_vec;
	std::vector<floatd2> offsets_vec;

	for( int p=0; p<indices.size(); p++ ){

		int from_id = indices[p];

		if( from_id >= num_projections_in_all_bins ) {
			throw std::runtime_error("Error: conebeam_backwards_projection: illegal index in bin");
		}

		angles_vec.push_back(angles[from_id]);
		offsets_vec.push_back(offsets[from_id]);
	}

	thrust::device_vector<float> angles_devVec(angles_vec);
	thrust::device_vector<floatd2> offsets_devVec(offsets_vec);

	// From/to for the first batch
	// - to enable working streams...
	//

	int from_projection = 0;
	int to_projection = projections_per_batch;

	if (to_projection > num_projections_in_bin )
		to_projection = num_projections_in_bin;

	int projections_in_batch = to_projection-from_projection;

	std::vector<size_t> dims;
	dims.push_back(projection_res_x);
	dims.push_back(projection_res_y);
	dims.push_back(projections_in_batch);

	std::vector<size_t> dims_next;

	cuNDArray<float> *projections_batch = new cuNDArray<float>(&dims, projections_DevPtr);

	// Upload first projections batch adhering to the binning.
	// Be sure to copy sequentially numbered projections in one copy operation.
	//

	{
		int p = from_projection;

		while( p<to_projection ) {

			int num_sequential_projections = 1;
			while( p+num_sequential_projections < to_projection &&
					indices[p+num_sequential_projections]==(indices[p+num_sequential_projections-1]+1) ){
				num_sequential_projections++;
			}

			int from_id = indices[p];
			int size = projection_res_x*projection_res_y;

			cudaMemcpyAsync( projections_batch->get_data_ptr()+(p-from_projection)*size,
					projections->get_data_ptr()+from_id*size,
					size*num_sequential_projections*sizeof(float), cudaMemcpyHostToDevice, mainStream );

			CHECK_FOR_CUDA_ERROR();

			p += num_sequential_projections;
		}
	}

	//
	// Iterate over batches
	//

	for( int batch = 0; batch < num_batches; batch++ ) {

		from_projection = batch * projections_per_batch;
		to_projection = (batch+1) * projections_per_batch;

		if (to_projection > num_projections_in_bin )
			to_projection = num_projections_in_bin;

		projections_in_batch = to_projection-from_projection;

		float* raw_angles = thrust::raw_pointer_cast(&angles_devVec[from_projection]);
		floatd2* raw_offsets = thrust::raw_pointer_cast(&offsets_devVec[from_projection]);


		if( FBP ){

			// Apply cosine weighting : "SDD / sqrt(SDD*SDD + u*u + v*v)"
			// - with (u,v) positions given in metric units on a virtual detector at the origin
			//

			*projections_batch *= *cosine_weights;

			// Redundancy correct
			// - for short scan mode
			//

			if( short_scan ){
				float delta = std::atan(ps_dims_in_mm[0]/(2.0f*SDD));
				redundancy_correct( projections_batch, raw_angles, delta );
			}

			// Apply frequency filter
			// - use zero padding to avoid the cyclic boundary conditions induced by the fft
			//

			std::vector<size_t> batch_dims = *projections_batch->get_dimensions();
			uint64d3 pad_dims(batch_dims[0]<<1, batch_dims[1], batch_dims[2]);
			boost::shared_ptr< cuNDArray<float> > padded_projections = pad<float,3>( pad_dims, projections_batch );
			boost::shared_ptr< cuNDArray<complext<float> > > complex_projections = cb_fft( padded_projections.get() );
			*complex_projections *= *frequency_filter;
			cb_ifft( complex_projections.get(), padded_projections.get() );
			uint64d3 crop_offsets(batch_dims[0]>>1, 0, 0);
			crop<float,3>( crop_offsets, padded_projections.get(), projections_batch );

			// Apply offset correction
			// - for half fan mode, sag correction etc.
			//
			if (use_offset_correction)
				offset_correct( projections_batch, raw_offsets, ps_dims_in_mm, SAD, SDD );


		} else if (use_offset_correction)
			offset_correct_sqrt( projections_batch, raw_offsets, ps_dims_in_mm, SAD, SDD );

		// Build array for input texture
		//

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaExtent extent;
		extent.width = projection_res_x;
		extent.height = projection_res_y;
		extent.depth = projections_in_batch;

		cudaArray *projections_array;
		cudaMalloc3DArray( &projections_array, &channelDesc, extent, cudaArrayLayered );
		CHECK_FOR_CUDA_ERROR();

		cudaMemcpy3DParms cpy_params = {0};
		cpy_params.extent = extent;
		cpy_params.dstArray = projections_array;
		cpy_params.kind = cudaMemcpyDeviceToDevice;
		cpy_params.srcPtr =
				make_cudaPitchedPtr( (void*)projections_batch->get_data_ptr(), projection_res_x*sizeof(float),
						projection_res_x, projection_res_y );
		cudaMemcpy3DAsync( &cpy_params, mainStream );
		CHECK_FOR_CUDA_ERROR();

		cudaBindTextureToArray( projections_tex, projections_array, channelDesc );
		CHECK_FOR_CUDA_ERROR();

		// Upload projections for the next batch
		// - to enable streaming
		//

		if( batch < num_batches-1 ){ // for using multiple streams to hide the cost of the uploads

			int from_projection_next = (batch+1) * projections_per_batch;
			int to_projection_next = (batch+2) * projections_per_batch;

			if (to_projection_next > num_projections_in_bin )
				to_projection_next = num_projections_in_bin;

			int projections_in_batch_next = to_projection_next-from_projection_next;

			// printf("batch: %03i, handling projections: %03i - %03i, angles: %.2f - %.2f\n",
			//	 batch+1, from_projection_next, to_projection_next-1, angles[from_projection_next], angles[to_projection_next-1]);

			// Allocate device memory for projections and upload
			//

			dims_next.clear();
			dims_next.push_back(projection_res_x);
			dims_next.push_back(projection_res_y);
			dims_next.push_back(projections_in_batch_next);

			cuNDArray<float> projections_batch_next(&dims, projections_DevPtr2);

			// Upload projections adhering to the binning.
			// Be sure to copy sequentially numbered projections in one copy operation.
			//

			int p = from_projection_next;

			while( p<to_projection_next ) {

				int num_sequential_projections = 1;
				while( p+num_sequential_projections < to_projection_next &&
						indices[p+num_sequential_projections]==(indices[p+num_sequential_projections-1]+1) ){
					num_sequential_projections++;
				}

				int from_id = indices[p];
				int size = projection_res_x*projection_res_y;

				cudaMemcpyAsync( projections_batch_next.get_data_ptr()+(p-from_projection_next)*size,
						projections->get_data_ptr()+from_id*size,
						size*num_sequential_projections*sizeof(float), cudaMemcpyHostToDevice, indyStream );

				CHECK_FOR_CUDA_ERROR();

				p += num_sequential_projections;
			}
		}

		// Define dimensions of grid/blocks.
		//

		dim3 dimBlock, dimGrid;
		setup_grid( matrix_size_x*matrix_size_y*matrix_size_z, &dimBlock, &dimGrid );

		// Invoke kernel
		//

		cudaFuncSetCacheConfig(conebeam_backwards_projection_kernel<FBP>, cudaFuncCachePreferL1);

		conebeam_backwards_projection_kernel<FBP><<< dimGrid, dimBlock, 0, mainStream >>>
				( image_device->get_data_ptr(), raw_angles, raw_offsets,
						is_dims_in_pixels, is_dims_in_mm, ps_dims_in_pixels, ps_dims_in_mm,
						projections_in_batch,  SDD, SAD, (batch==0) ? accumulate : true );

		CHECK_FOR_CUDA_ERROR();

		// Cleanup
		//

		cudaUnbindTexture(projections_tex);
		cudaFreeArray(projections_array);
		CHECK_FOR_CUDA_ERROR();

		std::swap(projections_DevPtr, projections_DevPtr2);
		std::swap(mainStream, indyStream);

		delete projections_batch;
		if( batch < num_batches-1 )
			projections_batch = new cuNDArray<float>(&dims_next, projections_DevPtr);
	}

	// Copy result from device to host
	//

	cudaMemcpy( image->get_data_ptr(), image_device->get_data_ptr(),
			num_image_elements*sizeof(float), cudaMemcpyDeviceToHost );

	CHECK_FOR_CUDA_ERROR();

	cudaFree(projections_DevPtr2);
	cudaFree(projections_DevPtr);
	CUDA_CALL(cudaStreamDestroy(indyStream));
	CUDA_CALL(cudaStreamDestroy(mainStream));
	CHECK_FOR_CUDA_ERROR();
}

struct cuMultBool {
	__device__ float operator()(float x, bool y) { return x*y;}
};
void apply_mask(cuNDArray<float>* image, cuNDArray<bool>* mask){
	thrust::transform(image->begin(),image->end(),mask->begin(),image->begin(),cuMultBool());

}

// Template instantiations
//

template void conebeam_backwards_projection<false>
( hoCuNDArray<float>*, hoCuNDArray<float>*, std::vector<float>, std::vector<floatd2>, std::vector<unsigned int>,
		int, intd3, floatd3, floatd2, float, float, bool, bool, bool, cuNDArray<float>*, cuNDArray<float>* );

template void conebeam_backwards_projection<true>
( hoCuNDArray<float>*, hoCuNDArray<float>*, std::vector<float>, std::vector<floatd2>, std::vector<unsigned int>,
		int, intd3, floatd3, floatd2, float, float, bool, bool, bool, cuNDArray<float>*, cuNDArray<float>* );
}
