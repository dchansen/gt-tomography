//
// This code performs 3D cone beam CT forwards and backwards projection
//

#include "ct_projection.h"
#include "float3x3.h"
#include "hoCuNDArray_math.h"
#include "vector_td.h"
#include "cuNDArray_math.h"
#include "cuNDArray_utils.h"
#include "cuNFFT.h"
#include "check_CUDA.h"
#include "GPUTimer.h"
#include "cudaDeviceManager.h"
#include "hoNDArray_fileio.h"
#include "vector_td_io.h"
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <cufft.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include <host_defines.h>
#include <driver_functions.h>
#include <device_launch_parameters.h>

// Read the projection/image data respectively as a texture (for input)
// - taking advantage of the cache and hardware interpolation
//

#define NORMALIZED_TC 0

static texture<float, 3, cudaReadModeElementType>
        image_tex(true, cudaFilterModeLinear, cudaAddressModeBorder);

static texture<float, cudaTextureType2DLayered, cudaReadModeElementType>
        projections_tex(false, cudaFilterModeLinear, cudaAddressModeBorder);

/*
static texture<float, cudaTextureType2DLayered, cudaReadModeElementType>
        projections_tex(false, cudaFilterModePoint, cudaAddressModeBorder);
*/
namespace Gadgetron {

    static inline
    void setup_grid(unsigned int number_of_elements, dim3 *blockDim, dim3 *gridDim, unsigned int num_batches = 1) {
        int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
        //int maxGridDim = cudaDeviceManager::Instance()->max_griddim(cur_device);
        int maxBlockDim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
        int maxGridDim = 65535;

        // The default one-dimensional block dimension is...
        *blockDim = dim3(256);
        *gridDim = dim3((number_of_elements + blockDim->x - 1) / blockDim->x, num_batches);

        // Extend block/grid dimensions if we exceeded the maximum grid dimension
        if (gridDim->x > maxGridDim) {
            blockDim->x = maxBlockDim;
            gridDim->x = (number_of_elements + blockDim->x - 1) / blockDim->x;
        }

        if (gridDim->x > maxGridDim) {
            gridDim->x = (unsigned int) std::floor(std::sqrt(float(number_of_elements) / float(blockDim->x)));
            unsigned int num_elements_1d = blockDim->x * gridDim->x;
            gridDim->y *= ((number_of_elements + num_elements_1d - 1) / num_elements_1d);
        }

        if (gridDim->x > maxGridDim || gridDim->y > maxGridDim) {
            // If this ever becomes an issue, there is an additional grid dimension to explore for compute models >= 2.0.
            throw cuda_error("setup_grid(): too many elements requested.");
        }
    }

    static inline
    void setup_grid3D(intd3 dims, dim3 *blockDim, dim3 *gridDim) {
        int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
        //int maxGridDim = cudaDeviceManager::Instance()->max_griddim(cur_device);
        int maxBlockDim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
        int maxGridDim = 65535;

        // The default one-dimensional block dimension is...
        *blockDim = dim3(std::min(8,dims[0]),std::min(8,dims[1]),std::min(8,dims[2]));

        *gridDim = dim3((dims[0]+blockDim->x-1)/blockDim->x,
                        (dims[1]+blockDim->y-1)/blockDim->y,
                        (dims[2]+blockDim->z-1)/blockDim->z);



    }
// Utility to convert from degrees to radians
//

    static inline __device__
            floatd3 cylindrical_to_cartesian(const floatd3 &cyl) {
        return floatd3(-sinf(cyl[0]) * cyl[1], cos(cyl[0]) * cyl[1], cyl[2]);
    }


    static inline __device__
            floatd3

    calculate_endpoint(const floatd3 &det_focal_cyl, const float &ADD, const floatd2 &spacing,
                       const floatd2 &central_element, const intd3 &co, const intd2 &ps_dims_in_pixels) {

        float phi = (det_focal_cyl[0] + CUDART_PI_F) + (co[0] - central_element[0]) * spacing[0] / ADD;
        //float phi = spacing[0]*elements[0]*(co[0]-central_element[0])/(det_focal_cyl[1]);
/*
        float x = ADD*cos(phi)+det_focal_cyl[1]*cos(det_focal_cyl[0]);
        float y = ADD*sin(phi)+det_focal_cyl[1]*sin(det_focal_cyl[0]);
        float z = (co[1]-central_element[1])*spacing[1]+det_focal_cyl[2];

        return floatd3(x,y,z);
        */
        floatd3 tmp(phi, ADD, (ps_dims_in_pixels[1] - co[1]-1.0f - central_element[1]) * spacing[1]);
        return cylindrical_to_cartesian(tmp) + cylindrical_to_cartesian(det_focal_cyl);
    }


//
// Forwards projection
//

    __global__ void
    ct_forwards_projection_kernel(float *__restrict__ projections,
                                  const floatd3 *__restrict__ detector_focal_cyls,
                                  const floatd3 *__restrict__ focal_offset_cyls,
                                  const floatd2 *__restrict__ centralElements,
                                  floatd3 is_dims_in_pixels,
                                  floatd3 is_dims_in_mm,
                                  intd2 ps_dims_in_pixels,
                                  floatd2 ps_spacing,
                                  int num_projections,
                                  float ADD,
                                  int num_samples_per_ray, bool accumulate) {
//const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
        const int num_elements = prod(ps_dims_in_pixels) * num_projections;
        const intd3 co = intd3(threadIdx.x+blockDim.x*blockIdx.x,threadIdx.y+blockDim.y*blockIdx.y,threadIdx.z+blockDim.z*blockIdx.z);

        if (co[0] < ps_dims_in_pixels[0] && co[1] < ps_dims_in_pixels[1] && co[2] < num_projections) {



            // Projection space dimensions and spacing
            //


            // Determine projection angle and rotation matrix
            //

            floatd3 detector_focal_cyl = detector_focal_cyls[co[2]];
            floatd3 focal_cyl = detector_focal_cyl + focal_offset_cyls[co[2]];
            const floatd2 centralElement = centralElements[co[2]];
            // Find start and end point for the line integral (image space)
            //
            //floatd3 startPoint = floatd3(focal_cyl[1]*cos(focal_cyl[0]),focal_cyl[1]*sin(focal_cyl[0]),focal_cyl[2]);
            floatd3 startPoint = cylindrical_to_cartesian(focal_cyl);
            //if (co[0] == 350 && co[1] == 32) printf("Start point %f %f %f \n",startPoint[0],startPoint[1],startPoint[2]);

            // Projection plate indices
            //

            //if (co[0] == 350 && co[1] == 32) printf("Start point %f %f %f \n",startPoint[0],startPoint[1],startPoint[2]);

            // Find direction vector of the line integral
            //
            floatd3 endPoint = calculate_endpoint(detector_focal_cyl, ADD, ps_spacing, centralElement, co,
                                                  ps_dims_in_pixels);

            floatd3 dir = endPoint - startPoint;

            for (int i = 0; i < 3; i++){
                if (abs(dir[i]) < 1e-6f) dir[i] = 1e-6f;
            }
            //if (co[0] == 350 && co[1] == 32) printf("End point %f %f %f %f \n",endPoint[0],endPoint[1],endPoint[2],norm(dir));
            // Perform integration only inside the bounding cylinder of the image volume
            //

            const floatd3 vec_over_dir = (is_dims_in_mm / 2 - startPoint) / dir;
            const floatd3 vecdiff_over_dir = (-is_dims_in_mm / 2 - startPoint) / dir;
            floatd3 start = amin(vecdiff_over_dir, vec_over_dir);
            floatd3 end = amax(vecdiff_over_dir, vec_over_dir);



            float a1 = fmax(max(start), 0.0f);
            float aend = fmin(min(end), 1.0f);
            startPoint += a1 * dir;

            //if (co[0] == 350 && co[1] == 32) printf("Start point %f %f %f %f \n",startPoint[0],startPoint[1],startPoint[2],a1);
            const float sampling_distance = norm((aend - a1) * dir) / num_samples_per_ray;
            //if (isnan(sampling_distance) || isinf(sampling_distance))
             //   printf("Sampling distance %f %f %f %f %f %f\n",sampling_distance,a1,aend,dir[0],dir[1],dir[2]);
            //if (co[0] == 350 && co[1] == 32) printf("sampling distance %f \n",sampling_distance);

            // Now perform conversion of the line integral start/end into voxel coordinates
            //

            startPoint /= is_dims_in_mm;

            startPoint += 0.5f;
            dir /= is_dims_in_mm;
            dir *= (aend-a1)/float(num_samples_per_ray); // now in step size units

            //if (co[0] == 350 && co[1] == 32) printf("Dir %f %f %f \n",dir[0],dir[1],dir[2]);
            //if (co[0] == 350 && co[1] == 32) printf("ADD %f \n",ADD);
            //
            // Perform line integration
            //

            float result = 0.0f;

            for (int sampleIndex = 0; sampleIndex < num_samples_per_ray; sampleIndex++) {


                floatd3 samplePoint = startPoint + dir * float(sampleIndex);

                // Accumulate result
                //

                result += tex3D(image_tex, samplePoint[0], samplePoint[1], samplePoint[2]);
            }

            // Output (normalized to the length of the ray)
            //
            if (accumulate)
                projections[co[0]+co[1]*ps_dims_in_pixels[0]+co[2]*ps_dims_in_pixels[0]*ps_dims_in_pixels[1]] += result * sampling_distance;
            else
                projections[co[0]+co[1]*ps_dims_in_pixels[0]+co[2]*ps_dims_in_pixels[0]*ps_dims_in_pixels[1]] = result * sampling_distance;
        }
    }

    void ct_forwards_projection(cuNDArray<float> *projections,
                                cuNDArray<float> *image,
                                std::vector<floatd3> &detector_focal_cyls, //phi,rho,z of the detector focal spot in units of rad, mm and mm
                                std::vector<floatd3> &focal_offset_cyls, // phi,rho,z offset of the source focal spot compared to detector focal spot
                                std::vector<floatd2> &centralElements, // Central element on the detector
                                floatd3 is_dims_in_mm, // Image size in mm
                                floatd2 ps_spacing,  //Size of each projection element in mm
                                float ADD, //focalpoint - detector disance
                                float samples_per_ray,
                                bool accumulate
    ) {

        //
        // Validate the input
        //

        if (projections == 0x0 || image == 0x0) {
            throw std::runtime_error("Error: conebeam_forwards_projection: illegal array pointer provided");
        }

        if (projections->get_number_of_dimensions() != 3) {
            throw std::runtime_error(
                    "Error: conebeam_forwards_projection: projections array must be three-dimensional");
        }

        if (image->get_number_of_dimensions() != 3) {
            throw std::runtime_error("Error: conebeam_forwards_projection: image array must be three-dimensional");
        }


        CHECK_FOR_CUDA_ERROR();
        int projection_res_x = projections->get_size(0);
        int projection_res_y = projections->get_size(1);
        int num_projections = projections->get_size(2);

        int matrix_size_x = image->get_size(0);
        int matrix_size_y = image->get_size(1);
        int matrix_size_z = image->get_size(2);


        image_tex.addressMode[0] = cudaAddressModeBorder;
        image_tex.addressMode[1] = cudaAddressModeBorder;
        image_tex.addressMode[2] = cudaAddressModeBorder;

        // Build texture from input image
        //

        //cudaFuncSetCacheConfig(ct_forwards_projection_kernel, cudaFuncCachePreferL1);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc < float > ();
        cudaExtent extent;
        extent.width = matrix_size_x;
        extent.height = matrix_size_y;
        extent.depth = matrix_size_z;

        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.kind = cudaMemcpyDeviceToDevice;
        cpy_params.extent = extent;

        cudaArray *image_array;
        cudaMalloc3DArray(&image_array, &channelDesc, extent);
        cpy_params.dstArray = image_array;
        cpy_params.srcPtr = make_cudaPitchedPtr
                ((void *) image->get_data_ptr(), extent.width * sizeof(float), extent.width, extent.height);
        cudaMemcpy3D(&cpy_params);

        cudaBindTextureToArray(image_tex, image_array, channelDesc);CHECK_FOR_CUDA_ERROR();

        // Allocate the angles, offsets and projections in device memory
        //
        thrust::device_vector<floatd3> detector_focal_cylVec(detector_focal_cyls);
        thrust::device_vector<floatd2> centralElementVec(centralElements);
        thrust::device_vector<floatd3> focal_offset_cylVec(focal_offset_cyls);

        auto raw_focal_cyl = thrust::raw_pointer_cast(detector_focal_cylVec.data());
        auto raw_centralElements = thrust::raw_pointer_cast(centralElementVec.data());
        auto raw_focal_offsets = thrust::raw_pointer_cast(focal_offset_cylVec.data());

        //
        // Iterate over the batches
        //


        // Block/grid configuration
        //

        dim3 dimBlock, dimGrid;
        setup_grid3D(intd3(projection_res_x,projection_res_y,num_projections), &dimBlock, &dimGrid);


        // Launch kernel
        //

        floatd3 is_dims_in_pixels(matrix_size_x, matrix_size_y, matrix_size_z);
        intd2 ps_dims_in_pixels(projection_res_x, projection_res_y);


        ct_forwards_projection_kernel << < dimGrid, dimBlock >> >
                                                    (projections->get_data_ptr(), raw_focal_cyl, raw_focal_offsets, raw_centralElements, is_dims_in_pixels,
                                                            is_dims_in_mm, ps_dims_in_pixels, ps_spacing, projections->get_size(
                                                            2), ADD,
                                                            samples_per_ray *
                                                            std::max(matrix_size_x, matrix_size_y), accumulate);

        // If not initial batch, start copying the old stuff
        //

        cudaUnbindTexture(image_tex);
        cudaFreeArray(image_array);

        CHECK_FOR_CUDA_ERROR();

    }


    void ct_forwards_projection(hoCuNDArray<float> *projections,
                                hoCuNDArray<float> *image,
                                std::vector<floatd3> &detector_focal_cyls, //phi,rho,z of the detector focal spot in units of rad, mm and mm
                                std::vector<floatd3> &focal_offset_cyls, // phi,rho,z offset of the source focal spot compared to detector focal spot
                                std::vector<floatd2> &centralElements, // Central element on the detector
                                floatd3 is_dims_in_mm, // Image size in mm
                                floatd2 ps_spacing,  //Size of each projection element in mm
                                float ADD, //focalpoint - detector disance
                                float samples_per_ray,
                                bool accumulate
    ) {

        //
        // Validate the input
        //

        if (projections == 0x0 || image == 0x0) {
            throw std::runtime_error("Error: conebeam_forwards_projection: illegal array pointer provided");
        }

        if (projections->get_number_of_dimensions() != 3) {
            throw std::runtime_error(
                    "Error: conebeam_forwards_projection: projections array must be three-dimensional");
        }

        if (image->get_number_of_dimensions() != 3) {
            throw std::runtime_error("Error: conebeam_forwards_projection: image array must be three-dimensional");
        }


        CHECK_FOR_CUDA_ERROR();
        size_t projection_res_x = projections->get_size(0);
        size_t projection_res_y = projections->get_size(1);
        size_t num_projections = projections->get_size(2);

        int matrix_size_x = image->get_size(0);
        int matrix_size_y = image->get_size(1);
        int matrix_size_z = image->get_size(2);

        // Build texture from input image
        //

        image_tex.addressMode[0] = cudaAddressModeBorder;
        image_tex.addressMode[1] = cudaAddressModeBorder;
        image_tex.addressMode[2] = cudaAddressModeBorder;
        //cudaFuncSetCacheConfig(ct_forwards_projection_kernel, cudaFuncCachePreferL1);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc < float > ();
        cudaExtent extent;
        extent.width = matrix_size_x;
        extent.height = matrix_size_y;
        extent.depth = matrix_size_z;

        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.kind = cudaMemcpyHostToDevice;
        cpy_params.extent = extent;


        cudaArray *image_array;
        cudaMalloc3DArray(&image_array, &channelDesc, extent);
        cpy_params.dstArray = image_array;
        cpy_params.srcPtr = make_cudaPitchedPtr
                ((void *) image->get_data_ptr(), extent.width * sizeof(float), extent.width, extent.height);
        cudaMemcpy3D(&cpy_params);

        cudaBindTextureToArray(image_tex, image_array, channelDesc);CHECK_FOR_CUDA_ERROR();

        // Allocate the angles, offsets and projections in device memory
        //
        thrust::device_vector<floatd3> detector_focal_cylVec(detector_focal_cyls);
        thrust::device_vector<floatd2> centralElementVec(centralElements);
        thrust::device_vector<floatd3> focal_offset_cylVec(focal_offset_cyls);

        auto raw_focal_cyl = thrust::raw_pointer_cast(detector_focal_cylVec.data());
        auto raw_centralElements = thrust::raw_pointer_cast(centralElementVec.data());
        auto raw_focal_offsets = thrust::raw_pointer_cast(focal_offset_cylVec.data());

        //
        // Iterate over the batches
        //


        // Block/grid configuration
        //
        cudaStream_t mainStream, indyStream;
        cudaStreamCreate(&mainStream);
        cudaStreamCreate(&indyStream);

        dim3 dimBlock, dimGrid;


        // Launch kernel
        //

        floatd3 is_dims_in_pixels(matrix_size_x, matrix_size_y, matrix_size_z);
        intd2 ps_dims_in_pixels(projection_res_x, projection_res_y);

        size_t batchSize = 2048;

        size_t remaining_projections = num_projections;

        cuNDArray<float> cu_proj1({projection_res_x, projection_res_y, batchSize});
        cuNDArray<float> cu_proj2({projection_res_x, projection_res_y, batchSize});

        hoCuNDArray<float> *tmp_proj;
        if (accumulate)
            tmp_proj = new hoCuNDArray<float>(projections->get_dimensions());
        else
            tmp_proj = projections;

        float *host_proj_ptr = tmp_proj->get_data_ptr();
        float *dev_proj_ptr = cu_proj1.get_data_ptr();
        float *dev_proj_ptr2 = cu_proj2.get_data_ptr();

        for (size_t i = 0; i < (num_projections + batchSize - 1) / batchSize; i++) {

            size_t batch_projections = std::min(batchSize, remaining_projections);
            setup_grid3D(intd3(projection_res_x,projection_res_y,batch_projections), &dimBlock, &dimGrid);

            size_t nelements = batch_projections * projection_res_x * projection_res_y;


            ct_forwards_projection_kernel << < dimGrid, dimBlock, 0, mainStream >> >
                                                                     (dev_proj_ptr, raw_focal_cyl, raw_focal_offsets, raw_centralElements, is_dims_in_pixels,
                                                                             is_dims_in_mm, ps_dims_in_pixels, ps_spacing, batch_projections, ADD,
                                                                             samples_per_ray *
                                                                             std::max(matrix_size_x,
                                                                                      matrix_size_y), false);
            cudaMemcpyAsync(host_proj_ptr, dev_proj_ptr, nelements * sizeof(float), cudaMemcpyDeviceToHost, mainStream);

            std::cout << " cuproj " << nrm2(&cu_proj1) << std::endl;
            raw_focal_cyl += batch_projections;
            raw_centralElements += batch_projections;
            raw_focal_offsets += batch_projections;

            host_proj_ptr += nelements;
            remaining_projections -= batch_projections;
            std::swap(mainStream, indyStream);
            std::swap(dev_proj_ptr, dev_proj_ptr2);
        }

        CUDA_CALL(cudaStreamDestroy(indyStream));
        CUDA_CALL(cudaStreamDestroy(mainStream));


        if (accumulate) {
            *projections += *tmp_proj;
            delete tmp_proj;
        }


        cudaUnbindTexture(image_tex);
        cudaFreeArray(image_array);

        CHECK_FOR_CUDA_ERROR();

    }
//
// Forwards projection of a 3D volume onto a set of (binned) projections
//

/**
 *
 * Assumption - projections are sorted by z. Each slice
 **/
    __global__ void
    ct_backwards_projection_kernel(float *__restrict__ image, // Image of size [nx,ny,nz]
                                   const floatd3 *__restrict__ detector_focal_cyls, //phi,rho,z of the detector focal spot in units of rad, mm and mm
                                   const floatd3 *__restrict__ focal_offset_cyls, // phi,rho,z offset of the source focal spot compared to detector focal spot
                                   const floatd2 *__restrict__ centralElements, // Central element on the detector
                                   const intd2 *__restrict__ proj_indices, // Array of size nz containing first to last projection for slice
                                   intd3 is_dims_in_pixels, //nx, ny, nz
                                   floatd3 is_dims_in_mm, // Image size in mm
                                   intd2 ps_dims_in_pixels, //Projection size
                                   floatd2 ps_spacing,  //Size of each projection element in mm
                                   float ADD, //focalpoint - detector disance
                                   int offset,
                                   int nprojs) {
        // Image voxel to backproject into (pixel coordinate and index)
        //

        const int num_elements = prod(is_dims_in_pixels);

        const intd3 co = intd3(threadIdx.x+blockDim.x*blockIdx.x,threadIdx.y+blockDim.y*blockIdx.y,threadIdx.z+blockDim.z*blockIdx.z);

        if (co[0] < is_dims_in_pixels[0] && co[1] < is_dims_in_pixels[1] && co[2] < is_dims_in_pixels[2]) {

            const floatd3 is_pc = floatd3(co) + floatd3(0.5);


            // Normalized image space coordinate [-0.5, 0.5[
            //

            //const floatd3 is_dims_in_pixels(is_dims_in_pixels);


            const floatd3 is_nc = is_pc / is_dims_in_pixels - floatd3(0.5f);

            // Image space coordinate in metric units
            //

            const floatd3 pos = is_nc * is_dims_in_mm;

            // Backprojection loop
            //

            //if (co[0] == 100 && co[1] == 100 && co[2] == 50) printf("ADD back %f\n",ADD);
            float result = 0.0f;

            for (int projection = proj_indices[co[2]][0];
                 projection < (proj_indices[co[2]][1]) && (projection - offset) < nprojs; projection++) {

                // Projection angle
                //
                const floatd3 detector_focal_cyl = detector_focal_cyls[projection];
                const floatd3 focal_cyl = focal_offset_cyls[projection] + detector_focal_cyl;

                floatd3 startPoint = cylindrical_to_cartesian(focal_cyl);

                //if (co[0] == 255 && co[1] == 255 && co[2] == 100) printf("Startpoint back %f %f %f \n",startPoint[0],startPoint[1],startPoint[2]);
                const floatd3 focal_point = cylindrical_to_cartesian(detector_focal_cyl);
                const floatd3 dir = pos - startPoint;
                startPoint -= focal_point;

                const float a = (dir[0] * dir[0] + dir[1] * dir[1]);
                const float b = 2 * startPoint[0] * dir[0] + 2 * startPoint[1] * dir[1];
                const float c = startPoint[0] * startPoint[0] + startPoint[1] * startPoint[1] - ADD * ADD;
                float t = (-b + ::sqrt(b * b - 4 * a * c)) / (2 * a);

                const floatd3 detectorPoint = startPoint + dir * t;
                //if (co[0] == 255 && co[1] == 255 && co[2] == 100) printf("Endpiont back %f %f %f %f \n",detectorPoint[0],detectorPoint[1],detectorPoint[2],norm(detectorPoint-startPoint));
                float test = fmod(atan2(-detectorPoint[0], detectorPoint[1]) - detector_focal_cyl[0] + 4 * CUDART_PI_F,
                                  2 * CUDART_PI_F) - CUDART_PI_F;
                const floatd2 element_rad = floatd2(test * ADD / ps_spacing[0],
                                                    detectorPoint[2] / ps_spacing[1]) +centralElements[projection];


                // Convert metric projection coordinates into pixel coordinates
                //

                // Read the projection data (bilinear interpolation enabled) and accumulate
                //

                //if (co[0] == 255 && co[1] == 255 && co[2] == 100) printf("Element rad %f %f %f %f \n",element_rad[0],element_rad[1],atan2(detectorPoint[1],detectorPoint[0]),detector_focal_cyl[0]);
                if (element_rad[1] > 0 && element_rad[1] < ps_dims_in_pixels[1])
                    result += tex2DLayered(projections_tex, element_rad[0], ps_dims_in_pixels[1] - element_rad[1]-1.0f,
                                           projection - offset);
                //result += pos[2];

            }

            // Output normalized image
            //

            image[co[0]+co[1]*is_dims_in_pixels[0]+co[2]*is_dims_in_pixels[0]*is_dims_in_pixels[1]] += result;
        }
    }

//
// Backprojection
//

    void ct_backwards_projection(cuNDArray<float> *projections,
                                 cuNDArray<float> *image,
                                 std::vector<floatd3> &detector_focal_cyls, //phi,rho,z of the detector focal spot in units of rad, mm and mm
                                 std::vector<floatd3> &focal_offset_cyls, // phi,rho,z offset of the source focal spot compared to detector focal spot
                                 std::vector<floatd2> &centralElements, // Central element on the detector
                                 std::vector<intd2> &proj_indices, // Array of size nz containing first to last projection for slice
                                 floatd3 is_dims_in_mm, // Image size in mm
                                 floatd2 ps_spacing,  //Size of each projection element in mm
                                 float ADD, //focalpoint - detector disance
                                 bool accumulate
    ) {
        //
        // Validate the input
        //

        if (projections == 0x0 || image == 0x0) {
            throw std::runtime_error("Error: conebeam_backwards_projection: illegal array pointer provided");
        }

        if (projections->get_number_of_dimensions() != 3) {
            throw std::runtime_error(
                    "Error: conebeam_backwards_projection: projections array must be three-dimensional");
        }

        if (image->get_number_of_dimensions() != 3) {
            throw std::runtime_error("Error: conebeam_backwards_projection: image array must be three-dimensional");
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


        // Allocate device memory for the backprojection result
        //
        // Allocate the angles, offsets and projections in device memory
        //

        thrust::device_vector<floatd3> detector_focal_cylVec(detector_focal_cyls);
        thrust::device_vector<floatd2> centralElementVec(centralElements);
        thrust::device_vector<floatd3> focal_offset_cylVec(focal_offset_cyls);
        thrust::device_vector<intd2> proj_indicesVec(proj_indices);

        auto raw_focal_cyl = thrust::raw_pointer_cast(detector_focal_cylVec.data());
        auto raw_centralElements = thrust::raw_pointer_cast(centralElementVec.data());
        auto raw_focal_offsets = thrust::raw_pointer_cast(focal_offset_cylVec.data());
        auto raw_proj_indices = thrust::raw_pointer_cast(proj_indicesVec.data());

        std::vector<size_t> dims;
        dims.push_back(projection_res_x);
        dims.push_back(projection_res_y);
        dims.push_back(num_projections);


        if (!accumulate)
            clear(image);
        //
        // Iterate over batches
        //

        intd3 is_dims_in_pixels(matrix_size_x, matrix_size_y, matrix_size_z);
        intd2 ps_dims_in_pixels(projection_res_x, projection_res_y);
//	int batchsize = cudaDeviceManager::Instance()->max_texture3d()[2];
        int batchsize = 2048;
        size_t num_batches = (num_projections + batchsize - 1) / batchsize;

        float *proj_ptr = projections->get_data_ptr();
        size_t elements_left = projections->get_number_of_elements();
        // Build array for input texture
        //
        int offset = 0;
        //cudaFuncSetCacheConfig(ct_backwards_projection_kernel , cudaFuncCachePreferL1);
        projections_tex.addressMode[0] = cudaAddressModeBorder;
        projections_tex.addressMode[1] = cudaAddressModeBorder;
        for (size_t batch = 0; batch < num_batches; batch++) {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc < float > ();
            cudaExtent extent;
            extent.width = projection_res_x;
            extent.height = projection_res_y;
            extent.depth = std::min(size_t(batchsize), elements_left / (projection_res_x * projection_res_y));

            //std::cout << "Extent " << extent.width << " " << extent.height << " " << extent.depth << std::endl;
            //std::cout << "Elements left " << elements_left << std::endl;
            //std::cout << "Pixels " << is_dims_in_pixels << std::endl;
            cudaArray *projections_array;
            cudaMalloc3DArray(&projections_array, &channelDesc, extent, cudaArrayLayered);CHECK_FOR_CUDA_ERROR();

            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.extent = extent;
            cpy_params.dstArray = projections_array;
            cpy_params.kind = cudaMemcpyDeviceToDevice;
            cpy_params.srcPtr =
                    make_cudaPitchedPtr((void *) proj_ptr, projection_res_x * sizeof(float),
                                        projection_res_x, projection_res_y);
            cudaMemcpy3D(&cpy_params);CHECK_FOR_CUDA_ERROR();

            cudaBindTextureToArray(projections_tex, projections_array, channelDesc);
            cudaThreadSynchronize();CHECK_FOR_CUDA_ERROR();

            // Upload projections for the next batch
            // - to enable streaming
            //
            // Define dimensions of grid/blocks.
            //

            dim3 dimBlock, dimGrid;
            setup_grid3D(is_dims_in_pixels, &dimBlock, &dimGrid);

            // Invoke kernel
            //


            ct_backwards_projection_kernel << < dimGrid, dimBlock >> >
                                                         (image->get_data_ptr(), raw_focal_cyl, raw_focal_offsets, raw_centralElements, raw_proj_indices,
                                                                 is_dims_in_pixels, is_dims_in_mm, ps_dims_in_pixels, ps_spacing,
                                                                 ADD, offset, extent.depth);

            CHECK_FOR_CUDA_ERROR();

            offset += extent.depth;
            proj_ptr += extent.width * extent.depth * extent.height;
            elements_left -= extent.width * extent.depth * extent.height;

            // Cleanup
            //

            cudaUnbindTexture(projections_tex);
            cudaFreeArray(projections_array);CHECK_FOR_CUDA_ERROR();
        }
    }


    void ct_backwards_projection(hoCuNDArray<float> *projections,
                                 hoCuNDArray<float> *image,
                                 std::vector<floatd3> &detector_focal_cyls, //phi,rho,z of the detector focal spot in units of rad, mm and mm
                                 std::vector<floatd3> &focal_offset_cyls, // phi,rho,z offset of the source focal spot compared to detector focal spot
                                 std::vector<floatd2> &centralElements, // Central element on the detector
                                 std::vector<intd2> &proj_indices, // Array of size nz containing first to last projection for slice
                                 floatd3 is_dims_in_mm, // Image size in mm
                                 floatd2 ps_spacing,  //Size of each projection element in mm
                                 float ADD, //focalpoint - detector disance
                                 bool accumulate
    ) {
        //
        // Validate the input
        //

        if (projections == 0x0 || image == 0x0) {
            throw std::runtime_error("Error: conebeam_backwards_projection: illegal array pointer provided");
        }

        if (projections->get_number_of_dimensions() != 3) {
            throw std::runtime_error(
                    "Error: conebeam_backwards_projection: projections array must be three-dimensional");
        }

        if (image->get_number_of_dimensions() != 3) {
            throw std::runtime_error("Error: conebeam_backwards_projection: image array must be three-dimensional");
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


        // Allocate device memory for the backprojection result
        //
        // Allocate the angles, offsets and projections in device memory
        //

        thrust::device_vector<floatd3> detector_focal_cylVec(detector_focal_cyls);
        thrust::device_vector<floatd2> centralElementVec(centralElements);
        thrust::device_vector<floatd3> focal_offset_cylVec(focal_offset_cyls);
        thrust::device_vector<intd2> proj_indicesVec(proj_indices);

        auto raw_focal_cyl = thrust::raw_pointer_cast(detector_focal_cylVec.data());
        auto raw_centralElements = thrust::raw_pointer_cast(centralElementVec.data());
        auto raw_focal_offsets = thrust::raw_pointer_cast(focal_offset_cylVec.data());
        auto raw_proj_indices = thrust::raw_pointer_cast(proj_indicesVec.data());

        std::vector<size_t> dims;
        dims.push_back(projection_res_x);
        dims.push_back(projection_res_y);
        dims.push_back(num_projections);

        cudaStream_t mainStream, indyStream;
        cudaStreamCreate(&mainStream);
        cudaStreamCreate(&indyStream);


        cuNDArray<float> cuImage;
        if (!accumulate) {
            cuImage = cuNDArray<float>(image->get_dimensions());
            clear(&cuImage);

        } else {
            cuImage = cuNDArray<float>(image);
        }

        //
        // Iterate over batches
        //

        intd3 is_dims_in_pixels(matrix_size_x, matrix_size_y, matrix_size_z);
        intd2 ps_dims_in_pixels(projection_res_x, projection_res_y);
//	int batchsize = cudaDeviceManager::Instance()->max_texture3d()[2];
        int batchsize = 2048;
        size_t num_batches = (num_projections + batchsize - 1) / batchsize;

        float *proj_ptr = projections->get_data_ptr();
        size_t elements_left = projections->get_number_of_elements();
        // Build array for input texture
        //
        int offset = 0;
        //cudaFuncSetCacheConfig(ct_backwards_projection_kernel , cudaFuncCachePreferL1);
        projections_tex.addressMode[0] = cudaAddressModeBorder;
        projections_tex.addressMode[1] = cudaAddressModeBorder;

        for (size_t batch = 0; batch < num_batches; batch++) {


            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc < float > ();
            cudaExtent extent;
            extent.width = projection_res_x;
            extent.height = projection_res_y;
            extent.depth = std::min(size_t(batchsize), elements_left / (projection_res_x * projection_res_y));

            //std::cout << "Extent " << extent.width << " " << extent.height << " " << extent.depth << std::endl;
            //std::cout << "Elements left " << elements_left << std::endl;
            //std::cout << "Pixels " << is_dims_in_pixels << std::endl;
            cudaArray *projections_array;
            cudaMalloc3DArray(&projections_array, &channelDesc, extent, cudaArrayLayered);CHECK_FOR_CUDA_ERROR();

            cudaMemcpy3DParms cpy_params = {0};
            cpy_params.extent = extent;
            cpy_params.dstArray = projections_array;
            cpy_params.kind = cudaMemcpyHostToDevice;
            cpy_params.srcPtr =
                    make_cudaPitchedPtr((void *) proj_ptr, projection_res_x * sizeof(float),
                                        projection_res_x, projection_res_y);
            cudaMemcpy3DAsync(&cpy_params, mainStream);CHECK_FOR_CUDA_ERROR();

            cudaBindTextureToArray(projections_tex, projections_array, channelDesc);


            // Upload projections for the next batch
            // - to enable streaming
            //
            // Define dimensions of grid/blocks.
            //

            dim3 dimBlock, dimGrid;
            setup_grid3D(is_dims_in_pixels, &dimBlock, &dimGrid);


            // Invoke kernel
            //


            ct_backwards_projection_kernel << < dimGrid, dimBlock, 0, mainStream >> >
                                                                      (cuImage.get_data_ptr(), raw_focal_cyl, raw_focal_offsets, raw_centralElements, raw_proj_indices,
                                                                              is_dims_in_pixels, is_dims_in_mm, ps_dims_in_pixels, ps_spacing,
                                                                              ADD, offset, extent.depth);

            CHECK_FOR_CUDA_ERROR();

            offset += extent.depth;
            proj_ptr += extent.width * extent.depth * extent.height;
            elements_left -= extent.width * extent.depth * extent.height;

            // Cleanup
            //

            cudaUnbindTexture(projections_tex);
            cudaFreeArray(projections_array);CHECK_FOR_CUDA_ERROR();
            std::swap(mainStream, indyStream);
        }
        CUDA_CALL(cudaStreamDestroy(indyStream));
        CUDA_CALL(cudaStreamDestroy(mainStream));
        cuImage.to_host(image);
    }
}
