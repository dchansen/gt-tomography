#include "cuBilateralFilter.h"
#include <vector_td_utilities.h>
using namespace Gadgetron;

template<int D> static __global__ void
bilateral_kernel1D(float* __restrict__ out, const float* __restrict__ image, const float* __restrict__ ref_image,vector_td<int,3> dims,float sigma_spatial,float sigma_int){

    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;



    if (ixo < dims[0] && iyo < dims[1] && izo < dims[2]){
        vector_td<int,3> coord(ixo,iyo,izo);
        vector_td<int,3> coord2(ixo,iyo,izo);
        int elements = prod(dims);
        int steps = max(ceil(sigma_spatial*4),1.0f);

        int idx = co_to_idx<3>(coord,dims);
        float res = 0;
        float image_value = ref_image[idx];


        float norm = 0;

        for (int i = -steps; i < steps; i++){
            coord2[D] = coord[D]+ i;

            int idx2 = co_to_idx<3>((coord2+dims)%dims,dims);

            float image_diff = image_value-ref_image[idx2];

            float weight = expf(-0.5f*float(i*i)/(2*sigma_spatial*sigma_spatial)
                                -image_diff*image_diff/(2*sigma_int*sigma_int));

            norm += weight;



            res += weight*image[idx2];

        }

        //atomicAdd(&out[co_to_idx<3>(coord,dims)],res/norm);



        out[idx] = res/norm;
    }

}


template<int D> static __global__ void
bilateral_kernel1D_unnormalized(float* __restrict__ out, const float* __restrict__ image, const float* __restrict__ ref_image,
                           vector_td<int,3> dims,float sigma_spatial,float sigma_int){

    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;



    if (ixo < dims[0] && iyo < dims[1] && izo < dims[2]){
        vector_td<int,3> coord(ixo,iyo,izo);
        vector_td<int,3> coord2(ixo,iyo,izo);
        int elements = prod(dims);
        int steps = max(ceil(sigma_spatial*4),1.0f);

        int idx = co_to_idx<3>(coord,dims);
        float res = 0;
        float image_value = ref_image[idx];


        float norm = 0;

        for (int i = -steps; i < steps; i++){
            coord2[D] = coord[D]+ i;

            int idx2 = co_to_idx<3>((coord2+dims)%dims,dims);

            float image_diff = image_value-ref_image[idx2];

            float weight = expf(-0.5f*float(i*i)/(2*sigma_spatial*sigma_spatial)
                                -image_diff*image_diff/(2*sigma_int*sigma_int));

            norm += weight;



            res += weight*image[idx2];

        }

        //atomicAdd(&out[co_to_idx<3>(coord,dims)],res/norm);



        out[idx] = res;
    }

}



void Gadgetron::bilateral_filter(cuNDArray<float>* image, cuNDArray<float>* ref_image,float  sigma_spatial,float sigma_int){
    cudaExtent extent;
    extent.width = image->get_size(0);
    extent.height = image->get_size(1);
    extent.depth = image->get_size(2);


    dim3 threads(8,8,8);
    dim3 grid((extent.width+threads.x-1)/threads.x, (extent.height+threads.y-1)/threads.y,(extent.depth+threads.z-1)/threads.z);


    vector_td<int, 3> image_dims(image->get_size(0),image->get_size(1), image->get_size(2));


    auto imcopy = *image;
    bilateral_kernel1D<0><<<grid,threads>>>(image->get_data_ptr(),imcopy.get_data_ptr(),ref_image->get_data_ptr(), image_dims, sigma_spatial, sigma_int);
    cudaDeviceSynchronize();
    bilateral_kernel1D<1><<<grid,threads>>>(imcopy.get_data_ptr(),image->get_data_ptr(),ref_image->get_data_ptr(), image_dims, sigma_spatial, sigma_int);
    cudaDeviceSynchronize();
    bilateral_kernel1D<2><<<grid,threads>>>(image->get_data_ptr(),imcopy.get_data_ptr(),ref_image->get_data_ptr(), image_dims, sigma_spatial, sigma_int);
    cudaDeviceSynchronize();

}

void Gadgetron::bilateral_filter_unnormalized(cuNDArray<float>* image, cuNDArray<float>* ref_image,float  sigma_spatial,float sigma_int){
    cudaExtent extent;
    extent.width = image->get_size(0);
    extent.height = image->get_size(1);
    extent.depth = image->get_size(2);


    dim3 threads(8,8,8);
    dim3 grid((extent.width+threads.x-1)/threads.x, (extent.height+threads.y-1)/threads.y,(extent.depth+threads.z-1)/threads.z);


    vector_td<int, 3> image_dims(image->get_size(0),image->get_size(1), image->get_size(2));


    auto imcopy = *image;
    bilateral_kernel1D_unnormalized<0><<<grid,threads>>>(image->get_data_ptr(),imcopy.get_data_ptr(),ref_image->get_data_ptr(), image_dims, sigma_spatial, sigma_int);
    cudaDeviceSynchronize();
    bilateral_kernel1D_unnormalized<1><<<grid,threads>>>(imcopy.get_data_ptr(),image->get_data_ptr(),ref_image->get_data_ptr(), image_dims, sigma_spatial, sigma_int);
    cudaDeviceSynchronize();
    bilateral_kernel1D_unnormalized<2><<<grid,threads>>>(image->get_data_ptr(),imcopy.get_data_ptr(),ref_image->get_data_ptr(), image_dims, sigma_spatial, sigma_int);
    cudaDeviceSynchronize();

}