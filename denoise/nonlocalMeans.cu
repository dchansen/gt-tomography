/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define NLM_WINDOW_RADIUS   3
#define NLM_BLOCK_RADIUS    3

#define NLM_WINDOW_AREA     ( (2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1)*(2 * NLM_WINDOW_RADIUS + 1) )
#define NLM_WINDOW_AREA2D     ( (2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1) )

#define INV_NLM_WINDOW_AREA ( 1.0f / (float)NLM_WINDOW_AREA )
#define INV_NLM_WINDOW_AREA2D ( 1.0f / (float)NLM_WINDOW_AREA2D )

#define NLM_WEIGHT_THRESHOLD    0.10f
#define NLM_LERP_THRESHOLD      0.10f

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8

#include "nonlocalMeans.h"
using namespace Gadgetron;
texture<float, 3, cudaReadModeElementType> nlmTex;

texture<float, 2, cudaReadModeElementType> nlmTex2D;

////////////////////////////////////////////////////////////////////////////////
// NLM kernel
////////////////////////////////////////////////////////////////////////////////
__global__ static void NLM3D(
    float *dst,
    int imageW,
    int imageH,
    int imageD,
    float Noise
)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const int iz = blockDim.z * blockIdx.z + threadIdx.z;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;
    const float z = (float)iz + 0.5f;

    if (ix < imageW && iy < imageH && iz < imageD)
    {
        //Normalized counter for the NLM weight threshold
        //float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float accum = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS; i++)
            for (float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS; j++)
                for (float k = -NLM_WINDOW_RADIUS; k <= NLM_WINDOW_RADIUS; k++)
            {
                //Find color distance from (x, y) to (x + j, y + i)
                float weightIJK = 0;

                for (float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
                    for (float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++)
                        for (float l = -NLM_BLOCK_RADIUS; l <= NLM_BLOCK_RADIUS; l++){
                            float diff =tex3D(nlmTex, x + j + m, y + i + n, z + l + k) -
                                        tex3D(nlmTex, x + m, y + n, z + l);
                            weightIJK += diff*diff;
                        }




                //Derive final weight from color and geometric distance
                weightIJK     = expf(-(weightIJK * Noise + (i * i + j * j+k*k) * INV_NLM_WINDOW_AREA));
                //weightIJK     = expf(-(weightIJK * Noise ));




                //Accumulate (x + j, y + i) texel color with computed weight#
                float IJK = tex3D(nlmTex, x + j, y + i,z+k);
                accum += IJK * weightIJK;


                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJK;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                //fCount      += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights

        accum /= sumWeights;

        //if (fCount > NLM_LERP_THRESHOLD)
        dst[imageW * iy + ix+imageH*imageW*iz] = accum ;
    }
}

__global__ static void NLM2D(
        float *dst,
        int imageW,
        int imageH,
        float Noise
)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;


    if (ix < imageW && iy < imageH)
    {
        //Normalized counter for the NLM weight threshold
        //float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float accum = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS; i++)
            for (float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS; j++)

                {
                    //Find color distance from (x, y) to (x + j, y + i)
                    float weightIJK = 0;

                    for (float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
                        for (float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++){

                                float diff =tex2D(nlmTex2D, x + j + m, y + i + n) -
                                            tex2D(nlmTex2D, x + m, y + n);
                                weightIJK += diff*diff;
                            }




                    //Derive final weight from color and geometric distance
                    weightIJK     = expf(-(weightIJK * Noise + (i * i + j * j) * INV_NLM_WINDOW_AREA2D));
                    //weightIJK     = expf(-(weightIJK * Noise ));




                    //Accumulate (x + j, y + i) texel color with computed weight#
                    float IJK = tex2D(nlmTex2D, x + j, y + i);
                    accum += IJK * weightIJK;


                    //Sum of weights for color normalization to [0..1] range
                    sumWeights  += weightIJK;

                    //Update weight counter, if NLM weight for current window texel
                    //exceeds the weight threshold
                    //fCount      += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
                }

        //Normalize result color by sum of weights

        accum /= sumWeights;

        //if (fCount > NLM_LERP_THRESHOLD)
        dst[imageW * iy + ix] = accum ;
    }
}




void Gadgetron::nonlocal_means(
    cuNDArray<float> *input, cuNDArray<float> *output ,
    float Noise
)
{
    if (!input->dimensions_equal(output))
        throw std::runtime_error("Input and output dimensions must agree");

    int imageW = input->get_size(0);
    int imageH = input->get_size(1);
    int imageD = input->get_size(2);


    nlmTex.addressMode[0] = cudaAddressModeClamp;
    nlmTex.addressMode[1] = cudaAddressModeClamp;
    nlmTex.addressMode[2] = cudaAddressModeClamp;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc < float > ();
    cudaExtent extent;
    extent.width = imageW;
    extent.height = imageH;
    extent.depth = imageD;

    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.kind = cudaMemcpyDeviceToDevice;
    cpy_params.extent = extent;

    cudaArray *image_array;
    cudaMalloc3DArray(&image_array, &channelDesc, extent);
    cpy_params.dstArray = image_array;
    cpy_params.srcPtr = make_cudaPitchedPtr
            ((void *) input->get_data_ptr(), extent.width * sizeof(float), extent.width, extent.height);
    cudaMemcpy3D(&cpy_params);

    cudaBindTextureToArray(nlmTex, image_array, channelDesc);CHECK_FOR_CUDA_ERROR();


    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y,BLOCKDIM_Z);

    dim3 grid((imageW+threads.x-1)/threads.x, (imageH+threads.y-1)/threads.y,(imageD+threads.z-1)/threads.z);

    NLM3D<<<grid, threads>>>(output->get_data_ptr(), imageW, imageH,imageD, 1.0/(Noise*Noise));
    cudaFreeArray(image_array);
}

void Gadgetron::nonlocal_means2D(
        cuNDArray<float> *input, cuNDArray<float> *output ,
        float Noise
)
{
    if (!input->dimensions_equal(output))
        throw std::runtime_error("Input and output dimensions must agree");

    int imageW = input->get_size(0);
    int imageH = input->get_size(1);
    int imageD = input->get_size(2);


    nlmTex2D.addressMode[0] = cudaAddressModeClamp;
    nlmTex2D.addressMode[1] = cudaAddressModeClamp;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc < float > ();

    cudaArray *image_array;
    cudaMallocArray(&image_array, &channelDesc, imageW, imageH);

    float* input_ptr = input->get_data_ptr();
    float* output_ptr = output->get_data_ptr();

    std::vector<size_t> dims2D = {imageW,imageH};
    for (int i = 0; i < imageD; i++) {


        cuNDArray<float> input_view(dims2D,input_ptr);

        cudaMemcpyToArray(image_array, 0, 0, input_view.get_data_ptr(), input_view.get_number_of_bytes(),
                          cudaMemcpyDeviceToDevice);

        cudaBindTextureToArray(nlmTex2D, image_array, channelDesc);CHECK_FOR_CUDA_ERROR();


        dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

        dim3 grid((imageW + threads.x - 1) / threads.x, (imageH + threads.y - 1) / threads.y);

        NLM2D << < grid, threads >> > (output_ptr, imageW, imageH, 1.0 / (Noise * Noise));

        output_ptr += input_view.get_number_of_elements();
        input_ptr += input_view.get_number_of_elements();


    }
    cudaFreeArray(image_array);
}
