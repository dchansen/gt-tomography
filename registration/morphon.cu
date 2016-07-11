#include "quadratureKernels.h"
#include "complext.h"
#include <thrust/device_vector.h>
#include <boost/make_shared.hpp>
#include "cuNDArray.h"
#include "vector_td_operators.h"
#include "vector_td_utilities.h"
#include <cub/cub.cuh>
#include <armadillo>
#include "cuNDArray_math.h"
#include "host_defines.h"

using namespace Gadgetron;
#define DIRECTIONS 6
#define KERNEL_HALF 4
#define KERNEL_SIZE 9


__device__ vector_td<float,6>  cholesky(vector_td<float,6>& A){
 vector_td<float,6> result;
    result[0] = sqrt(A[0]);
    result[1] = A[1]/result[0];
    result[2] = A[2]/result[0];
    result[3] = sqrt(A[3]-result[1]*result[1]);
    result[4] = (A[4]-result[2]*result[1])/result[3];
    result[5] = sqrt(A[5]-result[2]*result[2]-result[4]*result[4]);
    return result;

};

__device__ vector_td<float,3> solve(vector_td<float,6>& L, vector_td<float,3>& b){
    vector_td<float,3> result;

    result[2] = b[2]/L[5];
    result[1] = (b[1]-L[4]*result[2])/L[3];
    result[0] = (b[0]-L[2]*result[2]-L[1]*result[1])/L[0];


    result[0] = result[0]/L[0];
    result[1] = (result[1]-L[1]*result[0])/L[3];
    result[2] = (result[2]-L[2]*result[0]-L[4]*result[1])/L[5];

    return result;

};


static std::tuple<cudaTextureObject_t,cudaArray*> createTexture3D(cuNDArray<float>* image){

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

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = image_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);



    return std::make_tuple(texObj,image_array);


}
static __inline__ __device__ float_complext convolve(cudaTextureObject_t tex , const float_complext* kernels, const int ixo, const int iyo, const int izo ){
    float_complext result = 0;



    for (int dz = 0; dz < KERNEL_SIZE; dz++ ) {
        float uz = izo+0.5f+dz-KERNEL_HALF;
        for (int dy = 0; dy < KERNEL_SIZE; dy++) {
            float uy = iyo+0.5f+dy-KERNEL_HALF;
            for (int dx = 0; dx < KERNEL_SIZE; dx++) {
                float ux = ixo+0.5f+dx-KERNEL_HALF;
                result += tex3D<float>(tex, ux, uy, uz) * kernels[dx+dy*KERNEL_SIZE+dz*KERNEL_SIZE*KERNEL_SIZE];
            }
        }
    }

    return result;

}



static __global__ void  morphon_kernel(cudaTextureObject_t moving,cudaTextureObject_t fixed, float* __restrict__ update, float* __restrict__ tnorms, const float_complext* __restrict kernels, const vector_td<float,DIRECTIONS>* __restrict__ m, const vector_td<float,3>* __restrict__ filterDirections,int width, int height, int depth){


    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;

    if (ixo < width && iyo < height && izo < depth) {

        vector_td<float,DIRECTIONS> dk;
        vector_td<float,DIRECTIONS> ck;
        vector_td<float,DIRECTIONS> t;
        /*
        if (ixo == 100 && iyo == 100 && izo == 20) {
            for (int i  = 0; i < KERNEL_SIZE*KERNEL_SIZE*KERNEL_SIZE*DIRECTIONS; i++)
                printf("K %f + %f i\n",kernels[i].real(),kernels[i].imag());


        }
         */
        for (int dir = 0; dir < DIRECTIONS; dir++) {
            const float_complext * kernel = kernels + dir*KERNEL_SIZE*KERNEL_SIZE*KERNEL_SIZE;
            float_complext conv_mov = convolve(moving,kernel,ixo,iyo,izo);
            float_complext conv_fix = convolve(fixed,kernel,ixo,iyo,izo);


            float_complext qq = conv_mov*conv_fix;


            dk[dir] = atan2(imag(qq),real(qq));
            float dk_cos = cos(dk[dir]*0.5f);
            ck[dir] = sqrt(abs(qq))*dk_cos*dk_cos;
            t[dir] = abs(conv_fix);
        }

        for (int dir = 0; dir < DIRECTIONS; dir++){
            t += m[dir]*t[dir];
        }

        float tnorm = sqrt(t[0]*t[0]+2*t[1]*t[1]+2*t[2]*t[2]+t[3]*t[3]+2*t[4]*t[4]+t[5]*t[5]);
        /*
        typedef cub::BlockReduce<float, 8,cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,8,8> BlockReduce;
        // Allocate shared memory for BlockReduce
        __shared__ typename BlockReduce::TempStorage temp_storage;
        float tmax  = BlockReduce(temp_storage).Reduce(tnorm, cub::Max());
        if (threadIdx.x == 0 & threadIdx.y == 0 && threadIdx.z == 0)
            tnorms[blockIdx.x+blockIdx.y*gridDim.x+blockIdx.z*gridDim.x*gridDim.y] = tmax;
*/


        //t /= tnorm+1e-8;



        vector_td<float,6> A(0);
        vector_td<float,3> b(0);

        vector_td<float,6> t2 = t*t;

        vector_td<float,6> tt = {
                t2[0]+t2[1]+t2[3],
                t[0]*t[1]+t[1]*t[3]+t[2]*t[4],
                t[0]*t[2]+t[1]*t[4]+t[2]*t[5],
                t2[1]+t2[3]+t2[4],
                t[1]*t[2]+t[3]*t[4]+t[4]*t[5],
                t2[2]+t2[4]+t2[5]
        };

        for (int dir = 0; dir < DIRECTIONS; dir++){
            A += ck[dir]*tt;
            b[0] += ck[dir]*dk[dir]*(filterDirections[dir][0]*tt[0]+ filterDirections[dir][1]*tt[1]+filterDirections[dir][2]*tt[2]);
            b[1] += ck[dir]*dk[dir]*(filterDirections[dir][0]*tt[1]+ filterDirections[dir][1]*tt[3]+filterDirections[dir][2]*tt[4]);
            b[2] += ck[dir]*dk[dir]*(filterDirections[dir][0]*tt[2]+ filterDirections[dir][1]*tt[4]+filterDirections[dir][2]*tt[5]);


        }

        const int idx = ixo+iyo*width+izo*width*height;
        const int elements = width*depth*height;



        float anorm = A[0]*A[3]*A[5]-A[0]*A[4]*A[4]-A[1]*A[1]*A[5]
                     +A[1]*A[4]*A[2]+A[2]*A[1]*A[4]-A[2]*A[2]*A[3]+1e-8;



/*
        update[idx] = (b[1]*(A[0]*A[5]-A[2]*A[2])-b[2]*(A[0]*A[4]-A[2]*A[1])-b[0]*(A[1]*A[5]-A[4]*A[2]))/anorm;
        update[idx+elements] = (b[2]*(A[1]*A[4]-A[2]*A[3])-b[1]*(A[1]*A[5]-A[2]*A[4])+b[0]*(A[3]*A[5]-A[4]*A[4]))/anorm;
        update[idx+2*elements] = (b[2]*(A[0]*A[3]-A[1]*A[1])-b[1]*(A[0]*A[4]-A[1]*A[2])+b[0]*(A[1]*A[4]-A[3]*A[2]))/anorm;
*/

        tnorms[idx] = tnorm;


        auto L = cholesky(A);
        auto res = solve(L,b);
        update[idx] = res[0];
        update[idx+elements] = res[1];
        update[idx+2*elements] = res[2];



    }
    //Convolve

}

 boost::shared_ptr<cuNDArray<float>> morphon(cuNDArray<float>* moving, cuNDArray<float>* fixed){

     auto movTex = createTexture3D(moving);
     auto fixedTex = createTexture3D(fixed);

     auto outdims = *moving->get_dimensions();
     outdims.push_back(3);

     auto output = boost::make_shared<cuNDArray<float>>(outdims);



     thrust::device_vector<float_complext> kernel(quadratureKernels::kernels);


     thrust::device_vector<vector_td<float,6> > Tensor(quadratureKernels::Tensor);
     thrust::device_vector<floatd3> filterDirections(quadratureKernels::directions);

     dim3 threads(8, 8, 8);
     dim3 grid((moving->get_size(0) + threads.x - 1) / threads.x, (moving->get_size(1) + threads.y - 1) / threads.y,
               (moving->get_size(2) + threads.z - 1) / threads.z);


     //cuNDArray<float> tnorm(grid.x*grid.y*grid.z);
     cuNDArray<float> tnorm(moving->get_number_of_elements());
     clear(&tnorm);



    morphon_kernel<<<grid,threads>>>(std::get<0>(movTex),std::get<0>(fixedTex),output->get_data_ptr(),tnorm.get_data_ptr(),thrust::raw_pointer_cast(kernel.data()),
            thrust::raw_pointer_cast(Tensor.data()),thrust::raw_pointer_cast(filterDirections.data()),
    moving->get_size(0),moving->get_size(1),moving->get_size(2));


     float tmax = max(&tnorm);
     std::cout << " TMAX " << tmax << std::endl;
     *output /= tmax;

     cudaDestroyTextureObject(std::get<0>(movTex));
     cudaDestroyTextureObject(std::get<0>(fixedTex));

     cudaFreeArray(std::get<1>(movTex));
     cudaFreeArray(std::get<1>(fixedTex));


     return output;





 }