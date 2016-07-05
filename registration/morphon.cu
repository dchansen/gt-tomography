#include "quadratureKernels.h"
#include "complext.h"

using namespace Gadgetron;
#define DIRECTIONS 6
#define KERNEL_HALF 4
#define KERNEL_SIZE 9

static __inline__ __device__ complext_float convolve(cudaTextureObject_t tex , const complext_float* __restrict__ kernel ){
    complext_float result = 0;


#pragma unroll
    for (int dz = -KERNEL_HALF; dz++; dz <= KERNEL_HALF ) {
        float uz = izo+0.5f+dz;
#pragma unroll
        for (int dy = -KERNEL_HALF; dy++; dy <= KERNEL_HALF) {
            float uy = iyo+0.5f+dy;
#pragma unroll
            for (int dx = -KERNEL_HALF; dx++; dx <= KERNEL_HALF) {
                float ux = ixo+0.5f+dx;
                conv_mov += tex3D<float>(moving, ux, uy, uz) * kernels[dx+dy*KERNEL_SIZE+dz*KERNEL_SIZE*KERNEL_SIZE];
            }
        }
    }

    return result;

}

static __global__ void  morphon_kernel(cudaTextureObject_t moving,cudaTextureObject_t fixed, const complext_float* __restrict kernels, vector_td<float,DIRECTIONS> m,int width, int height, int depth){


    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;

    if (ixo < width && iyo < height && izo < depth) {

        vector_td<float,DIRECTIONS> dk;
        vector_td<float,DIRECTIONS> ck;
        vector_td<float,DIRECTIONS> t;
        for (int dir = 0; dir++; dir < DIRECTIONS) {
            const float * kernel = kernels + dir*KERNEL_SIZE*KERNEL_SIZE*KERNEL_SIZE;
            complext_float conv_mov = convolve(moving,kernel);
            complext_float conv_fix = convolve(fixed,kernel);

            complext_float qq = conv_fix*conv_mov;
            dk[dir] = atan2(real(qq),imag(qq));
            float dk_cos = cos(dk[dir]*0.5f);
            ck[dir] = sqrt(abs(qq))*dk_cos*dk_cos;
            t += abs(conv_fix);
        }

        t *= m;

        vector_td<float,6> A(0);
        vector_td<float,3> b(0);

        vector_td<float,6> t2 = t*t;

        vector_td<float,6> tt = {
                t2[0]+t2[1]+t2[3],
                t[0]*t[1]+t[1]*t[3]+t[2]*t[4],
                t[0]*t[2]+t[1]*t[4]+t[2]*t[5],
                t2[1]+t2[3]+t2[5],
                t[1]*t[2]+t[3]*t[4]+t[4]*t[5],
                t2[2]+t2[4]+t2[5]
        };

        for (int dir = 0; dir < DIRECTIONS; dir++){
            A += ck[dir]*tt;
            b[0] += ck[dir]*dk[dir]*(filterDirections[dir][0]*tt[0]+ filterDirections[dir][1]*tt[1]+filterDirections[dir][2]*tt[2]);
            b[1] += ck[dir]*dk[dir]*(filterDirections[dir][0]*tt[1]+ filterDirections[dir][1]*tt[3]+filterDirections[dir][2]*tt[4]);
            b[2] += ck[dir]*dk[dir]*(filterDirections[dir][0]*tt[2]+ filterDirections[dir][1]*tt[4]+filterDirections[dir][2]*tt[5]);


        }

        float norm = A[0]*A[3]*A[5]-A[0]*A[4]*A[4]-A[1]*A[1]*A[5]
                     +A[1]*A[4]*A[2]+A[2]*A[1]*A[4]-A[2]*A[2]*A[3];

        update[idx] = (b[1]*(A[0]*A[5]-A[2]*A[2])-b[2]*(A[0]*A[4]-A[2]*A[1])-b[0]*(A[1]*A[5]-A[4]*A[2]))/norm;
        update[idx+elements] = (b[2]*(A[1]*A[4]-A[2]*A[3])-b[1]*(A[1]*A[5]-A[2]*A[4])+b[0]*(A[3]*A[5]-A[4]*A[4]))/norm;
        update[idx+2*elements] = (b[2]*(A[0]*A[3]-A[1]*A[1])-b[1]*(A[0]*A[4]-A[1]*A[2])+b[0]*(A[1]*A[4]-A[3]*A[2]))/norm;




    }
    //Convolve

}