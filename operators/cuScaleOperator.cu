#include <vector_td_utilities.h>
#include "cuScaleOperator.h"
#include "cudaDeviceManager.h"
#include <vector_td_io.h>
using namespace Gadgetron;

static inline
void setup_grid( unsigned int number_of_elements, dim3 *blockDim, dim3* gridDim)
{
    int cur_device = cudaDeviceManager::Instance()->getCurrentDevice();
    //int maxGridDim = cudaDeviceManager::Instance()->max_griddim(cur_device);
    int maxBlockDim = cudaDeviceManager::Instance()->max_blockdim(cur_device);
    int maxGridDim = 65535;

    // The default one-dimensional block dimension is...
    *blockDim = dim3(256);
    *gridDim = dim3((number_of_elements+blockDim->x-1)/blockDim->x);

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

template<class T, unsigned int D, int DIM > __global__ static void downscale1DKernel(const T* __restrict__ in, T* __restrict__ out, vector_td<int,D> out_dims){

    const int elements = prod(out_dims);
    const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    auto in_dims = out_dims;
    in_dims[DIM] *= 2;

    if (idx < elements ){

        auto co_out = idx_to_co(idx,out_dims);
        auto co_in = co_out;
        co_in[DIM] *= 2;

        auto val1 = in[co_to_idx(co_in,in_dims)];
        co_in[DIM] +=1;
        auto val2 = in[co_to_idx(co_in,in_dims)];

        out[idx] = (val1+val2)*0.5f;
    }


};

template<class T, unsigned int D> __global__ static void upscaleKernel(const T* __restrict__ in, T* __restrict__ out, vector_td<int,D> out_dims, bool accumulate){

    const int elements = prod(out_dims);
    const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    auto in_dims = out_dims/2;

    if (idx < elements ){

        auto co_out = idx_to_co(idx,out_dims);
        auto co_in = co_out/2;


        int idx2 = co_to_idx(co_in,in_dims);
        //if (threadIdx.x == 31 && blockIdx.x == 13) printf("%i %i %i %i %i %i %i\n",co_in[0],co_in[1],co_in[2],in_dims[0],in_dims[1],in_dims[2],idx2);
        auto val1 = in[idx2];


        if (accumulate)
            out[idx] += val1/powf(2.0f,T(D));
        else
            out[idx] = val1/powf(2.0f,T(D));
    }


};


template<class T, unsigned int D> static void upscale2x(cuNDArray<T>* in, cuNDArray<T>* out,
                                                                                   bool accumulate){

    auto dims = *out->get_dimensions();
    auto vdims = vector_td<int,D>(from_std_vector<size_t,D>(dims));



    auto vdims_in =  vector_td<int,D>(from_std_vector<size_t,D>(*in->get_dimensions()));

    const size_t elements_per_batch = prod(vdims);
    const size_t elements_total = out->get_number_of_elements();

    const size_t elements_in_per_batch = prod(vdims_in);

    dim3 grid,block;

    setup_grid(elements_per_batch,&block,&grid);


    for (int i = 0; i < elements_total/elements_per_batch; i++){
        upscaleKernel<T,D><<<grid,block>>>(in->get_data_ptr()+i*elements_in_per_batch,
                out->get_data_ptr()+i*elements_per_batch, vdims,accumulate);
    }

}


template<class T, unsigned int D> static cuNDArray<T> downscale2x1D(cuNDArray<T>* in,int dim){

    auto in_dims = *in->get_dimensions();
    auto out_dims = in_dims;
    out_dims[dim] = in_dims[dim]/2;
    cuNDArray<T> out(out_dims);

    auto vdims = vector_td<int,D>(from_std_vector<size_t,D>(out_dims));
    auto vdims_in = vector_td<int,D>(from_std_vector<size_t,D>(in_dims));

    const size_t elements_per_batch = prod(vdims);
    const size_t elements_in_per_batch = prod(vdims);
    const size_t elements_total = out.get_number_of_elements();


    dim3 grid,block;

    setup_grid(elements_per_batch,&block,&grid);


    for (int i = 0; i < elements_total/elements_per_batch; i++){
        switch (dim){
            case 0:
                downscale1DKernel<T,D,0><<<grid,block>>>(in->get_data_ptr()+i*elements_in_per_batch,
                        out.get_data_ptr()+i*elements_per_batch, vdims);
                break;
            case 1:
                downscale1DKernel<T,D,1><<<grid,block>>>(in->get_data_ptr()+i*elements_in_per_batch,
                        out.get_data_ptr()+i*elements_per_batch, vdims);
                break;
            case 2:
                downscale1DKernel<T,D,2><<<grid,block>>>(in->get_data_ptr()+i*elements_in_per_batch,
                        out.get_data_ptr()+i*elements_per_batch, vdims);
                break;
            case 3:
                downscale1DKernel<T,D,3><<<grid,block>>>(in->get_data_ptr()+i*elements_in_per_batch,
                        out.get_data_ptr()+i*elements_per_batch, vdims);
                break;
        }

    }
    return out;
}


template<class T, unsigned int D> void cuScaleOperator<T,D>::mult_M(cuNDArray<T> *in, cuNDArray<T> *out,
                                                                                bool accumulate) {

    auto tmp_array = downscale2x1D<T,D>(in,0);
    for (int i = 1; i < D; i++){
        tmp_array = downscale2x1D<T,D>(&tmp_array,i);
    }
    if (accumulate)
        *out += tmp_array;
    else
        *out = tmp_array;


}

template<class T, unsigned int D> void cuScaleOperator<T,D>::mult_MH(cuNDArray<T> *in, cuNDArray<T> *out,
                                                                                bool accumulate) {


   upscale2x<T,D>(in,out,accumulate);


}

template class cuScaleOperator<float,1>;
template class cuScaleOperator<float,2>;
template class cuScaleOperator<float,3>;
template class cuScaleOperator<float,4>;