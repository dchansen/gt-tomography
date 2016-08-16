#include <vector_td_utilities.h>
#include "cuPartialDifferenceOperator.h"
#include "cudaDeviceManager.h"
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

template<class T, unsigned int D, int SKIP, int DIM > __global__ static void partialDifferenceKernel(const T* __restrict__ in, T* __restrict__ out, vector_td<int,D> dims, bool accumulate){

    const int elements = prod(dims);
    const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < prod(dims) ){

        auto co = idx_to_co(idx,dims);
        //co[DIM] = (co[DIM]+SKIP+dims[DIM])%dims[DIM];
        auto val1 = in[co_to_idx(co,dims)];
        co[DIM] = (co[DIM]+SKIP+dims[DIM])%dims[DIM];
        auto val2 = in[co_to_idx(co,dims)];
        if (accumulate)
            out[idx] += (val2-val1);
        else
            out[idx] = (val2-val1);
    }


};


template<class T, unsigned int D, int SKIP, int DIM> static void partialDifference(cuNDArray<T>* in, cuNDArray<T>* out,
                                                                                   bool accumulate){

    auto dims = *in->get_dimensions();
    auto vdims = vector_td<int,D>(from_std_vector<size_t,D>(dims));

    const size_t elements_per_batch = prod(vdims);
    const size_t elements_total = in->get_number_of_elements();


    dim3 grid,block;

    setup_grid(elements_per_batch,&block,&grid);


    for (int i = 0; i < elements_total/elements_per_batch; i++){
        partialDifferenceKernel<T,D,SKIP,DIM><<<grid,block>>>(in->get_data_ptr()+i*elements_per_batch,
                out->get_data_ptr()+i*elements_per_batch, vdims,accumulate);
    }

}


template<class T, unsigned int D> void cuPartialDifferenceOperator<T,D>::mult_M(cuNDArray<T> *in, cuNDArray<T> *out,
                                                                                bool accumulate) {

    switch(dim) {
        case 0:
            partialDifference<T,D,1,0>(in,out,accumulate);
            break;
        case 1:
            partialDifference<T,D,1,1>(in,out,accumulate);
            break;
        case 2:
            partialDifference<T,D,1,2>(in,out,accumulate);
            break;
        case 3:
            partialDifference<T,D,1,3>(in,out,accumulate);
            break;
        default:
            throw std::runtime_error("Unsupported dimension");

    }

}

template<class T, unsigned int D> void cuPartialDifferenceOperator<T,D>::mult_MH(cuNDArray<T> *in, cuNDArray<T> *out,
                                                                                bool accumulate) {

    switch(dim) {
        case 0:
            partialDifference<T,D,-1,0>(in,out,accumulate);
            break;
        case 1:
            partialDifference<T,D,-1,1>(in,out,accumulate);
            break;
        case 2:
            partialDifference<T,D,-1,2>(in,out,accumulate);
            break;
        case 3:
            partialDifference<T,D,-1,3>(in,out,accumulate);
            break;
        default:
            throw std::runtime_error("Unsupported dimension");

    }

}

template class cuPartialDifferenceOperator<float,1>;
template class cuPartialDifferenceOperator<float,2>;
template class cuPartialDifferenceOperator<float,3>;
template class cuPartialDifferenceOperator<float,4>;