



#include <vector_td_utilities.h>


template<class T, unsigned int D> __global__ static void cuTVPrimalDualKernel(float* in, float* out, vector_td<float,D> dims){

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