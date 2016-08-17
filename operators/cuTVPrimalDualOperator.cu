



#include <vector_td_utilities.h>


template<class T> __global__ static void cuTVPrimalKernel(float* in, float* out, vector_td<float,3> dims){

    const int elements = prod(dims);
	
    const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	const int iy = blockIdx.y*blockDim.y+threadIdx.y;
	const int iz = blockIdx.z*blockDim.z+threadIdx.z;
	const auto co = vector_td<int,3>(ix,iy,iz);
	vector_td<float,3> result;
    if (co < dims){
		auto val1 = in[co_to_idx(co,dims)];
		auto co2 = co;
        for (int i = 0; i < 3; i++){
			//co[DIM] = (co[DIM]+SKIP+dims[DIM])%dims[DIM];
			
			co2[i] = (co[i]+dims[i]+1)%dims[i];
			result[i] = in[co_to_idx(co2,dims)]-val1;
			co2[i] = co[i];
		}
		result /= max(1,norm(result));
        const int idx = ix+iy*dims[0]+iz*dims[0]*dims[1];
		for (int i =0; i < 3; i++)
			atomicAdd(&out[idx+i*elements],result[i]);
		
    }

};


template<class T> __global__ static void cuTVDualKernel(float* in, float* out, vector_td<float,3> dims){

    const int elements = prod(dims);
	
    const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	const int iy = blockIdx.y*blockDim.y+threadIdx.y;
	const int iz = blockIdx.z*blockDim.z+threadIdx.z;
	const auto co = vector_td<int,3>(ix,iy,iz);
	
    if (co < dims){
		const int idx = ix+iy*dims[0]+iz*dims[0]*dims[1];
		auto val1 = in[co_to_idx(co,dims)];
		auto co2 = co;
        for (int i = 0; i < 3; i++){
			//co[DIM] = (co[DIM]+SKIP+dims[DIM])%dims[DIM];
			
			co2[i] = (co[i]+dims[i]-1)%dims[i];
			auto result = in[co_to_idx(co2,dims)]-val1;
			co2[i] = co[i];
			atomicAdd(&out[idx+i*elements],result);
		}
		
			
		
    }

};