#include "splineBackprojectionOperator.h"
#include "vector_td_utilities.h"

#include "vector_td_io.h"


#include "check_CUDA.h"

#include <vector>

#include <stdio.h>
#include <stdexcept>
#include <algorithm>

#include "hoNDArray_operators.h"
#include "hoNDArray_elemwise.h"

#include "hoCuNDArray_blas.h"
#include "cuNDArray_blas.h"
#include "cuNDArray_elemwise.h"

#include "proton_kernels.cu"


#define MAX_THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 65535
#define MAX_BLOCKS 4096*4


 static size_t calculate_batch_size(){

	int mem_per_proton = 14*sizeof(float); // Need 12 floatS for the splines and 1 for the projection
	size_t free;
	size_t total;

	int res = cudaMemGetInfo(&free,&total);
	return 1024*1024*(free/(1024*1024*mem_per_proton)); //Divisons by 1024*1024 to ensure MB batch size
}

template<> void splineBackprojectionOperator<hoCuNDArray>
    ::mult_M( hoCuNDArray<float>* in, hoCuNDArray<float>* out_orig, bool accumulate ) {
	 if( !in || !out_orig){
	   throw std::runtime_error("cuOperatorPathBackprojection: mult_M empty data pointer");
	  }

	 hoCuNDArray<float>* out = out_orig;
	 if (data->get_weights() && accumulate){
		 out = new hoCuNDArray<float>(out_orig->get_dimensions());
		 clear(out);
	 }
	 cuNDArray<float> image(in);
	 size_t max_batch_size = calculate_batch_size();
	 size_t elements = out->get_number_of_elements();
	 size_t offset = 0;


	 for (size_t n = 0; n < (elements+max_batch_size-1)/max_batch_size; n++){

		 size_t batch_size = std::min(max_batch_size,elements-offset);
		 std::vector<size_t> batch_dim;
		 batch_dim.push_back(batch_size);

		 hoCuNDArray<float> out_view(&batch_dim,out->get_data_ptr()+offset); //This creates a "view" of out

		 cuNDArray<float> out_dev(&out_view);
		 batch_dim.push_back(4);
		 hoCuNDArray<vector_td<float,3> > splines_view(&batch_dim,data->get_splines()->get_data_ptr()+offset*4); // This creates a "view" of splines
		 cuNDArray<vector_td<float,3> > splines_dev(&splines_view);
		 if (!accumulate) clear(&out_dev);

		 int threadsPerBlock = std::min((int)batch_size,MAX_THREADS_PER_BLOCK);
		 dim3 dimBlock( threadsPerBlock);
		 int totalBlocksPerGrid = (batch_size+threadsPerBlock-1)/threadsPerBlock;
		 dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));
		 typename uint64d<3>::Type _dims = from_std_vector<size_t,3>( *(image.get_dimensions().get()) );

		 // Invoke kernel
		 int offset_k = 0;
		 //std::cout << "Starting forward kernel with grid " << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl;
		 cudaFuncSetCacheConfig(forward_kernel<float>, cudaFuncCachePreferL1);
		 for (int i = 0; i <= (totalBlocksPerGrid+dimGrid.x-1)/dimGrid.x; i++){
			forward_kernel<float><<< dimGrid, dimBlock >>> (image.get_data_ptr(), out_dev.get_data_ptr(),splines_dev.get_data_ptr(),physical_dims, (vector_td<int,3>)_dims, batch_size,offset_k);
			offset_k += dimBlock.x*dimGrid.x;
		 }
		 //cudaDeviceSynchronize();
		 CHECK_FOR_CUDA_ERROR();



		 cudaMemcpy(out_view.get_data_ptr(),out_dev.get_data_ptr(),batch_size*sizeof(float),cudaMemcpyDeviceToHost); //Copies back the data to the host

		offset += batch_size;

	 }

	 if (data->get_weights()){
		 *out *= *data->get_weights();
		 if (accumulate){
			 *out_orig += *out;
			 delete out;
		 }

	 }

}

template<> void splineBackprojectionOperator<hoCuNDArray>
    ::mult_MH( hoCuNDArray<float>* in_orig, hoCuNDArray<float>* out, bool accumulate ) {
	 if( !in_orig || !out){
	   throw std::runtime_error("cuOperatorPathBackprojection: mult_MH empty data pointer");
	  }
	 hoCuNDArray<float>* in = in_orig;

	 if (data->get_weights()){
		 in = new hoCuNDArray<float>(*in_orig);
		 *in *= *data->get_weights();
	 }


	 if (data->get_weights()){
		 delete in;
	 }
}


// Instantiations
template class splineBackprojectionOperator<hoCuNDArray>;

