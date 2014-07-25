
#include "proton_utils.h"
#include "proton_kernels.h"

#include "vector_td_utilities.h"



#define MAX_THREADS_PER_BLOCK 512
#define BLOCKS_PER_GRID 65535
#define MAX_BLOCKS 4096*4

namespace Gadgetron {
static size_t calculate_batch_size(size_t floats_per_proton = 14){

	size_t mem_per_proton = floats_per_proton*sizeof(float); // Need 12 floatS for the splines and 1 for the projection
	size_t free;
	size_t total;

	int res = cudaMemGetInfo(&free,&total);
	return 1024*1024*(free/(1024*1024*mem_per_proton)); //Divisons by 1024*1024 to ensure MB batch size
}

void rotate_splines(cuNDArray<floatd3> * splines,float angle){

	unsigned int elements = splines->get_number_of_elements()/4;
	unsigned int threadsPerBlock =std::min(elements,(unsigned int) MAX_THREADS_PER_BLOCK);
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (elements+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));


	// Invoke kernel
	int batchSize = dimGrid.x*dimBlock.x;
	//std::cout << "Starting forward kernel with grid " << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl;

	for (unsigned int offset = 0; offset < (elements+batchSize); offset += batchSize){
		rotate_splines_kernel<float><<< dimGrid, dimBlock >>> (splines->get_data_ptr(),angle,elements,offset);
	}

	cudaDeviceSynchronize();
	CHECK_FOR_CUDA_ERROR();


}

template<> void protonProjection<cuNDArray>(cuNDArray<float>* image, cuNDArray<float>* projections, cuNDArray<floatd3>* splines, floatd3 phys_dims, cuNDArray<float>* exterior_path_lengths){
	int dims =  projections->get_number_of_elements();

	int threadsPerBlock =std::min(dims,MAX_THREADS_PER_BLOCK);
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));
	uint64d3 _dims = from_std_vector<size_t,3>( *(image->get_dimensions().get()) );

	// Invoke kernel
	int batchSize = dimGrid.x*dimBlock.x;
	//std::cout << "Starting forward kernel with grid " << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl;
	cudaFuncSetCacheConfig(forward_kernel<float>, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(forward_kernel2<float>, cudaFuncCachePreferL1);
	for (int offset = 0; offset < (dims+batchSize); offset += batchSize){
		if (exterior_path_lengths != NULL)
			forward_kernel2<float><<< dimGrid, dimBlock >>> (image->get_data_ptr(), projections->get_data_ptr(),splines->get_data_ptr(),exterior_path_lengths->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
		else
			forward_kernel<float><<< dimGrid, dimBlock >>> (image->get_data_ptr(), projections->get_data_ptr(),splines->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
	}

	//cudaDeviceSynchronize();
	//CHECK_FOR_CUDA_ERROR();
}

template<> void protonProjection<hoCuNDArray>(hoCuNDArray<float> * image,hoCuNDArray<float> * projections, hoCuNDArray<floatd3> * splines, floatd3 phys_dims, hoCuNDArray<float>* exterior_path_lengths){
	size_t floats_per_proton = exterior_path_lengths == NULL ? 13 : 15;
	size_t max_batch_size = calculate_batch_size(floats_per_proton);
	size_t elements = projections->get_number_of_elements();
	size_t offset = 0;

	cuNDArray<float> cu_image(image);
	for (size_t n = 0; n < (elements+max_batch_size-1)/max_batch_size; n++){

		size_t batch_size = std::min(max_batch_size,elements-offset);
		std::vector<size_t> batch_dim;
		batch_dim.push_back(batch_size);

		hoCuNDArray<float> projections_view(&batch_dim,projections->get_data_ptr()+offset); //This creates a "view" of out

		cuNDArray<float> cu_projections(&projections_view);
		batch_dim.push_back(4);
		hoCuNDArray<vector_td<float,3> > splines_view(&batch_dim,splines->get_data_ptr()+offset*4); // This creates a "view" of splines
		cuNDArray<vector_td<float,3> > cu_splines(&splines_view);

		if (exterior_path_lengths != NULL){
			batch_dim.back() = 2;
			hoCuNDArray<float> EPL_view(&batch_dim,exterior_path_lengths->get_data_ptr()+offset*2); //This creates a "view" of out
			cuNDArray<float> cu_EPL(&EPL_view);
			protonProjection(&cu_image,&cu_projections, &cu_splines,phys_dims,&cu_EPL);
		}else
			protonProjection(&cu_image,&cu_projections, &cu_splines,phys_dims);

		cudaMemcpy(projections_view.get_data_ptr(),cu_projections.get_data_ptr(),batch_size*sizeof(float),cudaMemcpyDeviceToHost); //Copies back the data to the host
		offset += batch_size;

	}
}


template<> void protonBackprojection<cuNDArray>(cuNDArray<float> * image, cuNDArray<float> * projections, cuNDArray<floatd3>* splines, floatd3 phys_dims, cuNDArray<float>* exterior_path_lengths){
	int dims =  projections->get_number_of_elements();
	int threadsPerBlock =std::min(dims,MAX_THREADS_PER_BLOCK);
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));
	uint64d3 _dims = from_std_vector<size_t,3>( *(image->get_dimensions().get()) );

	// Invoke kernel
	int batchSize = dimBlock.x*dimGrid.x;

	cudaFuncSetCacheConfig(backwards_kernel<float>, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(backwards_kernel2<float>, cudaFuncCachePreferL1);
	for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){
		if (exterior_path_lengths != NULL){
			std::cout << "DEBUG DUCK SAYS HI!" << std::endl;
			backwards_kernel2<float><<< dimGrid, dimBlock >>> (projections->get_data_ptr(), image->get_data_ptr(),splines->get_data_ptr(),exterior_path_lengths->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
		}else
			backwards_kernel<float><<< dimGrid, dimBlock >>> (projections->get_data_ptr(), image->get_data_ptr(),splines->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
	}

}

template<> void protonBackprojection<hoCuNDArray>(hoCuNDArray<float>* image, hoCuNDArray<float>* projections, hoCuNDArray<floatd3>* splines, floatd3 phys_dims, hoCuNDArray<float>* exterior_path_lengths) {

	cuNDArray<float> cu_image(image);
	CHECK_FOR_CUDA_ERROR();

	size_t floats_per_proton = exterior_path_lengths == NULL ? 13 : 15;
	size_t max_batch_size = calculate_batch_size(floats_per_proton);
	size_t elements = projections->get_number_of_elements();
	size_t offset = 0;

	for (size_t n = 0; n < (elements+max_batch_size-1)/max_batch_size; n++){
		size_t batch_size = std::min(max_batch_size,elements-offset);
		std::vector<size_t> batch_dim;
		batch_dim.push_back(batch_size);

		hoCuNDArray<float> projection_view(&batch_dim,projections->get_data_ptr()+offset); //This creates a "view" of projections
		cuNDArray<float> cu_projections(&projection_view);
		batch_dim.push_back(4);
		hoCuNDArray<floatd3 > splines_view(&batch_dim,splines->get_data_ptr()+offset*4); // This creates a "view" of splines
		cuNDArray<floatd3 > cu_splines(&splines_view);

		if (exterior_path_lengths != NULL){
			batch_dim.back() = 2;
			hoCuNDArray<float> EPL_view(&batch_dim,exterior_path_lengths->get_data_ptr()+offset*2); //This creates a "view" of projections
			cuNDArray<float> cu_EPL(&EPL_view);
			protonBackprojection<cuNDArray>(&cu_image,&cu_projections,&cu_splines,phys_dims,&cu_EPL);
		} else
			protonBackprojection<cuNDArray>(&cu_image,&cu_projections,&cu_splines,phys_dims);
		CHECK_FOR_CUDA_ERROR();
		offset += batch_size;
	}

	cudaMemcpy(image->get_data_ptr(),cu_image.get_data_ptr(),cu_image.get_number_of_elements()*sizeof(float),cudaMemcpyDeviceToHost);
}

}
