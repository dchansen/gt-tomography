
#include "proton_utils.h"
#include "proton_kernels.h"

#include "vector_td_utilities.h"
#include "setup_grid.h"

#include "cudaDeviceManager.h"


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

	unsigned int threadsPerBlock =std::min(elements,(unsigned int) cudaDeviceManager::Instance()->max_blockdim());
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (elements+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,cudaDeviceManager::Instance()->max_griddim()));


	// Invoke kernel
	int batchSize = dimGrid.x*dimBlock.x;
	//std::cout << "Starting forward kernel with grid " << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl;

	for (unsigned int offset = 0; offset < (elements+batchSize); offset += batchSize){
		rotate_splines_kernel<float><<< dimGrid, dimBlock >>> (splines->get_data_ptr(),angle,elements,offset);
	}

	cudaDeviceSynchronize();
	CHECK_FOR_CUDA_ERROR();


}

// Expand and fill with nearest value
template<class T, unsigned int D>
__global__ void pad_nearest_kernel( vector_td<unsigned int,D> matrix_size_in, vector_td<unsigned int,D> matrix_size_out,
                            const T * __restrict__ in, T * __restrict__ out, unsigned int number_of_batches, unsigned int num_elements)
{
  const unsigned int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned int frame_offset = idx/num_elements;

  if( idx < num_elements*number_of_batches ){

    const typename uintd<D>::Type co_out = idx_to_co<D>( idx-frame_offset*num_elements, matrix_size_out );
    const typename uintd<D>::Type offset = (matrix_size_out-matrix_size_in)>>1;
    T _out;
    bool inside = (co_out>=offset) && (co_out<(matrix_size_in+offset));

    if( inside )
      _out = in[co_to_idx<D>(co_out-offset, matrix_size_in)+frame_offset*prod(matrix_size_in)];
    else{
    	const vector_td<unsigned int,D> co_in = amax(amin(co_out,matrix_size_in-1u),0u);
      _out = in[co_to_idx<D>(co_in,matrix_size_in)+frame_offset*prod(matrix_size_in)];
    }

    out[idx] = _out;
  }
}

template<class T, unsigned int D>
 void pad_nearest( cuNDArray<T> *in, cuNDArray<T> *out )
 {
   if( in == 0x0 || out == 0x0 ){
     throw std::runtime_error("pad: 0x0 ndarray provided");;
   }

   if( in->get_number_of_dimensions() != out->get_number_of_dimensions() ){
     throw std::runtime_error("pad: image dimensions mismatch");;
   }

   if( in->get_number_of_dimensions() < D ){
     std::stringstream ss;
     ss << "pad: number of image dimensions should be at least " << D;
     throw std::runtime_error(ss.str());;
   }

   typename uint64d<D>::Type matrix_size_in = from_std_vector<size_t,D>( *in->get_dimensions() );
   typename uint64d<D>::Type matrix_size_out = from_std_vector<size_t,D>( *out->get_dimensions() );

   unsigned int number_of_batches = 1;
   for( unsigned int d=D; d<in->get_number_of_dimensions(); d++ ){
     number_of_batches *= in->get_size(d);
   }

   if( weak_greater(matrix_size_in,matrix_size_out) ){
     throw std::runtime_error("pad: size mismatch, cannot expand");
   }

   // Setup block/grid dimensions
   dim3 blockDim; dim3 gridDim;
   setup_grid( prod(matrix_size_out), &blockDim, &gridDim, number_of_batches );

   // Invoke kernel
   pad_nearest_kernel<T,D><<< gridDim, blockDim >>>
     ( vector_td<unsigned int,D>(matrix_size_in), vector_td<unsigned int,D>(matrix_size_out),
       in->get_data_ptr(), out->get_data_ptr(), number_of_batches, prod(matrix_size_out) );

   CHECK_FOR_CUDA_ERROR();
 }

template<> void protonProjection<cuNDArray>(cuNDArray<float>* image, cuNDArray<float>* projections, cuNDArray<floatd3>* splines, floatd3 phys_dims, cuNDArray<float>* exterior_path_lengths){
	int dims =  projections->get_number_of_elements();

	int threadsPerBlock =std::min(dims,cudaDeviceManager::Instance()->max_blockdim());
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,cudaDeviceManager::Instance()->max_griddim()));
	uint64d3 _dims = from_std_vector<size_t,3>( *(image->get_dimensions()) );

	// Invoke kernel
	int batchSize = dimGrid.x*dimBlock.x;
	//std::cout << "Starting forward kernel with grid " << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl;
	cudaFuncSetCacheConfig(path_kernel2<float,forward_functor<float> >, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(path_kernel<float,forward_functor<float> >, cudaFuncCachePreferL1);
	for (int offset = 0; offset < (dims+batchSize); offset += batchSize){
		forward_functor<float> functor(image->get_data_ptr(),projections->get_data_ptr());
		if (exterior_path_lengths != NULL){
			path_kernel2<float,forward_functor<float> ><<< dimGrid, dimBlock >>> (functor,splines->get_data_ptr(),exterior_path_lengths->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
			//forward_kernel2<float><<< dimGrid, dimBlock >>> (image->get_data_ptr(), projections->get_data_ptr(),splines->get_data_ptr(),exterior_path_lengths->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
		}else
			path_kernel<float,forward_functor<float> ><<< dimGrid, dimBlock >>> (functor,splines->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
	}

	//cudaDeviceSynchronize();
	//CHECK_FOR_CUDA_ERROR();
}


template<> void protonPathNorm<cuNDArray>(std::vector<size_t> img_dims, cuNDArray<float>* projections, cuNDArray<floatd3>* splines, floatd3 phys_dims, cuNDArray<float>* exterior_path_lengths){
	int dims =  projections->get_number_of_elements();

	int threadsPerBlock =std::min(dims,cudaDeviceManager::Instance()->max_blockdim());
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,cudaDeviceManager::Instance()->max_griddim()));
	uint64d3 _dims = from_std_vector<size_t,3>( img_dims );

	// Invoke kernel
	int batchSize = dimGrid.x*dimBlock.x;
	//std::cout << "Starting forward kernel with grid " << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl;
	cudaFuncSetCacheConfig(path_kernel2<float,forward_functor<float> >, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(path_kernel<float,forward_functor<float> >, cudaFuncCachePreferL1);
	for (int offset = 0; offset < (dims+batchSize); offset += batchSize){
		forward_norm_functor<float> functor(projections->get_data_ptr());
		if (exterior_path_lengths != NULL){
			path_kernel2<float,forward_norm_functor<float> ><<< dimGrid, dimBlock >>> (functor,splines->get_data_ptr(),exterior_path_lengths->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
			//forward_kernel2<float><<< dimGrid, dimBlock >>> (image->get_data_ptr(), projections->get_data_ptr(),splines->get_data_ptr(),exterior_path_lengths->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
		}else
			path_kernel<float,forward_norm_functor<float> ><<< dimGrid, dimBlock >>> (functor,splines->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
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
	int threadsPerBlock =std::min(dims,cudaDeviceManager::Instance()->max_blockdim());
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,cudaDeviceManager::Instance()->max_griddim()));
	uint64d3 _dims = from_std_vector<size_t,3>( *(image->get_dimensions().get()) );

	// Invoke kernel
	int batchSize = dimBlock.x*dimGrid.x;

	cudaFuncSetCacheConfig(path_kernel2<float,backward_functor<float> >, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(path_kernel<float,backward_functor<float> >, cudaFuncCachePreferL1);
	for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){
		backward_functor<float> functor(image->get_data_ptr(),projections->get_data_ptr());
		if (exterior_path_lengths != NULL){
		  path_kernel2<float,backward_functor<float> ><<< dimGrid, dimBlock >>> (functor,splines->get_data_ptr(),exterior_path_lengths->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
			//backwards_kernel2<float><<< dimGrid, dimBlock >>> (projections->get_data_ptr(), image->get_data_ptr(),splines->get_data_ptr(),exterior_path_lengths->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
		}else
			path_kernel<float, backward_functor<float> ><<< dimGrid, dimBlock >>> (functor,splines->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
	}

}



template<> void countProtonsPerVoxel<cuNDArray>(cuNDArray<float> * image, cuNDArray<floatd3>* splines, floatd3 phys_dims, cuNDArray<float>* exterior_path_lengths){
	int dims =  splines->get_number_of_elements()/4;
	int threadsPerBlock =std::min(dims,cudaDeviceManager::Instance()->max_blockdim());
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,cudaDeviceManager::Instance()->max_griddim()));
	uint64d3 _dims = from_std_vector<size_t,3>( *(image->get_dimensions().get()) );

	// Invoke kernel
	int batchSize = dimBlock.x*dimGrid.x;

	cudaFuncSetCacheConfig(path_kernel2<float,backward_functor<float> >, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(path_kernel<float,backward_functor<float> >, cudaFuncCachePreferL1);
	for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){
		backward_counting_functor<float> functor(image->get_data_ptr());
		if (exterior_path_lengths != NULL){
		  path_kernel2<float,backward_counting_functor<float> ><<< dimGrid, dimBlock >>> (functor,splines->get_data_ptr(),exterior_path_lengths->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
			//backwards_kernel2<float><<< dimGrid, dimBlock >>> (projections->get_data_ptr(), image->get_data_ptr(),splines->get_data_ptr(),exterior_path_lengths->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
		}else
			path_kernel<float, backward_counting_functor<float> ><<< dimGrid, dimBlock >>> (functor,splines->get_data_ptr(),phys_dims, (vector_td<int,3>)_dims, dims,offset);
	}

}


template<> void countProtonsPerVoxel<hoCuNDArray>(hoCuNDArray<float>* image, hoCuNDArray<floatd3>* splines, floatd3 phys_dims, hoCuNDArray<float>* exterior_path_lengths) {

	cuNDArray<float> cu_image(image);
	CHECK_FOR_CUDA_ERROR();

	size_t floats_per_proton = exterior_path_lengths == NULL ? 13 : 15;
	size_t max_batch_size = calculate_batch_size(floats_per_proton);
	size_t elements = splines->get_number_of_elements()/4;
	size_t offset = 0;

	for (size_t n = 0; n < (elements+max_batch_size-1)/max_batch_size; n++){
		size_t batch_size = std::min(max_batch_size,elements-offset);
		std::vector<size_t> batch_dim;
		batch_dim.push_back(batch_size);
		batch_dim.push_back(4);
		hoCuNDArray<floatd3 > splines_view(&batch_dim,splines->get_data_ptr()+offset*4); // This creates a "view" of splines
		cuNDArray<floatd3 > cu_splines(&splines_view);

		if (exterior_path_lengths != NULL){
			batch_dim.back() = 2;
			hoCuNDArray<float> EPL_view(&batch_dim,exterior_path_lengths->get_data_ptr()+offset*2); //This creates a "view" of projections
			cuNDArray<float> cu_EPL(&EPL_view);
			countProtonsPerVoxel<cuNDArray>(&cu_image,&cu_splines,phys_dims,&cu_EPL);
		} else
			countProtonsPerVoxel<cuNDArray>(&cu_image,&cu_splines,phys_dims);
		CHECK_FOR_CUDA_ERROR();
		offset += batch_size;
	}

	cudaMemcpy(image->get_data_ptr(),cu_image.get_data_ptr(),cu_image.get_number_of_elements()*sizeof(float),cudaMemcpyDeviceToHost);
}


template<> void protonPathNorm<hoCuNDArray>(std::vector<size_t> img_dims,hoCuNDArray<float> * projections, hoCuNDArray<floatd3> * splines, floatd3 phys_dims, hoCuNDArray<float>* exterior_path_lengths){
	size_t floats_per_proton = exterior_path_lengths == NULL ? 13 : 15;
	size_t max_batch_size = calculate_batch_size(floats_per_proton);
	size_t elements = projections->get_number_of_elements();
	size_t offset = 0;

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
			protonPathNorm(img_dims,&cu_projections, &cu_splines,phys_dims,&cu_EPL);
		}else
			protonPathNorm(img_dims,&cu_projections, &cu_splines,phys_dims);

		cudaMemcpy(projections_view.get_data_ptr(),cu_projections.get_data_ptr(),batch_size*sizeof(float),cudaMemcpyDeviceToHost); //Copies back the data to the host
		offset += batch_size;

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

template void pad_nearest<float,1>( cuNDArray<float> *in, cuNDArray<float> *out );
template void pad_nearest<float,2>( cuNDArray<float> *in, cuNDArray<float> *out );
template void pad_nearest<float,3>( cuNDArray<float> *in, cuNDArray<float> *out );
template void pad_nearest<float,4>( cuNDArray<float> *in, cuNDArray<float> *out );

template void pad_nearest<double,1>( cuNDArray<double> *in, cuNDArray<double> *out );
template void pad_nearest<double,2>( cuNDArray<double> *in, cuNDArray<double> *out );
template void pad_nearest<double,3>( cuNDArray<double> *in, cuNDArray<double> *out );
template void pad_nearest<double,4>( cuNDArray<double> *in, cuNDArray<double> *out );

}


