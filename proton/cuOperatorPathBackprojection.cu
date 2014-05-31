#include "cuOperatorPathBackprojection.h"
#include "vector_td_utilities.h"
#include "vector_td_io.h"
#include "cuNDArray_math.h"
#include "cuNDArray_reductions.h"
#include "cuGaussianFilterOperator.h"
#include "check_CUDA.h"

#include <vector>

#include <stdio.h>
#include "hoNDArray_fileio.h"

#include "proton_kernels.cu"

#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_new.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

#define MAX_THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 65535
#define MAX_BLOCKS 4096*4

/*template <typename T> __inline__ __host__ __device__ int sgn(T val)
{
    return (val > T(0)) - (val < T(0));
}*/

template <typename REAL> class greater_than_functor{
public:
	greater_than_functor(REAL val_): val(val_){}
	__inline__ __device__ bool operator()(const REAL &x){
		return (x > val );
	}
	REAL val;
};

template<typename REAL> class tuple_greater_than{
public:
	tuple_greater_than(REAL val_): val(val_){}

	REAL val;
	__inline__ __device__ bool operator()(const thrust::tuple<REAL, vector_td<REAL,12> > & tup){
		const REAL x = thrust::get<0>(tup);
		return (x > val);
	}

};

//For reasons unknown, thrust won't allow specialization here, so...
template<typename REAL> class tuple3_greater_than{
public:
	tuple3_greater_than(REAL val_): val(val_){}

	REAL val;
	__inline__ __device__ bool operator()(const thrust::tuple<REAL, REAL, vector_td<REAL,12> > & tup){
		const REAL x = thrust::get<0>(tup);
		return (x > val);
	}

};


template<class REAL> static void remove_outside_values(cuNDArray<REAL>* projections, cuNDArray<vector_td<REAL,3> >* splines, REAL cutoff){

	typedef typename thrust::device_vector<REAL>::iterator realIterator;
	typedef typename thrust::device_vector<vector_td<REAL,12> >::iterator splinesIterator;
	typedef thrust::tuple<realIterator,splinesIterator> iteratorTuple;
	typedef thrust::zip_iterator<iteratorTuple> ZipIterator;

	size_t new_size = thrust::count_if(projections->begin(),projections->end(),greater_than_functor<REAL>(cutoff));

	//Setup output arrays and make a nice zip iterator
	thrust::device_ptr<REAL> tmp_proj = thrust::device_new<REAL>(new_size);
	thrust::device_ptr<vector_td<REAL,12> > tmp_splines = thrust::device_new<vector_td<REAL,12> >(new_size);
	ZipIterator output(thrust::make_tuple(tmp_proj, tmp_splines));


	std::vector<size_t> dims;
	dims.push_back(splines->get_number_of_elements()/4);
	cuNDArray<vector_td<REAL,12> > splines_view(dims,(vector_td<REAL,12>*)splines->get_data_ptr());

	ZipIterator input_begin(thrust::make_tuple(projections->begin(), splines_view.begin()));
	ZipIterator input_end(thrust::make_tuple(projections->end(), splines_view.end()));

	ZipIterator tmp = thrust::copy_if(input_begin,input_end,output,tuple_greater_than<REAL>(cutoff));

	std::vector<size_t> output_dims;
	output_dims.push_back(new_size);

	std::vector<size_t> output_dims_splines;
	output_dims_splines.push_back(new_size*4);
	*projections = cuNDArray<REAL>(output_dims,thrust::raw_pointer_cast(tmp_proj),true);
	*splines = cuNDArray<vector_td<REAL,3> >(output_dims_splines, (vector_td<REAL,3>*)thrust::raw_pointer_cast(tmp_splines),true);

	CHECK_FOR_CUDA_ERROR();


}


template<class REAL> static void remove_outside_values(cuNDArray<REAL>* projections, cuNDArray<REAL>* uncertainties,cuNDArray<vector_td<REAL,3> >* splines, REAL cutoff){

	typedef typename thrust::device_vector<REAL>::iterator realIterator;
	typedef typename thrust::device_vector<vector_td<REAL,12> >::iterator splinesIterator;
	typedef thrust::tuple<realIterator,realIterator,splinesIterator> iteratorTuple;
	typedef thrust::zip_iterator<iteratorTuple> ZipIterator;

	size_t new_size = thrust::count_if(projections->begin(),projections->end(),greater_than_functor<REAL>(cutoff));

	//Setup output arrays and make a nice zip iterator
	thrust::device_ptr<REAL> tmp_proj = thrust::device_new<REAL>(new_size);
	thrust::device_ptr<REAL> tmp_uncertainties = thrust::device_new<REAL>(new_size);
	thrust::device_ptr<vector_td<REAL,12> > tmp_splines = thrust::device_new<vector_td<REAL,12> >(new_size);
	ZipIterator output(thrust::make_tuple(tmp_proj, tmp_uncertainties,tmp_splines));


	std::vector<size_t> dims;
	dims.push_back(splines->get_number_of_elements()/4);
	cuNDArray<vector_td<REAL,12> > splines_view(dims,(vector_td<REAL,12>*)splines->get_data_ptr());

	ZipIterator input_begin(thrust::make_tuple(projections->begin(),uncertainties->begin(), splines_view.begin()));
	ZipIterator input_end(thrust::make_tuple(projections->end(),uncertainties->end(), splines_view.end()));

	ZipIterator tmp = thrust::copy_if(input_begin,input_end,output,tuple3_greater_than<REAL>(cutoff));

	std::vector<size_t> output_dims;
	output_dims.push_back(new_size);

	std::vector<size_t> output_dims_splines;
	output_dims_splines.push_back(new_size*4);
	*projections = cuNDArray<REAL>(output_dims,thrust::raw_pointer_cast(tmp_proj),true);
	*uncertainties = cuNDArray<REAL>(output_dims,thrust::raw_pointer_cast(tmp_uncertainties),true);
	*splines = cuNDArray<vector_td<REAL,3> >(output_dims_splines, (vector_td<REAL,3>*)thrust::raw_pointer_cast(tmp_splines),true);

	CHECK_FOR_CUDA_ERROR();


}

template<class REAL> void cuOperatorPathBackprojection<REAL>
::mult_M( cuNDArray<REAL>* in_orig, cuNDArray<REAL>* out_orig, bool accumulate ) {
	if( !in_orig || !out_orig){
		throw std::runtime_error( "cuOperatorPathBackprojection: mult_M empty data pointer");
	}
	cuNDArray<REAL>* out = out_orig;
	if (accumulate) out = new cuNDArray<REAL>(out_orig->get_dimensions());
	clear(out);

	cuNDArray<REAL>* in = in_orig;
	if (hull_mask.get()){
		in = new cuNDArray<REAL>(*in_orig);
		*in *= *hull_mask;
	}

	int dims =  out->get_number_of_elements();

	int threadsPerBlock =std::min(dims,MAX_THREADS_PER_BLOCK);
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));
	typename uint64d<3>::Type _dims = from_std_vector<size_t,3>( *(in->get_dimensions().get()) );

	// Invoke kernel
	int batchSize = dimGrid.x*dimBlock.x;
	//std::cout << "Starting forward kernel with grid " << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl;
	cudaFuncSetCacheConfig(forward_kernel<REAL>, cudaFuncCachePreferL1);
	for (int offset = 0; offset < (dims+batchSize); offset += batchSize){
		forward_kernel<REAL><<< dimGrid, dimBlock >>> (in->get_data_ptr(), out->get_data_ptr(),splines->get_data_ptr(),physical_dims, (vector_td<int,3>)_dims, dims,offset);
	}

	cudaDeviceSynchronize();
	CHECK_FOR_CUDA_ERROR();
	if (this->weights.get()){
		*out *= *this->weights;
	}

	if (accumulate){
		*out_orig += *out;
		delete out;
	}
	if (hull_mask.get())
		delete in;

}

template<class REAL> void cuOperatorPathBackprojection<REAL>
::mult_MH( cuNDArray<REAL>* in_orig, cuNDArray<REAL>* out_orig, bool accumulate ) {
	if( !in_orig || !out_orig){
		throw std::runtime_error("cuOperatorPathBackprojection: mult_MH empty data pointer");
	}
	cuNDArray<REAL>* out = out_orig;
	if (accumulate) out = new cuNDArray<REAL>(out_orig->get_dimensions());

	clear(out);

	cuNDArray<REAL>* in = in_orig;
	if (weights.get()){
		in = new cuNDArray<REAL>(*in_orig);
		*in *= *weights;
	}
	int dims =  in->get_number_of_elements();
	int threadsPerBlock =std::min(dims,MAX_THREADS_PER_BLOCK);
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));
	typename uint64d<3>::Type _dims = from_std_vector<size_t,3>( *(out->get_dimensions().get()) );

	// Invoke kernel
	int batchSize = dimBlock.x*dimGrid.x;

	cudaFuncSetCacheConfig(backwards_kernel<REAL>, cudaFuncCachePreferL1);
	for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){
		backwards_kernel<REAL><<< dimGrid, dimBlock >>> (in->get_data_ptr(), out->get_data_ptr(),splines->get_data_ptr(),physical_dims, (vector_td<int,3>)_dims, dims,offset);
	}

	if (hull_mask.get()) *out *= *hull_mask;
	if (accumulate){
		*out_orig += *out;
	}
	if (accumulate) delete out;

	if (weights.get()) delete in;
}

template<class REAL> void cuOperatorPathBackprojection<REAL>
::mult_MH_M( cuNDArray<REAL>* in, cuNDArray<REAL>* out, bool accumulate ) {

	cuNDArray<REAL> tmp;

	std::vector<size_t> tmp_dim = *(splines->get_dimensions().get());
	tmp_dim[0] /= 4;

	tmp.create(&tmp_dim);

	mult_M(in,&tmp);

	mult_MH(&tmp,out);

}

template<class REAL> static float find_percentile(cuNDArray<REAL>* arr,float fraction){

	cuNDArray<REAL> tmp(*arr);
	thrust::sort(tmp.begin(),tmp.end());

	return tmp.at((size_t)(tmp.get_number_of_elements()*fraction));


}
template<class REAL> boost::shared_ptr<cuNDArray<REAL> > cuOperatorPathBackprojection<REAL>
::calc_hull( cuNDArray<REAL>* projections, std::vector<size_t>& img_dims ) {
	if( !projections){
		throw std::runtime_error("cuOperatorPathBackprojection: calc Hull empty data pointer");
	}


	int dims =  projections->get_number_of_elements();
	int threadsPerBlock =std::min(dims,MAX_THREADS_PER_BLOCK);
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));
	typename uint64d<3>::Type _dims = from_std_vector<size_t,3>( img_dims );

	boost::shared_ptr<cuNDArray<REAL> > hull(new cuNDArray<REAL>(img_dims));
	//clear(hull_mask.get());
	fill(hull.get(),REAL(1));
	// Invoke kernel
	int batchSize = dimBlock.x*dimGrid.x;

	cudaFuncSetCacheConfig(space_carver_kernel<REAL>, cudaFuncCachePreferL1);

	cutoff = find_percentile(projections,0.01);
	std::cout << "Cutoff set at: " << cutoff << std::endl;
	for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){
		space_carver_kernel<REAL><<< dimGrid, dimBlock >>> (projections->get_data_ptr(), hull->get_data_ptr(),splines->get_data_ptr(),this->origin,physical_dims, cutoff, (vector_td<int,3>)_dims, dims,offset);
	}

	std::cout << "Hull norm: " << nrm2(hull.get()) << std::endl;
	cuNDArray<REAL> tmp(*hull);
	cuGaussianFilterOperator<REAL,3> gauss;
	gauss.set_sigma(REAL(0.5));
	gauss.mult_M(&tmp,hull.get());

	std::cout << "Hull norm: " << nrm2(hull.get()) << std::endl;
	clamp(hull.get(),REAL(0),REAL(1e-6),REAL(0),REAL(1));
	std::cout << "Hull norm: " << nrm2(hull.get()) << std::endl;

	return hull;

}


template<class REAL> void cuOperatorPathBackprojection<REAL>
::setup(boost::shared_ptr< cuNDArray< vector_td<REAL,3> > > splines,  vector_td<REAL,3> physical_dims,  boost::shared_ptr< cuNDArray< REAL > > projections,  boost::shared_ptr< cuNDArray<REAL> > weights,vector_td<REAL,3> origin, std::vector<size_t> img_dims, bool use_hull, REAL background){
	this->weights=weights;
	setup(splines,physical_dims,projections,origin, img_dims,use_hull,background);
	*projections *= *weights;
}
template<class REAL> void cuOperatorPathBackprojection<REAL>
::setup(boost::shared_ptr< cuNDArray< vector_td<REAL,3> > > splines,  vector_td<REAL,3> physical_dims,boost::shared_ptr< cuNDArray< REAL > > projections, vector_td<REAL,3> origin, std::vector<size_t> img_dims, bool use_hull, REAL background){

	this->splines = splines;
	this->physical_dims = physical_dims;
	this->background = background;
	this->origin = origin;

	int dims = splines->get_number_of_elements()/4;


	if (!this->splines->get_data_ptr()) throw std::runtime_error("Splines data is empty.");
	if (!projections->get_data_ptr()) throw std::runtime_error("Projections data is empty.");
	if (projections->get_number_of_elements() != dims) throw std::runtime_error("Projections data does not match splines.");


	int threadsPerBlock =std::min(dims,MAX_THREADS_PER_BLOCK);
	dim3 dimBlock( threadsPerBlock);

	int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));



	int batchSize = dimGrid.x*dimBlock.x;

	if (use_hull){
		this->hull_mask = calc_hull(projections.get(),img_dims);
		vector_td<int,3> ndims = vector_td<int,3>(from_std_vector<size_t,3>(img_dims));

		if (this->weights.get())
			remove_outside_values(projections.get(),this->weights.get(),this->splines.get(),cutoff);
		else
			remove_outside_values(projections.get(),this->splines.get(),cutoff);
		dims = splines->get_number_of_elements()/4;
		threadsPerBlock =std::min(dims,MAX_THREADS_PER_BLOCK);
		dimBlock = dim3 ( threadsPerBlock);
		totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
		dimGrid =  dim3(std::min(totalBlocksPerGrid,MAX_BLOCKS));
		std::cout << "Minimum element " << min(projections.get()) << std::endl;

		for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){
			crop_splines_hull_kernel<<< dimGrid, dimBlock >>> (this->splines->get_data_ptr(),projections->get_data_ptr(),this->hull_mask->get_data_ptr(),ndims,physical_dims,origin,dims,background,offset);
			//crop_splines_kernel<<< dimGrid, dimBlock >>> (this->splines->get_data_ptr(),projections->get_data_ptr(),physical_dims,origin,dims,background,offset);
			cudaThreadSynchronize();
			CHECK_FOR_CUDA_ERROR();
		}
		std::cout << "Minimum element " << min(projections.get()) << std::endl;
		clamp_min(projections.get(),REAL(0));



	} else {
		for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){

			crop_splines_kernel<<< dimGrid, dimBlock >>> (this->splines->get_data_ptr(),projections->get_data_ptr(),physical_dims,origin,dims,background,offset);
			cudaThreadSynchronize();
			CHECK_FOR_CUDA_ERROR();
		}
	}
	cudaThreadSynchronize();
	CHECK_FOR_CUDA_ERROR();
	if (rescale_dirs){
		for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){
			rescale_directions_kernel<<< dimGrid, dimBlock >>> (this->splines->get_data_ptr(),projections->get_data_ptr(),physical_dims,(int) dims,offset);
			CHECK_FOR_CUDA_ERROR();
		}

	}
	cudaThreadSynchronize();

	for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){
		points_to_coefficients<<< dimGrid, dimBlock >>>(this->splines->get_data_ptr(),dims,offset);
		CHECK_FOR_CUDA_ERROR();
	}

	std::vector<size_t> codom;
	codom.push_back(dims);
	linearOperator<cuNDArray<REAL> >::set_codomain_dimensions(&codom);




}
// Instantiations
template class cuOperatorPathBackprojection<float>;
