#include "protonDataset.h"

#include <string>
#include <vector>
#include "proton_kernels.h"
#include "proton_utils.h"

#include <thrust/sort.h>
#include "vector_td_utilities.h"
#include <boost/math/constants/constants.hpp>
#include "cuNDArray_math.h"
#include "cuNDArray_reductions.h"
#include "hoCuNDArray_math.h"
#include "hoNDArray_reductions.h"

#include <functional>
#include <algorithm>
#define MAX_THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 65535
#define MAX_BLOCKS 4096*4

namespace Gadgetron{

static size_t calculate_batch_size(){

	int mem_per_proton = 14*sizeof(float); // Need 12 REALS for the splines and 1 for the projection an possibly 1 for weights
	size_t free;
	size_t total;

	int res = cudaMemGetInfo(&free,&total);
	return 1024*1024*(free/(1024*1024*mem_per_proton)); //Divisons by 1024*1024 to ensure MB batch size
}

template<template<class> class ARRAY> protonDataset<ARRAY>::protonDataset(const std::string & filename, bool load_weights){
	hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id < 0) throw std::runtime_error("Failed to open specified hdf5 file.");

	std::vector<std::string> groupnames = group_paths("/",file_id);
	size_t num_element = get_num_elements(file_id,groupnames);
	std::vector<size_t> dims;
	dims.push_back(num_element);

	std::vector<size_t> spline_dims;
	spline_dims.push_back(4);
	spline_dims.push_back(num_element);

	spline_data = boost::shared_ptr<ARRAY<vector_td<float,3> > >(new ARRAY<vector_td<float,3> >(spline_dims));
	projection_data =boost::shared_ptr<ARRAY<float > >(new ARRAY<float>(dims));

	spline_arrays = std::vector< boost::shared_ptr<ARRAY<vector_td<float,3> > > >();
	projection_arrays=std::vector< boost::shared_ptr<ARRAY<float > > >();

	if (has_weights(file_id,groupnames) && load_weights){
		std::cout << "Loading weights" << std::endl;
		weights =boost::shared_ptr<ARRAY<float > >(new ARRAY<float>(dims));
		weight_arrays =std::vector< boost::shared_ptr<ARRAY<float > > >();

	}

	angles = std::vector<float>();
	load_fromtable(file_id,groupnames);

	if (!angles.empty()){
		std::vector<float> neg_angles(angles.size());
		std::transform(angles.begin(),angles.end(),neg_angles.begin(),std::negate<float>());
		rotateSplines(spline_arrays,neg_angles);
	}
}


template<template<class> class ARRAY> void protonDataset<ARRAY>::preprocess(std::vector<size_t> & img_dims, floatd3 physical_dims, bool use_hull, float background){

	crop_splines(img_dims,physical_dims,background);
	if (use_hull){
		_hull = calc_hull(img_dims,physical_dims);

		exterior_path_lengths = calc_exterior_path_lengths(this->spline_data,_hull,physical_dims);

		exterior_path_lengths_arrays = std::vector< boost::shared_ptr<ARRAY<float > > >();
		//Split space lengths
		size_t offset = 0;
		for (size_t i = 0; i < projection_arrays.size(); i++){
			std::vector<size_t> space_length_dim = *projection_arrays[i]->get_dimensions();
			space_length_dim.push_back(2);
			boost::shared_ptr<ARRAY<float > > EPL_view(new ARRAY<float>(space_length_dim,exterior_path_lengths->get_data_ptr()+offset));
			exterior_path_lengths_arrays.push_back(EPL_view);
			offset += EPL_view->get_number_of_elements();
		}
	}
	_preprocessed = true;
}

template<> void protonDataset<cuNDArray>::crop_splines(std::vector<size_t> & img_dims, floatd3 physical_dims, float background){
	unsigned int dims = spline_data->get_number_of_elements()/4;
	unsigned int threadsPerBlock =std::min(dims, (unsigned int) MAX_THREADS_PER_BLOCK);
	dim3 dimBlock = dim3 ( threadsPerBlock);
	unsigned int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid =  dim3(std::min(totalBlocksPerGrid, (unsigned int)MAX_BLOCKS));

	unsigned int batchSize = dimGrid.x*dimBlock.x;
	vector_td<int,3> ndims = vector_td<int,3>(from_std_vector<size_t,3>(img_dims));

	for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){

		crop_splines_kernel<<< dimGrid, dimBlock >>> (spline_data->get_data_ptr(),projection_data->get_data_ptr(),physical_dims,dims,background,offset);
		//cudaThreadSynchronize();

	}
	CHECK_FOR_CUDA_ERROR();

	for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){
		rescale_directions_kernel<<< dimGrid, dimBlock >>>(spline_data->get_data_ptr(),projection_data->get_data_ptr(),physical_dims,(int) dims,offset);
	}
	CHECK_FOR_CUDA_ERROR();
}

template<> void protonDataset<hoCuNDArray>::crop_splines(std::vector<size_t> & img_dims, floatd3 physical_dims, float background){
	size_t max_batch_size = calculate_batch_size()/2;


	size_t elements = spline_data->get_number_of_elements()/4;
	size_t offset = 0;


	for (size_t n = 0; n < (elements-1)/max_batch_size+1; n++){
		std::cout << "Cropping spline batch " << n << std::endl;

		size_t batch_size = std::min(max_batch_size,elements-offset);
		std::vector<size_t> batch_dim;
		batch_dim.push_back(batch_size);

		hoCuNDArray<float> projections_view(&batch_dim,projection_data->get_data_ptr()+offset); //This creates a "view" of out
		cuNDArray<float> projections_dev(&projections_view);

		CHECK_FOR_CUDA_ERROR();
		batch_dim.push_back(4);
		hoCuNDArray<floatd3 > splines_view(&batch_dim,spline_data->get_data_ptr()+offset*4); // This creates a "view" of splines
		cuNDArray<floatd3 > splines_dev(&splines_view);
		CHECK_FOR_CUDA_ERROR();

		int threadsPerBlock = std::min((int)batch_size, MAX_THREADS_PER_BLOCK);
		dim3 dimBlock( threadsPerBlock);
		int totalBlocksPerGrid = (batch_size-1)/MAX_THREADS_PER_BLOCK+1;
		dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));
		vector_td<int,3> ndims = vector_td<int,3>(from_std_vector<size_t,3>(img_dims));
		// Invoke kernel
		int offset_k = 0;

		for (unsigned int i = 0; i <= (totalBlocksPerGrid-1)/MAX_BLOCKS+1; i++){
			crop_splines_kernel<<< dimGrid, dimBlock >>> (splines_dev.get_data_ptr(),projections_dev.get_data_ptr(),physical_dims, batch_size,background,offset_k);
			offset_k += dimGrid.x*dimBlock.x;
		}

		offset_k = 0;
		for (unsigned int i = 0; i <= (totalBlocksPerGrid-1)/MAX_BLOCKS+1; i++){
			rescale_directions_kernel<<< dimGrid, dimBlock >>>(splines_dev.get_data_ptr(),projections_dev.get_data_ptr(),physical_dims, batch_size,offset_k);
			offset_k += dimGrid.x*dimBlock.x;
		}
		cudaMemcpy(splines_view.get_data_ptr(),splines_dev.get_data_ptr(),batch_size*4*sizeof(floatd3),cudaMemcpyDeviceToHost);
		cudaMemcpy(projections_view.get_data_ptr(),projections_dev.get_data_ptr(),batch_size*sizeof(float),cudaMemcpyDeviceToHost);
		offset += batch_size;

	}

}

template<template<class> class ARRAY> void protonDataset<ARRAY>::rotateSplines(std::vector< boost::shared_ptr<ARRAY<vector_td<float,3> > > >  & splines, std::vector<float> & angles){

	for (int i = 0; i < splines.size(); i++){
		boost::shared_ptr<cuNDArray<vector_td<float,3> > > cu_splines = to_cuNDArray(splines[i]);
		rotate_splines(cu_splines.get(),angles[i]);
		copy_to_Array(cu_splines,splines[i]);
	}
}
template<template<class> class ARRAY> bool protonDataset<ARRAY>::has_weights(hid_t file_id, std::vector<std::string>& groupnames){


	for (size_t i = 0; i < groupnames.size(); i++){
		hid_t group = H5Gopen1(file_id,groupnames[i].c_str());
		herr_t err=H5LTfind_dataset(group,"weights");
		H5Gclose(group);
		if (!err) return false;
	}
	return true;
}
template<template<class> class ARRAY> void protonDataset<ARRAY>::load_fromtable(hid_t file_id, std::vector<std::string>& groupnames){

	const size_t dst_sizes[12] = { sizeof(float) ,sizeof(float), sizeof(float),
			sizeof(float) ,sizeof(float), sizeof(float),
			sizeof(float) ,sizeof(float), sizeof(float),
			sizeof(float) ,sizeof(float), sizeof(float)};
	const size_t dst_offset[12] = { HOFFSET( Spline, x ),HOFFSET( Spline, y),HOFFSET( Spline, z ),
			HOFFSET( Spline, x2 ),HOFFSET( Spline, y2),HOFFSET( Spline, z2 ),
			HOFFSET( Spline, dirx ),HOFFSET( Spline, diry ),HOFFSET( Spline, dirz ),
			HOFFSET( Spline, dirx2 ),HOFFSET( Spline, diry2 ),HOFFSET( Spline, dirz2 )};

	std::string splines_name = "splines";



	const size_t float_size[1] = {sizeof(float) };
	const size_t float_offset[1] = {0};
	const std::string projections_name = "projections";
	const std::string weights_name = "weights";

	//hid_t strtype;                     /* Datatype ID */
	//herr_t status;


	size_t offset = 0;

for (int i = 0; i < groupnames.size(); i++){
		hsize_t nfields,nrecords;
		herr_t err = H5TBget_table_info (file_id, (groupnames[i]+splines_name).c_str(), &nfields, &nrecords );
		if (err < 0) throw std::runtime_error("Illegal hdf5 dataset provided");
		std::vector<size_t> spline_dims;
		spline_dims.push_back(4);
		spline_dims.push_back(nrecords);
		boost::shared_ptr<ARRAY<vector_td<float,3> > > splines(new ARRAY<vector_td<float,3> >(spline_dims,spline_data->get_data_ptr()+offset*4));
		{
			hoCuNDArray<vector_td<float,3> >  tmp_splines(spline_dims);
			err = H5TBread_table (file_id, (groupnames[i]+splines_name).c_str(), sizeof(Spline),  dst_offset, dst_sizes,  tmp_splines.get_data_ptr());
			if (err < 0) throw std::runtime_error("Unable to read splines from hdf5 file");
			*splines = tmp_splines;
		}
		spline_arrays.push_back(splines);


		std::vector<size_t> proj_dims;
		proj_dims.push_back(nrecords);

		boost::shared_ptr<ARRAY<float > > projections(new ARRAY<float >(proj_dims,projection_data->get_data_ptr()+offset));
		{
			hoCuNDArray<float > tmp_proj(proj_dims);
			err = H5TBread_table (file_id, (groupnames[i]+projections_name).c_str(), sizeof(float),  float_offset, float_size,  tmp_proj.get_data_ptr());
			if (err < 0) throw std::runtime_error("Unable to read projections from hdf5 file");
			*projections = tmp_proj;
		}

		projection_arrays.push_back(projections);

		if (weights){
			boost::shared_ptr<ARRAY<float > > local_weights(new ARRAY<float >(proj_dims,weights->get_data_ptr()+offset));
			hoCuNDArray<float > tmp_weights(proj_dims);
			err = H5TBread_table (file_id, (groupnames[i]+projections_name).c_str(), sizeof(float),  float_offset, float_size,  tmp_weights.get_data_ptr());
			if (err < 0) throw std::runtime_error("Unable to read projections from hdf5 file");
			reciprocal_inplace(&tmp_weights);
			*local_weights= tmp_weights;
			weight_arrays.push_back(local_weights);
		}

		offset += nrecords;


		float angle;
		hid_t group = H5Gopen1(file_id,groupnames[i].c_str());
		err=H5LTfind_attribute(group,"angle");
		if (err){
			err = H5LTget_attribute_float(file_id,groupnames[i].c_str(),"angle",&angle);
			angles.push_back(angle*boost::math::constants::pi<float>()/180);
		}
		H5Gclose(group);

	}
	for (int i = 0; i < angles.size(); i++)
		std::cout << angles[i] << std::endl;
}


template<template<class> class ARRAY>	std::vector<std::string> protonDataset<ARRAY>::group_paths(std::string path,hid_t file_id){

	char node[2048];
	hsize_t nobj;

	hid_t group_id = H5Gopen1(file_id,path.c_str());

	H5Gget_num_objs(group_id, &nobj);

	std::vector<std::string> result;
	for(hsize_t i =0; i < nobj; i++){
		H5Gget_objname_by_idx(group_id, i,
				node, sizeof(node) );
		std::string nodestr = std::string(path).append(node).append("/");
		int otype =  H5Gget_objtype_by_idx(group_id, i );
		switch(otype){
		case H5G_GROUP:
			//cout << nodestr << " is a GROUP" << endl;
			result.push_back(nodestr);
			break;
		}

	}
	H5Gclose(group_id);
	return result;

}






template<template<class> class ARRAY> size_t protonDataset<ARRAY>::get_num_elements(hid_t file_id, std::vector<std::string>& groupnames){
	std::string projections_name = "projections";
	std::string splines_name = "splines";
	size_t total_elements = 0;

	for (int i = 0; i < groupnames.size(); i++){
		hsize_t nfields,nrecords,nrecords2;
		herr_t err = H5TBget_table_info (file_id, (groupnames[i]+projections_name).c_str(), &nfields, &nrecords );
		err = H5TBget_table_info (file_id, (groupnames[i]+splines_name).c_str(), &nfields, &nrecords2 );

		if (nrecords != nrecords2) throw std::runtime_error("Illegal data file: number of splines and projections do not match");
		total_elements += nrecords;
	}
	return total_elements;
}

static float find_percentile(cuNDArray<float>* arr,float fraction){
	cuNDArray<float> tmp(*arr);
	thrust::sort(tmp.begin(),tmp.end());
	return tmp.at((size_t)(tmp.get_number_of_elements()*fraction));
}

template<> boost::shared_ptr<hoCuNDArray<float> > protonDataset<hoCuNDArray>::calc_exterior_path_lengths(boost::shared_ptr<hoCuNDArray<vector_td<float,3> > > splines, boost::shared_ptr<hoCuNDArray<float> > hull, floatd3 physical_dims){

	size_t max_batch_size = calculate_batch_size();


	size_t elements = splines->get_number_of_elements()/4;
	size_t offset = 0;
	cuNDArray<float> cu_hull(*hull);

	std::vector<size_t> length_dims;
	length_dims.push_back(elements);
	length_dims.push_back(2);
	boost::shared_ptr<hoCuNDArray<float> > space_lengths(new hoCuNDArray<float>(length_dims));

	vector_td<int,3> ndims = vector_td<int,3>(from_std_vector<size_t,3>(*hull->get_dimensions()));

	for (size_t n = 0; n < (elements-1)/max_batch_size+1; n++){

		size_t batch_size = std::min(max_batch_size,elements-offset);
		std::vector<size_t> batch_dim;
		batch_dim.push_back(batch_size);
		batch_dim.push_back(2);


		hoCuNDArray<float> space_lengths_view(batch_dim,space_lengths->get_data_ptr()+offset*2);
		cuNDArray<float> space_lengths_dev(&space_lengths_view);

		CHECK_FOR_CUDA_ERROR();
		batch_dim.back() = 4;
		hoCuNDArray<floatd3 > splines_view(&batch_dim,splines->get_data_ptr()+offset*4); // This creates a "view" of splines
		cuNDArray<floatd3 > splines_dev(&splines_view);
		CHECK_FOR_CUDA_ERROR();

		int threadsPerBlock = std::min((int)batch_size, MAX_THREADS_PER_BLOCK);
		dim3 dimBlock( threadsPerBlock);
		int totalBlocksPerGrid = (batch_size-1)/MAX_THREADS_PER_BLOCK+1;
		dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));

		// Invoke kernel
		int offset_k = 0;

		for (unsigned int i = 0; i <= (totalBlocksPerGrid-1)/MAX_BLOCKS+1; i++){
			calc_spaceLengths_kernel<<< dimGrid, dimBlock >>>(splines_dev.get_data_ptr(),space_lengths_dev.get_data_ptr(),cu_hull.get_data_ptr(),ndims,physical_dims,batch_dim[0],offset_k);

			offset_k += dimGrid.x*dimBlock.x;
			//crop_splines_kernel<<< dimGrid, dimBlock >>> (this->splines->get_data_ptr(),projections->get_data_ptr(),physical_dims,origin,dims,background,offset);
			cudaThreadSynchronize();
			CHECK_FOR_CUDA_ERROR();
		}

		cudaMemcpy(space_lengths->get_data_ptr()+offset*2,space_lengths_dev.get_data_ptr(),batch_size*sizeof(float),cudaMemcpyDeviceToHost);
		offset += batch_size;


	}

	return space_lengths;
}


template<> boost::shared_ptr<cuNDArray<float> > protonDataset<cuNDArray>::calc_exterior_path_lengths(boost::shared_ptr<cuNDArray<vector_td<float,3> > > splines, boost::shared_ptr<cuNDArray<float> > hull, floatd3 physical_dims){

	size_t elements = splines->get_number_of_elements()/4;

	std::vector<size_t> length_dims;
	length_dims.push_back(elements);
	length_dims.push_back(2);
	boost::shared_ptr<cuNDArray<float> > space_lengths(new cuNDArray<float>(length_dims));

	//clear(space_lengths.get());
	vector_td<int,3> ndims = vector_td<int,3>(from_std_vector<size_t,3>(*hull->get_dimensions()));


	int threadsPerBlock = std::min((int)elements, MAX_THREADS_PER_BLOCK);
	dim3 dimBlock( threadsPerBlock);
	int totalBlocksPerGrid = (elements-1)/MAX_THREADS_PER_BLOCK+1;
	dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));
	// Invoke kernel
	int offset_k = 0;
	for (unsigned int i = 0; i <= (totalBlocksPerGrid-1)/MAX_BLOCKS+1; i++){
		calc_spaceLengths_kernel<<< dimGrid, dimBlock >>>(splines->get_data_ptr(),space_lengths->get_data_ptr(),hull->get_data_ptr(),ndims,physical_dims,elements,offset_k);

		offset_k += dimGrid.x*dimBlock.x;
		//crop_splines_kernel<<< dimGrid, dimBlock >>> (this->splines->get_data_ptr(),projections->get_data_ptr(),physical_dims,origin,dims,background,offset);
		cudaThreadSynchronize();
		CHECK_FOR_CUDA_ERROR();
	}

	std::cout << "Exterior path length mean: " << asum(space_lengths.get())/space_lengths->get_number_of_bytes() << std::endl;
	std::cout << "Max: " << max(space_lengths.get()) << std::endl;
	return space_lengths;
}

template<template<class> class ARRAY>  boost::shared_ptr<ARRAY<float> > protonDataset<ARRAY>::calc_hull(std::vector<size_t> & img_dims, floatd3 physical_dims){


	boost::shared_ptr<cuNDArray<float> > cuHull(new cuNDArray<float>(img_dims));
	//clear(hull_mask.get());
	fill(cuHull.get(),1.0f);
	// Invoke kernel


	unsigned int ngroups = spline_arrays.size();
	for (unsigned int group = 0; group < ngroups; group++){
		boost::shared_ptr< ARRAY<float> > projections = projection_arrays[group];
		int dims =  projections->get_number_of_elements();
		int threadsPerBlock =std::min(dims,MAX_THREADS_PER_BLOCK);
		dim3 dimBlock( threadsPerBlock);
		int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
		dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));
		uint64d3 _dims = from_std_vector<size_t,3>( img_dims );
		int batchSize = dimBlock.x*dimGrid.x;

		//cudaFuncSetCacheConfig(space_carver_kernel<REAL>, cudaFuncCachePreferL1);



		boost::shared_ptr< cuNDArray<vector_td<float,3> > >  cu_splines=to_cuNDArray(spline_arrays[group]);
		boost::shared_ptr< cuNDArray<float > >  cu_projections=to_cuNDArray(projection_arrays[group]);

		float cutoff = find_percentile(cu_projections.get(),0.001f);
		for (int offset = 0; offset <  (dims+batchSize); offset += batchSize){
			space_carver_kernel<float><<< dimGrid, dimBlock >>> (cu_projections->get_data_ptr(), cuHull->get_data_ptr(),cu_splines->get_data_ptr(),physical_dims, cutoff, (vector_td<int,3>)_dims, dims,offset);
		}

	}
	std::cout << "Hull norm: " << nrm2(cuHull.get()) << std::endl;
	cuNDArray<float> tmp(*cuHull);
	cuGaussianFilterOperator<float,3> gauss;
	gauss.set_sigma(2.0f);
	gauss.mult_M(&tmp,cuHull.get());

	std::cout << "Hull norm: " << nrm2(cuHull.get()) << std::endl;
	clamp(cuHull.get(),0.0f,1e-6f,0.0f,1.0f);
	std::cout << "Hull norm: " << nrm2(cuHull.get()) << std::endl;

	boost::shared_ptr<ARRAY<float> > hull = to_Array(cuHull);

	return hull;
}

template<>	void protonDataset<cuNDArray>::move_origin(floatd3 new_origin){
	unsigned int dims = spline_data->get_number_of_elements()/4;
	unsigned int threadsPerBlock =std::min(dims, (unsigned int) MAX_THREADS_PER_BLOCK);
	dim3 dimBlock = dim3 ( threadsPerBlock);
	unsigned int totalBlocksPerGrid = (dims+threadsPerBlock-1)/threadsPerBlock;
	dim3 dimGrid =  dim3(std::min(totalBlocksPerGrid, (unsigned int)MAX_BLOCKS));

	unsigned int batchSize = dimGrid.x*dimBlock.x;

	for (unsigned int offset = 0; offset <  (dims+batchSize); offset += batchSize){
		move_origin_kernel<<< dimGrid, dimBlock >>> (spline_data->get_data_ptr(),new_origin,dims,offset);
		//crop_splines_kernel<<< dimGrid, dimBlock >>> (this->splines->get_data_ptr(),projections->get_data_ptr(),physical_dims,origin,dims,background,offset);
		//cudaThreadSynchronize();

	}
	origin += new_origin;

}


template<>	void protonDataset<hoCuNDArray>::move_origin(floatd3 new_origin){
	size_t max_batch_size = calculate_batch_size();


	size_t elements = spline_data->get_number_of_elements()/4;
	size_t offset = 0;
	for (size_t n = 0; n < (elements-1)/max_batch_size+1; n++){

		size_t batch_size = std::min(max_batch_size,elements-offset);
		std::vector<size_t> batch_dim;
		batch_dim.push_back(batch_size);
		batch_dim.push_back(4);

		hoCuNDArray<floatd3 > splines_view(&batch_dim,spline_data->get_data_ptr()+offset*4); // This creates a "view" of splines
		cuNDArray<floatd3 > splines_dev(&splines_view);
		CHECK_FOR_CUDA_ERROR();

		int threadsPerBlock = std::min((int)batch_size, MAX_THREADS_PER_BLOCK);
		dim3 dimBlock( threadsPerBlock);
		int totalBlocksPerGrid = (batch_size-1)/MAX_THREADS_PER_BLOCK+1;
		dim3 dimGrid(std::min(totalBlocksPerGrid,MAX_BLOCKS));

		// Invoke kernel


		unsigned int offset_k = 0;
		for (unsigned int i = 0; i <= (totalBlocksPerGrid-1)/MAX_BLOCKS+1; i++){
			move_origin_kernel<<< dimGrid, dimBlock >>> (splines_dev.get_data_ptr(),new_origin, batch_size,offset_k);
			offset_k += dimGrid.x*dimBlock.x;
		}
		cudaMemcpy(splines_view.get_data_ptr(),splines_dev.get_data_ptr(),batch_size*4*sizeof(floatd3),cudaMemcpyDeviceToHost);


	}
	origin += new_origin;
}

template<template<class> class ARRAY> boost::shared_ptr<protonDataset<ARRAY> > protonDataset<ARRAY>::shuffle_dataset(boost::shared_ptr<protonDataset<ARRAY> > input, unsigned int subsets){
	boost::shared_ptr<protonDataset<ARRAY> > output(new protonDataset<ARRAY>);
	output->spline_data = boost::shared_ptr<ARRAY<floatd3> >(new ARRAY<floatd3>(input->spline_data->get_dimensions()));
	output->projection_data = boost::shared_ptr<ARRAY<float> >(new ARRAY<float>(input->projection_data->get_dimensions()));
	if (input->weights) output->weights = boost::shared_ptr<ARRAY<float> >(new ARRAY<float>(input->weights->get_dimensions()));

	//Calculate size of individual subsets
	std::vector<std::vector<size_t> > subset_dimensions(subsets, std::vector<size_t>(1) );

	for (unsigned int group = 0; group < input->get_number_of_groups(); group++){
		size_t nrecords = input->projection_arrays[group]->get_number_of_elements();
		size_t extra = nrecords%subsets;
		for (unsigned int subset =0; subset < subsets; subset++){
			subset_dimensions[subset][0] += nrecords/subsets;
			if (subset < extra) subset_dimensions[subset][0] += 1;
		}
	}

	//Create subsets
	output->spline_arrays = std::vector< boost::shared_ptr<ARRAY<floatd3 > > >();
	output->projection_arrays = std::vector< boost::shared_ptr<ARRAY<float > > >();
	if (input->weights) output->weight_arrays = std::vector< boost::shared_ptr<ARRAY<float > > >();

	size_t offset = 0;
	for (unsigned int i =0; i < subsets; i++){
		{
			boost::shared_ptr<ARRAY<float> > proj_subset(new ARRAY<float>(subset_dimensions[i],output->projection_data->get_data_ptr()+offset));
			output->projection_arrays.push_back(proj_subset);
		}
		if (input->weights){
			boost::shared_ptr<ARRAY<float> > weight_subset(new ARRAY<float>(subset_dimensions[i],output->weights->get_data_ptr()+offset));
			output->weight_arrays.push_back(weight_subset);
		}
		{
			std::vector<size_t> spline_dim = subset_dimensions[i];
			spline_dim.push_back(4);
			boost::shared_ptr<ARRAY<floatd3> > spline_subset(new ARRAY<floatd3>(spline_dim,output->spline_data->get_data_ptr()+offset*4));
			output->spline_arrays.push_back(spline_subset);
		}
		offset += subset_dimensions[i][0];
	}

	//Fill data into subsets
	std::vector<float* > proj_ptrs;
	std::vector<floatd3* > spline_ptrs;
	std::vector<float* > weight_ptrs;
	for (int i = 0; i < subsets; i++){
		proj_ptrs.push_back(output->projection_arrays[i]->get_data_ptr());
		spline_ptrs.push_back(output->spline_arrays[i]->get_data_ptr());
		if (input->weights) weight_ptrs.push_back(output->weight_arrays[i]->get_data_ptr());
	}

	for (unsigned int group = 0; group < input->get_number_of_groups(); group++){

		size_t nrecords = input->projection_arrays[group]->get_number_of_elements();
		size_t extra = nrecords%subsets;
		size_t offset =0;
		for (unsigned int i =0; i < subsets; i++){
			size_t batch_size = nrecords/subsets;
			if (i < extra) batch_size += 1;
			std::vector<size_t> dims(1,batch_size);
			{
				ARRAY<float> proj_view(dims,proj_ptrs[i]);
				ARRAY<float> in_proj_view(dims,input->projection_arrays[group]->get_data_ptr()+offset);
				proj_view = in_proj_view;
				proj_ptrs[i] += batch_size;
			}

			if (input->weights){
				ARRAY<float> weight_view(dims,weight_ptrs[i]);
				ARRAY<float> in_weight_view(dims,input->weight_arrays[group]->get_data_ptr()+offset);
				weight_view = in_weight_view;
				weight_ptrs[i] += batch_size;
			}
			dims.push_back(4);
			{
				ARRAY<floatd3> spline_view(dims,spline_ptrs[i]);
				ARRAY<floatd3> in_spline_view(dims,input->spline_arrays[group]->get_data_ptr()+offset*4);
				spline_view = in_spline_view;
				spline_ptrs[i] += batch_size*4;
			}
			offset += batch_size;
		}

	}
	return output;

}

template class protonDataset<cuNDArray>;
template class protonDataset<hoCuNDArray>;



}
