#pragma once

#include "hdf5.h"
#include "hdf5_hl.h"
#include <numeric>
#include "hoCuNDArray.h"
#include "cuGaussianFilterOperator.h"
#include "proton_kernels.h"

namespace Gadgetron{

/**
 * Storage class for proton data.
 */
template<template<class> class ARRAY> class protonDataset {

public:
	boost::shared_ptr<ARRAY<vector_td<float,3> > > get_splines(){
		return spline_data;
	}
	boost::shared_ptr<ARRAY<vector_td<float,3> > > get_spline_group(size_t group){
		return spline_arrays[group];
	}

	boost::shared_ptr<ARRAY<float> > get_projections(){
		return projection_data;
	}
	boost::shared_ptr<ARRAY<float> > get_projection_group(size_t group){
		return projection_arrays[group];
	}
	boost::shared_ptr<ARRAY<float> > get_weights(){
		return weights;
	}
	boost::shared_ptr<ARRAY<float> > get_weights_group(size_t group){
		return weight_arrays[group];
	}


	boost::shared_ptr<ARRAY<float> > get_EPL(){
		return exterior_path_lengths;
	}
	boost::shared_ptr<ARRAY<float> > get_EPL_group(size_t group){
		return exterior_path_lengths_arrays[group];
	}

	unsigned int get_number_of_groups(){return projection_arrays.size();}
	float get_angle(size_t group){
		if (angles.size() > 0) return angles[group];
		else return 0.0f;
	}

	bool preprocessed(){ return _preprocessed;}

	/**
	 * Moves the data so that the given origin becomes 0,0,0
	 * @param origin New origin point
	 */
	void move_origin(floatd3 origin);


	/**
	 * Creates a dataset from the hdf5 file
	 * @param filename Hdf5 file containing data
	 * @param load_weights If true, weights will be loaded from the hdf5 file, if present
	 */
	protonDataset(const std::string & filename, bool load_weights=true);

	/**
	 * Creates a dataset from spline and projection arrays
	 * @param projections
	 * @param splines
	 */
	protonDataset(boost::shared_ptr<hoNDArray<float> > projections,boost::shared_ptr<hoNDArray<floatd3> > splines){
		if (projections->get_number_of_elements() != splines->get_number_of_elements()/4) throw std::runtime_error("Number of projections do not match number of splines");
		projection_data = boost::shared_ptr<ARRAY<float> >(new ARRAY<float>(*projections));
		spline_data = boost::shared_ptr<ARRAY<floatd3> >(new ARRAY<floatd3>(*splines));
		spline_arrays = std::vector< boost::shared_ptr<ARRAY<floatd3> > >();
		projection_arrays=std::vector< boost::shared_ptr<ARRAY<float > > >();
		spline_arrays.push_back(spline_data);
		projection_arrays.push_back(projection_data);
		angles = std::vector<float>();
		origin = floatd3(0,0,0);
		_preprocessed = false;
	}

	protonDataset(boost::shared_ptr<hoNDArray<float> > projections,boost::shared_ptr<hoNDArray<floatd3> > splines, boost::shared_ptr<hoNDArray<float> > ho_weights){
		if (projections->get_number_of_elements() != splines->get_number_of_elements()/4) throw std::runtime_error("Number of projections do not match number of splines");

		spline_data = boost::shared_ptr<ARRAY<floatd3> >(new ARRAY<floatd3>(*splines));
		spline_arrays = std::vector< boost::shared_ptr<ARRAY<floatd3> > >();
		spline_arrays.push_back(spline_data);

		projection_data = boost::shared_ptr<ARRAY<float> >(new ARRAY<float>(*projections));
		projection_arrays=std::vector< boost::shared_ptr<ARRAY<float > > >();
		projection_arrays.push_back(projection_data);


		weights = boost::shared_ptr<ARRAY<float> >(new ARRAY<float>(*ho_weights));
		weight_arrays=std::vector< boost::shared_ptr<ARRAY<float > > >();
		weight_arrays.push_back(weights);

		angles = std::vector<float>();
		origin = floatd3(0,0,0);
		_preprocessed = false;
	}

	/***
	 * Calculates the convex hull of the object in the dataset.
	 * @param img_dims Image Output size of the hull
	 * @param physical_dims Physical dimensions of the output image
	 * @param origin If desired image is off-center.
	 * @return An image of 1s and 0s, describing the convex hull of the object.
	 */
	boost::shared_ptr<ARRAY<float> > calc_hull(std::vector<size_t> & img_dims, floatd3 physical_dims );

	/**
	 *
	 * @return The hull calculated during preprocess.
	 */
	boost::shared_ptr<ARRAY<float> > get_hull(){ return _hull;  };
	/**
	 * Does the preprocessing of the data, i.e. calculates the convex hull and moves the data to the edge of the hull.
	 * @param img_dims Image Output size of the hull
	 * @param physical_dims Physical dimensions of the output image
	 * @param origin If desired image is off-center.
	 * @param use_hull Wether to calculate the convex hull or simple use the image boundary instead
	 */
	void preprocess(std::vector<size_t> & img_dims, floatd3 physical_dims, bool use_hull=true, float background=0.0f);

	std::vector<boost::shared_ptr<protonDataset<ARRAY> > > get_subsets(){
		if (!_preprocessed) throw std::runtime_error("Cannot return subsets before preprocessing");

		std::vector<boost::shared_ptr<protonDataset<ARRAY> > > _subsets;

		for (int i = 0; i < this->get_number_of_groups(); i++){
			boost::shared_ptr<protonDataset<ARRAY> > subset(new protonDataset<ARRAY>);
			subset->spline_data = spline_arrays[i];
			subset->projection_data = projection_arrays[i];
			if (weights) subset->weights = weight_arrays[i];
			if (exterior_path_lengths) subset->exterior_path_lengths = exterior_path_lengths_arrays[i];

			subset->spline_arrays.push_back(subset->spline_data);
			subset->projection_arrays.push_back(subset->projection_data);
			if (weights) subset->weight_arrays.push_back(subset->weights);

			subset->_preprocessed = true;
			if (_hull) subset->_hull = _hull;

			_subsets.push_back(subset);
		}


		return _subsets;
	}



	/**
	 * Creates a copy of the input dataset shuffled so that each subset is continuous in memory
	 * @param input Input dataset
	 * @param subsets Number of subsets
	 * @return
	 */
		static boost::shared_ptr<protonDataset<ARRAY> > shuffle_dataset(boost::shared_ptr<protonDataset<ARRAY> > input, unsigned int subsets);





protected:

	/**
	 * Default constructor. Not for public consumption.
	 */
	protonDataset(){};

	struct Spline{
		float x,y,z,x2,y2,z2;
		float dirx,diry,dirz,dirx2,diry2,dirz2;
	};

	/**
	 * Calculates the total number of elements in the groups provided
	 * @param file_id HDF5 file_id for the file containing the data
	 * @param groupnames Vector of strings containing the names of the groups to count
	 * @return Total number of elements
	 * @throws std::runtime_error if the number of projection data does not match the number of splines
	 */
	size_t get_num_elements(hid_t file_id, std::vector<std::string>& groupnames);



	/**
	 * Loads the data from file in the specified groups
	 * @param file_id HDF5 file_id for the file containing the data
	 * @param groupnames Vector of strings containing the names of the groups to count
	 */
	void load_fromtable(hid_t file_id, std::vector<std::string>& groupnames);

	/***
	 * Returns a vector of strings for the paths.
	 * @param path
	 * @param file_id
	 * @return
	 */
	std::vector<std::string> group_paths(std::string path,hid_t file_id);


/**
 * Checks if a given data file has a complete set of weights
 * @param file_id
 * @param groupnames
 * @return
 */
	bool has_weights(hid_t file_id, std::vector<std::string>& groupnames);
	/**
	 * Converts to array of type ARRAY
	 * @param in
	 * @return
	 */
	template<class T> static boost::shared_ptr<ARRAY<T > > to_Array(boost::shared_ptr<cuNDArray<T> > in);
	template<class T> static boost::shared_ptr<ARRAY<T > > to_Array(boost::shared_ptr<hoCuNDArray<T> > in);

	template<class T> static boost::shared_ptr<cuNDArray<T> > to_cuNDArray(boost::shared_ptr<ARRAY<T> > in);
	template<class T> static void copy_to_Array(boost::shared_ptr<cuNDArray<T> > in,boost::shared_ptr<ARRAY<T> > out );


	boost::shared_ptr<ARRAY<vector_td<float,3> > > spline_data;
	boost::shared_ptr<ARRAY<float > > projection_data;
	boost::shared_ptr<ARRAY<float > > weights;
	std::vector<float> angles;

	floatd3 origin;

	//These two vectors contain views of the original data. This way we can pass around individual projections and the complete dataset. Just for fun.
	std::vector< boost::shared_ptr<ARRAY<float > > > projection_arrays;
	std::vector< boost::shared_ptr<ARRAY<float > > > weight_arrays;
	std::vector< boost::shared_ptr<ARRAY<vector_td<float,3> > > > spline_arrays;

	boost::shared_ptr<ARRAY<float> > exterior_path_lengths;
	std::vector< boost::shared_ptr<ARRAY<float > > > exterior_path_lengths_arrays;
	bool _preprocessed;

	boost::shared_ptr<ARRAY<float> > _hull;

	static void rotateSplines(std::vector< boost::shared_ptr<ARRAY<vector_td<float,3> > > >  &, std::vector<float> & );
	void crop_splines(std::vector<size_t> & img_dims, floatd3 physical_dims, float background = 0.0f);


	/**
	 * Based on the current hull, calculates the length of the linear segment of the curves
	 */
	static boost::shared_ptr<ARRAY<float> > calc_exterior_path_lengths(boost::shared_ptr<ARRAY<vector_td<float,3> > > splines, boost::shared_ptr<ARRAY<float> > hull, floatd3 physical_dims	);



};



//Trick to do partial function specialisation.
template<> template<class T> inline boost::shared_ptr<cuNDArray<T> > protonDataset<cuNDArray>::to_Array(boost::shared_ptr<cuNDArray<T> > in){ return in;}
template<> template<class T> inline boost::shared_ptr<cuNDArray<T> > protonDataset<cuNDArray>::to_Array(boost::shared_ptr<hoCuNDArray<T> > in){
	return boost::shared_ptr<cuNDArray<T> >(new cuNDArray<T>(*in));
}

template<> template<class T> inline boost::shared_ptr<hoCuNDArray<T> > protonDataset<hoCuNDArray>::to_Array(boost::shared_ptr<hoCuNDArray<T> > in){ return in;}
template<> template<class T> inline boost::shared_ptr<hoCuNDArray<T> > protonDataset<hoCuNDArray>::to_Array(boost::shared_ptr<cuNDArray<T> > in){
	boost::shared_ptr<hoCuNDArray<T> > result(new hoCuNDArray<T>(in->get_dimensions()));
	in->to_host(result.get());
	return result;
}

template<> template<class T> inline boost::shared_ptr<cuNDArray<T> > protonDataset<cuNDArray>::to_cuNDArray(boost::shared_ptr<cuNDArray<T> > in){ return in;}
template<> template<class T> inline boost::shared_ptr<cuNDArray<T> > protonDataset<hoCuNDArray>::to_cuNDArray(boost::shared_ptr<hoCuNDArray<T> > in){
	return boost::shared_ptr<cuNDArray<T> >(new cuNDArray<T>(*in));
}

template<> template<class T> void protonDataset<cuNDArray>::copy_to_Array(boost::shared_ptr<cuNDArray<T> > in,boost::shared_ptr<cuNDArray<T> > out ){
	if (in != out)
		*out = *in;
}

template<> template<class T> void protonDataset<hoCuNDArray>::copy_to_Array(boost::shared_ptr<cuNDArray<T> > in,boost::shared_ptr<hoCuNDArray<T> > out ){
	in->to_host(out.get());
}


}
