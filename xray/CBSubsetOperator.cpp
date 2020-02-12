/*
 * CBSubsetOperator.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: u051747
 */

#include "CBSubsetOperator.h"
#include <boost/make_shared.hpp>
#include <map>
#include <utility>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include "cuNDArray_math.h"

#include "conebeam_projection.h"
using namespace Gadgetron;

template<template<class> class ARRAY> Gadgetron::CBSubsetOperator<ARRAY>::CBSubsetOperator(int subsets): subsetOperator<ARRAY<float>>(subsets) {
	for (int i = 0; i < subsets; i++)
		operators.push_back(boost::make_shared<typename conebeamProjectionOperator<ARRAY>::type>());
}

template<template<class> class ARRAY> Gadgetron::CBSubsetOperator<ARRAY>::~CBSubsetOperator() {
}

template<template<class> class ARRAY> void Gadgetron::CBSubsetOperator<ARRAY>::mult_M(ARRAY<float>* image,
		ARRAY<float>* projections, int subset, bool accumulate) {
	operators[subset]->mult_M(image,projections,accumulate);

}

template<template<class> class ARRAY> void Gadgetron::CBSubsetOperator<ARRAY>::mult_MH(ARRAY<float>* projections,
		ARRAY<float>* image, int subset, bool accumulate) {
	operators[subset]->mult_MH(projections,image,accumulate);
}

template<template<class> class ARRAY> void Gadgetron::CBSubsetOperator<ARRAY>::mult_MH_M(ARRAY<float>* in,
		ARRAY<float>* out, int subset, bool accumulate) {
	operators[subset]->mult_MH_M(in,out,accumulate);
}


template<template<class> class ARRAY> void Gadgetron::CBSubsetOperator<ARRAY>::offset_correct(ARRAY<float>* projections){
	auto projection_sets = this->projection_subsets(projections);
	for (auto subset = 0u; subset < this->number_of_subsets; subset++){
		if (operators[subset]->get_use_offset_correction())
			operators[subset]->offset_correct(projection_sets[subset].get());
	}
}

template<template<class> class ARRAY> void Gadgetron::CBSubsetOperator<ARRAY>::setup(
		boost::shared_ptr<CBCT_acquisition> acq, boost::shared_ptr<CBCT_binning> binning, floatd3 is_dims_in_mm) {

	auto geometry = acq->get_geometry();
	auto all_angles = geometry->get_angles();
	auto all_offsets = geometry->get_offsets();
	//Set smallest angle to 0.
	float min_value = *std::min_element(all_angles.begin(),all_angles.end() );
	transform(all_angles.begin(), all_angles.end(), all_angles.begin(), bind2nd(std::minus<float>(), min_value));
	std::cout << "Min angle "<< min_value << std::endl;

	std::vector<std::vector<float> > subset_angles(this->number_of_subsets);
	std::vector<std::vector<floatd2>> subset_offsets(this->number_of_subsets);

	std::vector<std::vector<unsigned int> > subset_projections(this->number_of_subsets);

	auto bin_vector = binning->get_bins();

	std::map<unsigned int, unsigned int> bins;

	for (auto bin = 0u; bin < bin_vector.size(); bin++)
		for (auto proj : bin_vector[bin])
			bins[proj] = bin;

	int k = 0;
	for (int i = 0; i < all_angles.size(); i++){
		if (bins.find(i) != bins.end()){ //Check to make sure the projection is in the bins
			subset_projections[k%this->number_of_subsets].push_back(i);
			k++;
		}
	}


	//Sort subsets by bins
	for (auto & subset : subset_projections)
		std::sort(subset.begin(),subset.end(),[&](unsigned int val1, unsigned int val2){return (bins[val1] < bins[val2]);});

	//Fill in corresponding angles and offsets
	for (auto i = 0u; i < subset_projections.size(); i++){
		for (auto proj : subset_projections[i]){
			subset_angles[i].push_back(all_angles[proj]);
			subset_offsets[i].push_back(all_offsets[proj]);
		}
	}



	//Calculate the required permutation to order the actual projection array.
	for (auto &subset : subset_projections){
		for (auto proj : subset){
			permutations.push_back(proj);
		}
	}

	acq->set_projections(permute_projections(acq->get_projections(),permutations));

	std::vector<floatd2> new_offsets;
	std::vector<float> new_angles;
	for (auto & i : permutations){
		new_angles.push_back(all_angles[i]);
		new_offsets.push_back(all_offsets[i]);
	}
	geometry->set_angles(new_angles);
	geometry->set_offsets(new_offsets);


	for (int subset = 0; subset < this->number_of_subsets; subset++){
		auto subset_geometry = boost::make_shared<CBCT_geometry>(*geometry);
		subset_geometry->set_angles(subset_angles[subset]);
		subset_geometry->set_offsets(subset_offsets[subset]);

		auto subset_acquisistion = boost::make_shared<CBCT_acquisition>();
		subset_acquisistion->set_geometry(subset_geometry);

		std::vector<std::vector<unsigned int> > subset_bins(binning->get_number_of_bins());
		for ( unsigned int i = 0; i < subset_projections[subset].size(); i++){
			unsigned int proj = subset_projections[subset][i];
			subset_bins[bins[proj]].push_back(i);
		}



		auto subset_binning = boost::make_shared<CBCT_binning>(subset_bins);
		operators[subset]->setup(subset_acquisistion,subset_binning,is_dims_in_mm,false);
	}

	angles = std::vector<std::vector<float>>();
	offsets = std::vector<std::vector<floatd2>>();
	for (int bin =0; bin < binning->get_number_of_bins(); bin++){
		auto binvec = binning->get_bin(bin);
		angles.push_back(std::vector<float>());
		offsets.push_back(std::vector<floatd2>());
		for (auto index : binvec){
			angles.back().push_back(new_angles[index]);
			offsets.back().push_back(new_offsets[index]);
		}

	}

	uint64d2 proj_dim{ acq->get_projections()->get_size(0), acq->get_projections()->get_size(1)};
	for (int i = 0; i < this->number_of_subsets; i++)
		projection_dims.push_back(boost::shared_ptr<std::vector<size_t> >(new std::vector<size_t>{proj_dim[0],proj_dim[1],subset_angles[i].size()}));




}



template<template<class> class ARRAY> boost::shared_ptr<std::vector<size_t> > Gadgetron::CBSubsetOperator<ARRAY>::get_codomain_dimensions(
		int subset) {
	return projection_dims[subset];

}

template<template<class> class ARRAY> void Gadgetron::CBSubsetOperator<ARRAY>::setup(
		boost::shared_ptr<CBCT_acquisition> acq, floatd3 is_dims_in_mm) {
	std::vector<unsigned int> bin1(acq->get_projections()->get_size(2));
	std::iota(bin1.begin(),bin1.end(),0);

	auto binning = boost::make_shared<CBCT_binning>(std::vector<std::vector<unsigned int>>(1,bin1));

	setup(acq,binning,is_dims_in_mm);

}

template<template<class > class ARRAY>
inline boost::shared_ptr<ARRAY<bool> > Gadgetron::CBSubsetOperator<ARRAY>::calculate_mask(
		boost::shared_ptr<ARRAY<float> > projections, float limit) {

	auto mask = boost::make_shared<ARRAY<bool>>(this->get_domain_dimensions());
	fill(mask.get(),true);
	auto proj_subs = this->projection_subsets(projections.get());
	for (int i = 0; i < operators.size(); i++){
		*mask &= *operators[i]->calculate_mask(proj_subs[i].get(),limit);
	}
	return mask;
}

template<template<class> class ARRAY> boost::shared_ptr<hoCuNDArray<float> > Gadgetron::CBSubsetOperator<ARRAY>::permute_projections(
		boost::shared_ptr<hoNDArray<float> > projections,
		std::vector<unsigned int>  & permutations) {

	auto dims = *projections->get_dimensions();
	dims.back() = permutations.size();
	auto result = boost::make_shared<hoCuNDArray<float>>(dims);

	size_t nproj = permutations.size();
	size_t proj_size = projections->get_size(0)*projections->get_size(1);

	float * res_ptr = result->get_data_ptr();
	float * proj_ptr = projections->get_data_ptr();

	for (unsigned int i = 0; i < nproj; i++){
		cudaMemcpy(res_ptr+i*proj_size,proj_ptr+proj_size*permutations[i],proj_size*sizeof(float),cudaMemcpyHostToHost);
	}
	return result;


}

template class Gadgetron::CBSubsetOperator<cuNDArray>;
template class Gadgetron::CBSubsetOperator<hoCuNDArray>;
