/*
 * CTSubsetOperator.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: u051747
 */

#include "CTSubsetOperator.h"
#include <map>

using namespace Gadgetron;

template<template<class> class ARRAY> Gadgetron::CTSubsetOperator<ARRAY>::CTSubsetOperator(int subsets): subsetOperator<ARRAY<float>>(subsets) {
	for (int i = 0; i < subsets; i++)
		operators.push_back(boost::make_shared<CTProjectionOperator<ARRAY>>());
}

template<template<class> class ARRAY> Gadgetron::CTSubsetOperator<ARRAY>::~CTSubsetOperator() {
}

template<template<class> class ARRAY> void Gadgetron::CTSubsetOperator<ARRAY>::mult_M(ARRAY<float>* image,
		ARRAY<float>* projections, int subset, bool accumulate) {
	operators[subset]->mult_M(image,projections,accumulate);

}

template<template<class> class ARRAY> void Gadgetron::CTSubsetOperator<ARRAY>::mult_MH(ARRAY<float>* projections,
		ARRAY<float>* image, int subset, bool accumulate) {
	operators[subset]->mult_MH(projections,image,accumulate);
}

template<template<class> class ARRAY> void Gadgetron::CTSubsetOperator<ARRAY>::mult_MH_M(ARRAY<float>* in,
		ARRAY<float>* out, int subset, bool accumulate) {
	operators[subset]->mult_MH_M(in,out,accumulate);
}


template<template<class> class ARRAY> boost::shared_ptr<hoCuNDArray<float>> Gadgetron::CTSubsetOperator<ARRAY>::setup(
		boost::shared_ptr<CT_acquisition> acq, floatd3 is_dims_in_mm, boost::shared_ptr<hoCuNDArray<float>> weights) {





	std::vector<std::vector<unsigned int> > subset_projections(this->number_of_subsets);

	std::vector<boost::shared_ptr<CT_acquisition> > subset_acquisitions;
	for (int i = 0; i < this->number_of_subsets; i++) subset_acquisitions.push_back(boost::make_shared<CT_acquisition>());


	int k = 0;
	for (int i = 0; i < acq->geometry.detectorFocalCenterAngularPosition.size(); i++){
			subset_projections[i%this->number_of_subsets].push_back(i);
	}


	auto & geometry = acq->geometry;
	//Fill in corresponding angles and offsets
	for (auto i = 0u; i < subset_projections.size(); i++){
		auto & sub_geometry = subset_acquisitions[i]->geometry;
		std::cout << "Setting values for subset # " << i << std::endl;
		for (auto proj : subset_projections[i]){
			sub_geometry.detectorFocalCenterAngularPosition.push_back(geometry.detectorFocalCenterAngularPosition[proj]);
			sub_geometry.detectorFocalCenterAxialPosition.push_back(geometry.detectorFocalCenterAxialPosition[proj]);
			sub_geometry.detectorFocalRadialDistance.push_back(geometry.detectorFocalRadialDistance[proj]);

			sub_geometry.sourceAngularPositionShift.push_back(geometry.sourceAngularPositionShift[proj]);
			sub_geometry.sourceAxialPositionShift.push_back(geometry.sourceAxialPositionShift[proj]);
			sub_geometry.sourceRadialDistanceShift.push_back(geometry.sourceRadialDistanceShift[proj]);

			sub_geometry.detectorCentralElement.push_back(geometry.detectorCentralElement[proj]);
			sub_geometry.constantRadialDistance.push_back(geometry.constantRadialDistance[proj]);


		}
		sub_geometry.detectorSize = geometry.detectorSize;

		std::cout  << "Focal size " << sub_geometry.detectorFocalCenterAngularPosition.size() << std::endl;
	}



	//Calculate the required permutation to order the actual projection array.

	for (auto &subset : subset_projections){
		for (auto proj : subset){
			permutations.push_back(proj);
		}
	}


	boost::shared_ptr<hoCuNDArray<float>> permuted_weights;
	float * weights_ptr;
	if (weights) {
		permuted_weights = permute_projections(*weights, permutations);
		weights_ptr = permuted_weights->get_data_ptr();
	}



	for (int i = 0; i < this->number_of_subsets; i++){

		std::vector<size_t> codims{ acq->projections.get_size(0),acq->projections.get_size(1),subset_acquisitions[i]->geometry.detectorFocalCenterAngularPosition.size()};

		operators[i]->set_codomain_dimensions(&codims);

		operators[i]->set_domain_dimensions(this->get_domain_dimensions().get());
		if (weights){
			std::vector<size_t> weights_dim = {permuted_weights->get_size(0),permuted_weights->get_size(1),subset_acquisitions[i]->geometry.detectorFocalCenterAngularPosition.size()};
			auto weights_view = hoCuNDArray<float>(weights_dim,weights_ptr);
			weights_ptr += weights_view.get_number_of_elements();
			auto  subset_weights = boost::make_shared<ARRAY<float>>(weights_view );
			operators[i]->setup(subset_acquisitions[i],subset_weights,is_dims_in_mm);
		} else {
			operators[i]->setup(subset_acquisitions[i],is_dims_in_mm);
		}

	}




	return 	permute_projections(acq->projections,permutations);

}



template<template<class> class ARRAY> boost::shared_ptr<std::vector<size_t> > Gadgetron::CTSubsetOperator<ARRAY>::get_codomain_dimensions(
		int subset) {
	return operators[subset]->get_codomain_dimensions();

}



template<template<class> class ARRAY> boost::shared_ptr<hoCuNDArray<float> > Gadgetron::CTSubsetOperator<ARRAY>::permute_projections(
		hoCuNDArray<float> & projections,
		std::vector<unsigned int>  & permutations) {

	auto dims = *projections.get_dimensions();
	dims.back() = permutations.size();
	auto result = boost::make_shared<hoCuNDArray<float>>(dims);

	size_t nproj = permutations.size();
	size_t proj_size = projections.get_size(0)*projections.get_size(1);

	float * res_ptr = result->get_data_ptr();
	float * proj_ptr = projections.get_data_ptr();

	for (unsigned int i = 0; i < nproj; i++){
		cudaMemcpy(res_ptr+i*proj_size,proj_ptr+proj_size*permutations[i],proj_size*sizeof(float),cudaMemcpyHostToHost);
	}
	return result;


}

template class Gadgetron::CTSubsetOperator<cuNDArray>;
template class Gadgetron::CTSubsetOperator<hoCuNDArray>;
