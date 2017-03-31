/*

 * CBSubsetOperator.h
 *
 *  Created on: Feb 12, 2015
 *      Author: u051747
 */

#pragma once
#include "subsetOperator.h"
#include <boost/shared_ptr.hpp>
#include <vector>
#include "CBCT_acquisition.h"
#include "cuConebeamProjectionOperator.h"
#include "hoCuConebeamProjectionOperator.h"
namespace Gadgetron {

template<template<class> class T> struct conebeamProjectionOperator{};

template<> struct conebeamProjectionOperator<hoCuNDArray>{
	using type = hoCuConebeamProjectionOperator;
};

template<> struct conebeamProjectionOperator<cuNDArray>{
	using type = cuConebeamProjectionOperator;
};



template<template<class> class ARRAY> class CBSubsetOperator: public Gadgetron::subsetOperator<ARRAY<float>> {
public:
	CBSubsetOperator(int subsets);
	CBSubsetOperator() : CBSubsetOperator(1) {};
	virtual ~CBSubsetOperator();

	virtual void mult_M(ARRAY<float>* in, ARRAY<float> * out,int subset, bool accumulate) override;
	virtual void mult_MH(ARRAY<float>* in, ARRAY<float> * out,int subset, bool accumulate) override;
	virtual void mult_MH_M(ARRAY<float>* in, ARRAY<float> * out,int subset, bool accumulate) override;


	virtual boost::shared_ptr<std::vector<size_t>  > get_codomain_dimensions(int subset) override;

	virtual void set_domain_dimensions(std::vector<size_t>* dims) override {
		linearOperator<ARRAY<float>>::set_domain_dimensions(dims);
		for (auto op : operators)
			op->set_domain_dimensions(dims);
	}
	void setup(boost::shared_ptr<CBCT_acquisition> acq, boost::shared_ptr<CBCT_binning>, floatd3 is_dims_in_mm);
	void setup(boost::shared_ptr<CBCT_acquisition> acq, floatd3 is_dims_in_mm);
	boost::shared_ptr<ARRAY<bool>> calculate_mask(boost::shared_ptr<ARRAY<float> > projections, float limit);

	void offset_correct(ARRAY<float>*);
	void set_use_offset_correction(bool use){
		for (auto op : operators) op->set_use_offset_correction(use);
	}

	void set_mask(boost::shared_ptr<ARRAY<bool> > mask){
		for (auto op : operators)
			op->set_mask(mask);
	}


protected:

	boost::shared_ptr<hoCuNDArray<float>> permute_projections(boost::shared_ptr<hoNDArray<float>> projections, std::vector<unsigned int > & permutations);
	std::vector<boost::shared_ptr<typename conebeamProjectionOperator<ARRAY>::type>> operators;

	std::vector<boost::shared_ptr<std::vector<size_t> > > projection_dims;
	std::vector<std::vector<float> > angles;
	std::vector<std::vector<floatd2> > offsets;
	std::vector<unsigned int> permutations; //Projection permutations
};

} /* namespace Gadgetron */
