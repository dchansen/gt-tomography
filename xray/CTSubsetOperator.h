/*

 * CTSubsetOperator.h
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
#include "CTProjectionOperator.h"
namespace Gadgetron {



//template<> struct ctProjectionOperator<hoCuNDArray>{
	//using type = oCuConebeamProjectionOperator;
//};




template<template<class> class ARRAY> class CTSubsetOperator: public Gadgetron::subsetOperator<ARRAY<float>> {
public:
	CTSubsetOperator(int subsets);
	CTSubsetOperator() : CTSubsetOperator(1) {};
	virtual ~CTSubsetOperator();

	virtual void mult_M(ARRAY<float>* in, ARRAY<float> * out,int subset, bool accumulate) override;
	virtual void mult_MH(ARRAY<float>* in, ARRAY<float> * out,int subset, bool accumulate) override;
	virtual void mult_MH_M(ARRAY<float>* in, ARRAY<float> * out,int subset, bool accumulate) override;


	virtual boost::shared_ptr<std::vector<size_t>  > get_codomain_dimensions(int subset) override;

	virtual void set_domain_dimensions(std::vector<size_t>* dims) override {
		linearOperator<ARRAY<float>>::set_domain_dimensions(dims);
		for (auto op : operators)
			op->set_domain_dimensions(dims);
	}

	boost::shared_ptr<hoCuNDArray<float>> setup(boost::shared_ptr<CT_acquisition> acq, floatd3 is_dims_in_mm);




protected:

	boost::shared_ptr<hoCuNDArray<float>> permute_projections(hoCuNDArray<float>& projections, std::vector<unsigned int > & permutations);
	std::vector<boost::shared_ptr<CTProjectionOperator<ARRAY>>> operators;
	//std::vector<boost::shared_ptr<typename ctProjectionOperator<ARRAY>::type>> operators;

	std::vector<boost::shared_ptr<std::vector<size_t> > > projection_dims;

};

} /* namespace Gadgetron */
