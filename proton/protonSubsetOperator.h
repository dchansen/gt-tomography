#pragma once
#include "subsetOperator.h"
#include "hoCuNDArray.h"
#include "splineBackprojectionOperator.h"
#include "hdf5.h"
#include "hdf5_hl.h"
#include <numeric>
#include <thrust/reduce.h>
#include <thrust/functional.h>

/**
 * This class is inconsistent, messy and requires polish. It's also suspected to be unreliable
 */
namespace Gadgetron{
template<template<class> class ARRAY> class protonSubsetOperator : public subsetOperator<ARRAY<float> >{

public:
	protonSubsetOperator() : subsetOperator<ARRAY<float> >(1){

	}
	protonSubsetOperator(std::vector< boost::shared_ptr< protonDataset<ARRAY> > > datasets, floatd3 physical_dims ) : subsetOperator<ARRAY<float> >(datasets.size()){
		for (unsigned int i = 0; i < datasets.size(); i++){
			operators.push_back(splineBackprojectionOperator<ARRAY>(datasets[i],physical_dims));
			subset_dimensions.push_back(datasets[i]->get_projections()->get_dimensions());
		}

	}

	virtual ~protonSubsetOperator(){};


	virtual void mult_M(ARRAY<float>* in, ARRAY<float>* out, int subset, bool accumulate=false){
		std::stringstream ss;
		ss << "Subset " << subset << " out of bounds";
		if (subset >= operators.size() ) throw std::runtime_error(ss.str());
		operators[subset].mult_M(in,out,accumulate);
	}
	virtual void mult_MH(ARRAY<float>* in, ARRAY<float>* out, int subset, bool accumulate=false){
			std::stringstream ss;
			ss << "Subset " << subset << " out of bounds";
			if (subset >= operators.size() ) throw std::runtime_error(ss.str());
			operators[subset].mult_MH(in,out,accumulate);
	}
	virtual void mult_MH_M(ARRAY<float>* in, ARRAY<float>* out, int subset, bool accumulate=false){
	  if (subset >= operators.size() ) throw std::runtime_error("Subset out of bounds");
				operators[subset].mult_MH_M(in,out,accumulate);
	}
  virtual boost::shared_ptr< linearOperator<ARRAY<float> > > clone() {
       return linearOperator<ARRAY<float> >::clone(this);
     }


	virtual boost::shared_ptr< std::vector<size_t> > get_codomain_dimensions(int subset){
		return subset_dimensions[subset];
	}

protected:

	struct Spline{
		float x,y,z,x2,y2,z2;
		float dirx,diry,dirz,dirx2,diry2,dirz2;
	};



	std::vector< splineBackprojectionOperator<ARRAY> > operators;
	std::vector<boost::shared_ptr< std::vector<size_t> > > subset_dimensions;

};
}
