//
// Created by dch on 31/03/17.
//

#ifndef GT_TOMOGRAPHY_CBSUBSETWEIGHTOPERATOR_H
#define GT_TOMOGRAPHY_CBSUBSETWEIGHTOPERATOR_H

#include "CBSubsetOperator.h"
#include <hoNDArray.h>
namespace Gadgetron {
template<template<class> class ARRAY> class CBSubsetWeightOperator : public CBSubsetOperator<ARRAY> {

public:
    CBSubsetWeightOperator(int i) : CBSubsetOperator<ARRAY>(i){}
    CBSubsetWeightOperator() : CBSubsetOperator<ARRAY>(){}

    void setup(boost::shared_ptr<CBCT_acquisition> acq, boost::shared_ptr<CBCT_binning> binning, floatd3 is_dims_in_mm,
               boost::shared_ptr<hoNDArray<float>> projection_weights) {

        *acq->get_projections() *= *projection_weights;
        CBSubsetOperator<ARRAY>::setup(acq,binning,is_dims_in_mm);

        auto permuted_weights = this->permute_projections(projection_weights,this->permutations);
        weights = ARRAY<float>(*permuted_weights);
        std::cout << " Weight dims ";
        auto dims = *weights.get_dimensions();
        for (auto d : dims) std::cout << d << " ";
        std::cout << std::endl;
        subset_weights = this->projection_subsets(&weights);

    }

	virtual void mult_M(ARRAY<float>* in, ARRAY<float> * out,int subset, bool accumulate) override{
        ARRAY<float> *tmp_out = out;
        if (accumulate) tmp_out = new ARRAY<float>(out->get_dimensions());

        CBSubsetOperator<ARRAY>::mult_M(in,tmp_out,subset,false);
        *tmp_out *= *subset_weights[subset];
        if (accumulate) {
           *out += *tmp_out;
            delete(tmp_out);
        }

    }
	virtual void mult_MH(ARRAY<float>* in, ARRAY<float> * out,int subset, bool accumulate) override {
       ARRAY<float> tmp_in(*in);
        tmp_in *= *subset_weights[subset];
        CBSubsetOperator<ARRAY>::mult_MH(&tmp_in,out,subset,accumulate);

    }
	virtual void mult_MH_M(ARRAY<float>* in, ARRAY<float> * out,int subset, bool accumulate) override {

        auto codom = this->get_codomain_dimensions(subset);
        ARRAY<float> tmp(codom);
        this->mult_M(in,&tmp,subset,false);
        this->mult_MH(&tmp,out,subset,accumulate);

    }
protected:

    ARRAY<float> weights;
    std::vector<boost::shared_ptr<ARRAY<float>>> subset_weights;
    };
}


#endif //GT_TOMOGRAPHY_CBSUBSETWEIGHTOPERATOR_H
