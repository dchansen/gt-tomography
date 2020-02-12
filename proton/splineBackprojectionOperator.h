#pragma once
#include "cuNDArray_operators.h"
#include "linearOperator.h"
#include "cuNDArray.h"
#include <vector>
#include "GPUTimer.h"
#include "protonDataset.h"

#include "proton_utils.h"
namespace Gadgetron{
template<template<class> class ARRAY>
class splineBackprojectionOperator : public linearOperator<ARRAY<float> > {
 public:
	/**
	 * Creates linearOperator for proton tomography, based on a dataset and a set of physical dimensions
	 * @param data Proton data. Only splines are used internally.
	 * @param physical_dims Physical dimensions of the image in cm
	 */
	splineBackprojectionOperator(boost::shared_ptr<protonDataset<ARRAY> > data,floatd3 physical_dims) : linearOperator<ARRAY<float> >() {
		this->physical_dims = physical_dims;
		this->data = data;
		linearOperator<ARRAY<float> >::set_codomain_dimensions(data->get_projections()->get_dimensions().get());
		use_weights = true;
    }
    virtual ~splineBackprojectionOperator() {}

    /**
     * Applies the operator ( = projection)
     * @param in
     * @param out
     * @param accumulate
     */
    virtual void mult_M( ARRAY<float>* in, ARRAY<float>* out, bool accumulate = false );
    /**
     * Applies the adjoint of the operator ( = backprojection)
     * @param in
     * @param out
     * @param accumulate
     */
    virtual void mult_MH( ARRAY<float>* in, ARRAY<float>* out, bool accumulate = false );



    virtual void set_codomain_dimensions( std::vector<size_t> *dims ){
    	throw std::runtime_error("cuOperatorPathBackprojection::codomain dimension must be set through the setup function");
    }
    virtual boost::shared_ptr< linearOperator< ARRAY<float> > > clone() {
       return boost::shared_ptr< linearOperator< ARRAY<float> > >(new splineBackprojectionOperator<ARRAY>(data,physical_dims));
     }


    virtual void pathNorm(ARRAY<float>* projections){
    	protonPathNorm(*this->get_domain_dimensions(),projections,data->get_splines().get(),physical_dims,data->get_EPL().get());
    }

    virtual void protonCount(ARRAY<float>* count_img){
    	countProtonsPerVoxel(count_img,data->get_splines().get(),physical_dims,data->get_EPL().get());
    }

    void set_use_weights(bool use){ use_weights = use;}

 protected:

    floatd3 physical_dims;
    boost::shared_ptr<protonDataset<ARRAY> > data;
    bool use_weights;

};


}
