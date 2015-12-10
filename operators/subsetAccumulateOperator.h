#pragma once

#include "subsetOperator.h"

namespace Gadgetron {

template<class ARRAY> class subsetAccumulateOperator : public subsetOperator<ARRAY>{
public:
	subsetAccumulateOperator(boost::shared_ptr<subsetOperator<ARRAY> > op_) : subsetOperator<ARRAY>(), op(op_){
		this->number_of_subsets = op->get_number_of_subsets();
	}

	virtual void mult_M(ARRAY* in, ARRAY* out, int subset, bool accumulate = false ) override {
		auto dims = *in->get_dimensions();
		auto back_dim = dims.back();
		dims.pop_back();
		ARRAY tmp(dims);
		clear(&tmp);
		auto elements = tmp.get_number_of_elements();

		for (auto i =0u; i < back_dim; i++){
			ARRAY view(dims,in->get_data_ptr()+elements*i);
			tmp += view;
		}

		op->mult_M(&tmp,out,subset,accumulate);
	}

	virtual void mult_MH(ARRAY* in, ARRAY* out, int subset, bool accumulate = false ) override {
		auto dims = *out->get_dimensions();
		auto back_dim = dims.back();
		dims.pop_back();
		ARRAY tmp(dims);
		auto elements = tmp.get_number_of_elements();
		op->mult_MH(in,&tmp,subset,false);
		for (auto i =0u; i < back_dim; i++){
			ARRAY view(dims,out->get_data_ptr()+elements*i);
			if (accumulate)
				view += tmp;
			else
				view = tmp;
		}

	}


	virtual boost::shared_ptr< std::vector<size_t> > get_codomain_dimensions(int subset){
		return op->get_codomain_dimensions(subset);
	}
protected:
	boost::shared_ptr<subsetOperator<ARRAY>> op;
};
}
