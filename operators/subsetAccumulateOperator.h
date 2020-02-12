#pragma once

#include "subsetOperator.h"

namespace Gadgetron {

template<class ARRAY> class subsetAccumulateOperator : public subsetOperator<ARRAY>{
public:
	subsetAccumulateOperator(boost::shared_ptr<subsetOperator<ARRAY> > op_, unsigned int splits_) : subsetOperator<ARRAY>(), op(op_), splits(splits_){
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


		//ARRAY tmp(dims,in->get_data_ptr());
		//op->mult_M(&tmp,out,subset,accumulate);
	}

	virtual boost::shared_ptr< std::vector<size_t> > get_domain_dimensions() override {
		auto dims = op->get_domain_dimensions();
		dims->push_back(splits);
		return dims;
	}


	virtual void mult_MH(ARRAY* in, ARRAY* out, int subset, bool accumulate = false ) override {

		auto dims = *out->get_dimensions();
		auto back_dim = dims.back();
		dims.pop_back();

		ARRAY tmp(dims);
		auto elements = tmp.get_number_of_elements();
		op->mult_MH(in,&tmp,subset,false);
		if (!accumulate) clear(out);
		for (auto i =0u; i < back_dim; i++){
			ARRAY view(dims,out->get_data_ptr()+elements*i);
			view += tmp;
		}


		//ARRAY tmp(dims,out->get_data_ptr());
		//op->mult_MH(in,&tmp,subset,accumulate);

	}


	virtual boost::shared_ptr< std::vector<size_t> > get_codomain_dimensions(int subset) override{
		return op->get_codomain_dimensions(subset);
	}

	virtual int get_number_of_subsets() override {return op->get_number_of_subsets();}

protected:
	boost::shared_ptr<subsetOperator<ARRAY>> op;
	unsigned int splits;
};
}
