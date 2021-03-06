#pragma once

#include "linearOperator.h"

namespace Gadgetron {

template<class ARRAY> class accumulateOperator : public linearOperator<ARRAY>{
public:
	accumulateOperator(boost::shared_ptr<linearOperator<ARRAY> > op_) : linearOperator<ARRAY>(), op(op_){}

	virtual void mult_M(ARRAY* in, ARRAY* out, bool accumulate = false ){
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

		op->mult_M(&tmp,out,accumulate);

	}

	virtual void mult_MH(ARRAY* in, ARRAY* out, bool accumulate = false ){
		auto dims = *out->get_dimensions();
		auto back_dim = dims.back();
		dims.pop_back();
		ARRAY tmp(dims);
		auto elements = tmp.get_number_of_elements();
		op->mult_MH(in,&tmp);
		for (auto i =0u; i < back_dim; i++){
			ARRAY view(dims,out->get_data_ptr()+elements*i);
			if (accumulate)
				view += tmp;
			else
				view = tmp;
		}

	}
protected:
	boost::shared_ptr<linearOperator<ARRAY>> op;
};
}
