#pragma once
#include "linearOperator.h"
#include <numeric>
#include <functional>
namespace Gadgetron {

template<class ARRAY> class subselectionOperator : public linearOperator<ARRAY> {

public:

	subselectionOperator(boost::shared_ptr<linearOperator<ARRAY> > op_, unsigned int subview_ = 0) : linearOperator<ARRAY>(), op(op_), subview(subview_) {}

	virtual void mult_M(ARRAY* in, ARRAY* out, bool accumulate = false) override {
		auto dims = *in->get_dimensions();
		auto dims2 = dims;
		dims2.pop_back();

		size_t elements = std::accumulate(dims2.begin(),dims2.end(),1,std::multiplies<size_t>());
		ARRAY view(dims2,in->get_data_ptr()+elements*subview);
		op->mult_M(&view,out,accumulate);

	}

	virtual void mult_MH(ARRAY* in, ARRAY* out, bool accumulate = false) override {
		auto dims = *out->get_dimensions();
		auto dims2 = dims;
		dims2.pop_back();

		size_t elements = std::accumulate(dims2.begin(),dims2.end(),1,std::multiplies<size_t>());
		ARRAY out_view(dims2,out->get_data_ptr()+elements*subview);
		if (!accumulate)
			clear(out);
		op->mult_MH(in,&out_view,accumulate);


	}

	virtual void mult_MH_M(ARRAY* in, ARRAY *out, bool accumulate=false) override {
		auto dims = *out->get_dimensions();
		auto dims2 = dims;
		dims2.pop_back();

		size_t elements = std::accumulate(dims2.begin(),dims2.end(),1,std::multiplies<size_t>());
		ARRAY out_view(dims2,out->get_data_ptr()+elements*subview);
		ARRAY in_view(dims2,in->get_data_ptr()+elements*subview);
		if (!accumulate)
			clear(out);
		op->mult_MH_M(&in_view,&out_view,accumulate);


	}

protected:
	boost::shared_ptr<linearOperator<ARRAY>> op;
	unsigned int subview;
};


}
