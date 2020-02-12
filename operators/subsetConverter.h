#pragma once

#include "subsetOperator.h"
#include <boost/shared_ptr.hpp>
namespace Gadgetron {


template<class ARRAY_TYPE> class subsetConverter : public subsetOperator<ARRAY_TYPE> {
public:
	subsetConverter(boost::shared_ptr<linearOperator<ARRAY_TYPE>> op_) : subsetOperator<ARRAY_TYPE>(), op(op_) {

	};

	virtual ~subsetConverter(){};
	virtual void mult_M(ARRAY_TYPE* in, ARRAY_TYPE* out, int subset, bool accumulate){
		op->mult_M(in,out,accumulate);
	}
	virtual void mult_MH(ARRAY_TYPE* in, ARRAY_TYPE* out, int subset, bool accumulate){
		op->mult_MH(in,out,accumulate);
	}
	virtual void mult_MH_M(ARRAY_TYPE* in, ARRAY_TYPE* out, int subset, bool accumulate){
		op->mult_MH_M(in,out,accumulate);
	}


	virtual boost::shared_ptr< std::vector<size_t> > get_domain_dimensions(int subset){
		return op->get_domain_dimensions();
	}

	virtual boost::shared_ptr< std::vector<size_t> > get_codomain_dimensions(int subset){
		return op->get_codomain_dimensions();
	}
protected:
	boost::shared_ptr<linearOperator<ARRAY_TYPE>> op;


};
}
