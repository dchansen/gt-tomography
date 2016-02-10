/*
 * linearToSubsetOperator.h
 *
 *  Created on: Nov 27, 2015
 *      Author: dch
 */

#ifndef LINEARTOSUBSETOPERATOR_H_
#define LINEARTOSUBSETOPERATOR_H_
#include "subsetOperator.h"

namespace Gadgetron{
template<class ARRAY_TYPE> class linearToSubsetOperator : public subsetOperator<ARRAY_TYPE> {

public:
	linearToSubsetOperator(boost::shared_ptr<linearOperator<ARRAY_TYPE>> op): subsetOperator<ARRAY_TYPE>(), op_(op){

	}

	virtual void mult_M(ARRAY_TYPE* in, ARRAY_TYPE* out, int subset, bool accumulate) override {
		return op_->mult_M(in,out,accumulate);
	}
	virtual void mult_MH(ARRAY_TYPE* in, ARRAY_TYPE* out, int subset, bool accumulate) override {
		return op_->mult_MH(in,out,accumulate);
	}
	virtual void mult_MH_M(ARRAY_TYPE* in, ARRAY_TYPE* out, int subset, bool accumulate) override{
		return op_->mult_MH_M(in,out,accumulate);
	}

	virtual boost::shared_ptr< std::vector<size_t> > get_codomain_dimensions(int subset) override {
		return op_->get_codomain_dimensions();
	}


private:
	boost::shared_ptr<linearOperator<ARRAY_TYPE>> op_;


};
}



#endif /* LINEARTOSUBSETOPERATOR_H_ */
