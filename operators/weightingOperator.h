#pragma once

#include "linearOperator.h"
namespace Gadgetron{

template<class ARRAY_TYPE> class weightingOperator: public linearOperator<ARRAY_TYPE>{
public:
	weightingOperator(): linearOperator<ARRAY_TYPE>(){};
	weightingOperator(boost::shared_ptr<ARRAY_TYPE> _weights,boost::shared_ptr<linearOperator<ARRAY_TYPE>> _op): linearOperator<ARRAY_TYPE>(), weights(_weights), op(_op){};

	virtual void set_weights(boost::shared_ptr<ARRAY_TYPE> _weights){
		weights = _weights;
	}

	virtual void mult_M(ARRAY_TYPE* in, ARRAY_TYPE* out, bool accumulate=false){
		if (accumulate){
			ARRAY_TYPE tmp = *out;
			op->mult_M(in,&tmp);
			tmp *= *weights;
			*out += tmp;
		} else{
			op->mult_M(in,out);
			*out *= *weights;
		}
	}

	virtual void mult_MH(ARRAY_TYPE* in, ARRAY_TYPE* out, bool accumulate=false){
		ARRAY_TYPE tmp = *in;
		tmp *= *weights;
		op->mult_MH(&tmp,out,accumulate);
	}

protected:
	boost::shared_ptr<ARRAY_TYPE> weights;
	boost::shared_ptr<linearOperator<ARRAY_TYPE>> op;
};

}
