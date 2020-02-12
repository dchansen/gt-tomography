#pragma once


#include "circulantPreconditioner.h"


using namespace Gadgetron;

#include "cuNDFFT.h"
#include "hoNDFFT.h"


template<class ARRAY> void circulantPreconditioner<ARRAY>::apply(ARRAY * in, ARRAY * out){
	//boost::shared_ptr<>real_to_complex(in)
	real_to_complex()
	//FFTInstance<ARRAY>::instance()->
	convolutionOperator
}
