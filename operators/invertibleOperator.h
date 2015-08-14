/*
 * invertibleOperator.h
 *
 *  Created on: Jul 5, 2015
 *      Author: u051747
 */
#pragma once
#include "linearOperator.h"
namespace Gadgetron {

template<class ARRAY> class invertibleOperator : public linearOperator<ARRAY> {

public:
	virtual void inverse(ARRAY* in,ARRAY* out,  bool accumulate = true) = 0;
};

}
