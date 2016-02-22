/*
 * cuCTProjectionOperator.h

 *
 *  Created on: Feb 18, 2015
 *      Author: u051747
 */
#pragma once
#include <linearOperator.h>
#include <cuNDArray_math.h>

#include <boost/shared_ptr.hpp>
#include "CT_acquisition.h"
#include "CBCT_binning.h"
#include <vector>
#include "vector_td.h"


namespace Gadgetron {

class cuCTProjectionOperator: public Gadgetron::linearOperator<cuNDArray<float>> {
public:
	cuCTProjectionOperator();
	virtual ~cuCTProjectionOperator();

	virtual void mult_M(cuNDArray<float>* input, cuNDArray<float>* output, bool accumulate=false);
	virtual void mult_MH(cuNDArray<float>* input, cuNDArray<float>* output, bool accumulate=false);

	void setup(boost::shared_ptr<CT_acquisition> acquisition, floatd3 is_dims_in_mm);
	void setup(boost::shared_ptr<CT_acquisition> acquisition, boost::shared_ptr<CBCT_binning> binnning,floatd3 is_dims_in_mm);

protected:

	/**
	 * Calculates the start and end index of the projections fitting to each projection
	 */
	std::vector<intd2> calculate_slice_indices();
	boost::shared_ptr<CT_acquisition> acquisition_;
	boost::shared_ptr<CBCT_binning> binning_;
	floatd3 is_dims_in_mm_;
	float samples_per_pixel_;
	bool preprocessed_;
	floatd3 offset_;
};

} /* namespace Gadgetron */
