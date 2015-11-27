/*
 * cuConebeamProjectionOperator.h

 *
 *  Created on: Feb 18, 2015
 *      Author: u051747
 */
#pragma once
#include <linearOperator.h>
#include <cuNDArray_math.h>

#include <boost/shared_ptr.hpp>
#include "CBCT_acquisition.h"
#include "CBCT_binning.h"
#include <vector>
#include "vector_td.h"


namespace Gadgetron {

class cuConebeamProjectionOperator: public Gadgetron::linearOperator<cuNDArray<float>> {
public:
	cuConebeamProjectionOperator();
	virtual ~cuConebeamProjectionOperator();

	virtual void mult_M(cuNDArray<float>* input, cuNDArray<float>* output, bool accumulate=false);
	virtual void mult_MH(cuNDArray<float>* input, cuNDArray<float>* output, bool accumulate=false);

	boost::shared_ptr<cuNDArray<bool>> calculate_mask( cuNDArray<float>* projections, float limit);
	void setup(boost::shared_ptr<CBCT_acquisition> acquisition, floatd3 is_dims_in_mm, bool transform_angles = true);
	void setup(boost::shared_ptr<CBCT_acquisition> acquisition, boost::shared_ptr<CBCT_binning> binnning,floatd3 is_dims_in_mm,  bool transform_angles = true);
	void offset_correct(cuNDArray<float>*);
	bool get_use_offset_correction(){return use_offset_correction_;}
	void set_use_offset_correction(bool use){ use_offset_correction_ = use; allow_offset_correction_override_ = false;}

	void set_mask(boost::shared_ptr<cuNDArray<bool>> m){ mask = m;}
protected:
		boost::shared_ptr<hoCuNDArray<float>> permute_projections(boost::shared_ptr<hoCuNDArray<float>> projections,std::vector<unsigned int>& permutations);
	std::vector<unsigned int> new_order(std::vector<std::vector<unsigned int>>);
	boost::shared_ptr<CBCT_acquisition> acquisition_;
	boost::shared_ptr<CBCT_binning> binning_;
	std::vector<std::vector<float> > angles;
	std::vector<std::vector<floatd2> > offsets;
	floatd3 is_dims_in_mm_;
	float samples_per_pixel_;
	bool use_fbp_;
	unsigned int projections_per_batch_;
	bool preprocessed_;
	bool short_scan_;
	bool use_offset_correction_;
	bool allow_offset_correction_override_;
	boost::shared_ptr<cuNDArray<bool>> mask;
};

} /* namespace Gadgetron */
