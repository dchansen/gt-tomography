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

template< template<class> class ARRAY> class CTProjectionOperator: public Gadgetron::linearOperator<ARRAY<float>> {
public:
	CTProjectionOperator();
	virtual ~CTProjectionOperator();

	virtual void mult_M(ARRAY<float>* input, ARRAY<float>* output, bool accumulate=false);
	virtual void mult_MH(ARRAY<float>* input, ARRAY<float>* output, bool accumulate=false);

	void setup(boost::shared_ptr<CT_acquisition> acquisition, floatd3 is_dims_in_mm);
	void setup(boost::shared_ptr<CT_acquisition> acquisition, boost::shared_ptr<ARRAY<float>> weights,floatd3 is_dims_in_mm);
	void setup(boost::shared_ptr<CT_acquisition> acquisition, boost::shared_ptr<CBCT_binning> binnning,floatd3 is_dims_in_mm);


protected:

	/**
	 * Calculates the start and end index of the projections fitting to each projection
	 */
    std::vector<intd2> calculate_slice_indices(CT_acquisition &acquisition);

    boost::shared_ptr<CBCT_binning> binning;
    floatd3 is_dims_in_mm; //Image size in mm
    floatd2 ps_spacing; // Spacing between projection elements
    float ADD; //Aparture detector distance
    float samples_per_pixel_; //Number of samples used for the line integrals, divided by largest image slice dimensions
	bool preprocessed_;
    std::vector<std::vector<floatd3>> detector_focal_cyls;
    std::vector<std::vector<floatd3>> focal_offset_cyls;
    std::vector<std::vector<floatd2>> central_elements;
    std::vector<std::vector<intd2>> proj_indices;
	boost::shared_ptr<ARRAY<float>> weights;


};

} /* namespace Gadgetron */
