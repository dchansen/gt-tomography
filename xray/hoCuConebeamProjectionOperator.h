#pragma once

#include "cuNDArray.h"
#include "linearOperator.h"
#include "CBCT_acquisition.h"
#include "CBCT_binning.h"
#include "hoCuNDArray_math.h"
#include "hoCuNDArray_elemwise.h"

#include <numeric>
#include <math_constants.h>
#include <vector>
#include <boost/make_shared.hpp>

namespace Gadgetron{

class hoCuConebeamProjectionOperator : public linearOperator< hoCuNDArray<float> >
{
public:
	hoCuConebeamProjectionOperator() : linearOperator< hoCuNDArray<float> >()
	{
		samples_per_pixel_ = 1.5f;
		projections_per_batch_ = 20;
		use_fbp_ = false;
		short_scan_ = false;
		preprocessed_ = false;
		use_offset_correction_ = false;
		allow_offset_correction_override_ = true;
	}

	boost::shared_ptr<hoCuNDArray<bool>> calculate_mask( hoCuNDArray<float>* projections, float limit);
	virtual ~hoCuConebeamProjectionOperator() {}

	virtual void mult_M( hoCuNDArray<float> *in, hoCuNDArray<float> *out, bool accumulate = false );
	virtual void mult_MH( hoCuNDArray<float> *in, hoCuNDArray<float> *out, bool accumulate = false );

	virtual void offset_correct(hoCuNDArray<float>* proj);

	virtual void setup( boost::shared_ptr<CBCT_acquisition> acquisition,
			floatd3 is_dims_in_mm, bool transform_angles = true )
	{
		acquisition_ = acquisition;
		is_dims_in_mm_ = is_dims_in_mm;

		// Determine the minimum and maximum angles scanned and transform array angles from [0;max_angle_].
		//

		std::vector<float> &angles = acquisition->get_geometry()->get_angles();
		if (transform_angles){
			float min_value = *std::min_element(angles.begin(), angles.end() );
			transform(angles.begin(), angles.end(), angles.begin(), bind2nd(std::minus<float>(), min_value));
		}
		// Are we in a short scan setup?
		// - we say yes if we have covered less than PI+3*delta radians
		//

		float angle_span = *std::max_element(angles.begin(), angles.end() );
		floatd2 ps_dims_in_mm = acquisition_->get_geometry()->get_FOV();
		float SDD = acquisition_->get_geometry()->get_SDD();
		float delta = std::atan(ps_dims_in_mm[0]/(2.0f*SDD)); // Fan angle

		if( angle_span*CUDART_PI_F/180.0f > CUDART_PI_F+3.0f*delta )
			short_scan_ = false;
		else
			short_scan_ = true;

		/*
      std::cout << std::endl <<  *std::min_element(angles.begin(), angles.end() ) << " " 
      << *std::max_element(angles.begin(), angles.end() ) << std::endl;
		 */

		std::vector<floatd2> offsets = acquisition_->get_geometry()->get_offsets();
		floatd2 mean_offset = std::accumulate(offsets.begin(),offsets.end(),floatd2(0,0))/float(offsets.size());

		if( allow_offset_correction_override_ && std::abs(mean_offset[0]) > ps_dims_in_mm[0]*0.1f )
			use_offset_correction_ = true;

		std::cout << "Mean offset " << mean_offset << " Use offset correct: " << (use_offset_correction_ ? "true" : "false") << std::endl;
		preprocessed_ = true;

		std::vector<unsigned int> bins(angles.size());
		std::iota(bins.begin(),bins.end(),0);
		binning_ = boost::make_shared<CBCT_binning>(std::vector<std::vector<unsigned int>>(1,bins));

	}

	virtual void setup( boost::shared_ptr<CBCT_acquisition> acquisition,
			boost::shared_ptr<CBCT_binning> binning,
			floatd3 is_dims_in_mm,bool transform_angles=true )
	{
		setup( acquisition, is_dims_in_mm,transform_angles );
		binning_ = binning;
	}


	inline void set_use_filtered_backprojection( bool use_fbp ){
		use_fbp_ = use_fbp;
	}

	inline void set_use_offset_correction( bool use_correction ){
		use_offset_correction_ = use_correction;
		allow_offset_correction_override_ = false;
	}

	inline bool get_use_offset_correction(){
		return use_offset_correction_;
	}

	inline void set_num_projections_per_batch( unsigned int projections_per_batch ){
		projections_per_batch_ = projections_per_batch;
	}

	inline void set_num_samples_per_pixel( float samples_per_pixel ){
		samples_per_pixel_ = samples_per_pixel;
	}

	inline void set_frequency_filter( boost::shared_ptr< cuNDArray<float> > weights ){
		frequency_filter_ = weights;
	}

	void set_acquisition( boost::shared_ptr<CBCT_acquisition> acquisition ){
		acquisition_ = acquisition;
	}

	boost::shared_ptr<CBCT_acquisition> get_acquisition(){
		return acquisition_;
	}

	void set_binning( boost::shared_ptr<CBCT_binning> binning ){
		binning_ = binning;
	}

	boost::shared_ptr<CBCT_binning> get_binning(){
		return binning_;
	}
	void set_mask(boost::shared_ptr<hoCuNDArray<bool>> m){ mask_ = m;}

protected:
	virtual void compute_default_frequency_filter();
	virtual void compute_cosine_weights();

protected:
	boost::shared_ptr<CBCT_acquisition> acquisition_;
	boost::shared_ptr<CBCT_binning> binning_;
	floatd3 is_dims_in_mm_;
	float samples_per_pixel_;
	bool use_fbp_;
	unsigned int projections_per_batch_;
	bool preprocessed_;
	bool short_scan_;
	bool use_offset_correction_;
	bool allow_offset_correction_override_;
	boost::shared_ptr< cuNDArray<float> > cosine_weights_;
	boost::shared_ptr< cuNDArray<float> > frequency_filter_;
	boost::shared_ptr<hoCuNDArray<bool>> mask_;
};
}
