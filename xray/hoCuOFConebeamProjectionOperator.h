#pragma once

#include "cuNDArray.h"
#include "linearOperator.h"
#include "hoCuNDArray_math.h"
#include "hoLinearResampleOperator_eigen.h"
#include "hoLinearResampleOperator.h"
#include "cuDownsampleOperator.h"
#include "cuUpsampleOperator.h"

#include "CBCT_acquisition.h"
#include "CBCT_binning.h"
#include "conebeam_projection.h"

#include <numeric>

namespace Gadgetron{

class hoCuOFConebeamProjectionOperator : public linearOperator< hoCuNDArray<float> >
{
public:

	hoCuOFConebeamProjectionOperator() : linearOperator< hoCuNDArray<float> >()
	{
		samples_per_pixel_ = 1.5;
		projections_per_batch_ = 20;
		use_offset_correction_ = false;
		allow_offset_correction_override_ = true;
		phase_ = -1;
		displacements_set_ = false;
		preprocessed_ = false;
	}

	virtual ~hoCuOFConebeamProjectionOperator() {}

	virtual void set_encoding_phase( unsigned int phase ) { phase_ = (int) phase; }

	inline void set_use_offset_correction( bool use_correction ){
		use_offset_correction_ = use_correction;
		allow_offset_correction_override_ = false;
	}

	inline void set_num_projections_per_batch( unsigned int projections_per_batch ){
		projections_per_batch_ = projections_per_batch;
	}

	inline void set_num_samples_per_pixel( float samples_per_pixel ){
		samples_per_pixel_ = samples_per_pixel;
	}

	void offset_correct(hoCuNDArray<float>* projections){

		if( !preprocessed_ ){
			throw std::runtime_error( "Error: hoCuConebeamProjectionOperator::offset_correct: setup not performed");
		}
		float SDD = acquisition_->get_geometry()->get_SDD();
		float SAD = acquisition_->get_geometry()->get_SAD();
		floatd2 ps_dims_in_mm = acquisition_->get_geometry()->get_FOV();
		apply_offset_correct( projections,acquisition_->get_geometry()->get_offsets(),ps_dims_in_mm, SDD, SAD);
	}
	virtual void setup( boost::shared_ptr<CBCT_acquisition> acquisition,
			boost::shared_ptr<CBCT_binning> binning,
			floatd3 is_dims_in_mm )
	{
		if( !acquisition.get() || !binning.get() )
			throw std::runtime_error("Error: hoCuOFConebeamProjectionOperator::setup: illegal input");

		acquisition_ = acquisition;
		binning_ = binning;
		is_dims_in_mm_ = is_dims_in_mm;

		// Determine the minimum and maximum angles scanned and transform array angles from [0;max_angle_].
		// This is necessary to be consistent with the 'hoCuConebeamProjectionOperator'
		//

		std::vector<float> &angles = acquisition->get_geometry()->get_angles();
		float min_value = *std::min_element(angles.begin(), angles.end() );
		transform(angles.begin(), angles.end(), angles.begin(), bind2nd(std::minus<float>(), min_value));

		std::vector<floatd2> offsets = acquisition_->get_geometry()->get_offsets();
		floatd2 mean_offset = std::accumulate(offsets.begin(),offsets.end(),floatd2(0,0))/float(offsets.size());

		floatd2 ps_dims_in_mm = acquisition_->get_geometry()->get_FOV();

		if( allow_offset_correction_override_ && mean_offset[0] > ps_dims_in_mm[0]*0.1f )
			use_offset_correction_ = true;

		preprocessed_ = true;
	}

	virtual void set_displacement_field( boost::shared_ptr< hoNDArray<float> > displacements )
	{
		if( !displacements.get() || displacements->get_number_of_elements() == 0 ){
			throw std::runtime_error("Error: hoCuRegistrationCBCT_RegularizationOperator::set_displacement_field : illegal displacement field");
		}

		if( displacements->get_number_of_dimensions() != 5 ){
			throw std::runtime_error("Error: hoCuRegistrationCBCT_RegularizationOperator::set_displacement_field : displacement field not five-dimensional (x,y,z,t,{3,4})");
		}

		uint64d4 dims( displacements->get_size(0), displacements->get_size(1), displacements->get_size(2), displacements->get_size(3) );
		reg_is_dims_4d_ = to_std_vector<size_t,4>(dims);

		R_ = boost::make_shared<hoLinearResampleOperator_eigen<float,3>>();
		R_->set_displacement_field( displacements );

		displacements_set_ = true;
	}

	virtual void
	mult_M( hoCuNDArray<float> *image, hoCuNDArray<float> *projections, bool accumulate )
	{
		// Validate the input
		//

		if( image == 0x0 || projections == 0x0 ){
			throw std::runtime_error("Error: hoCuOFConebeamProjectionOperator::mult_M: illegal array pointer provided");
		}

		if( image->get_number_of_dimensions() != 3 ){
			throw std::runtime_error("Error: hoCuOFConebeamProjectionOperator::mult_M: image array must be three-dimensional");
		}

		if( projections->get_number_of_dimensions() != 3 ){
			throw std::runtime_error("Error: hoCuOFConebeamProjectionOperator::mult_M: projections array must be three-dimensional");
		}

		if( !preprocessed_ ){
			throw std::runtime_error( "Error: hoCuOFConebeamProjectionOperator::mult_M: setup not performed");
		}

		if( !displacements_set_ ){
			throw std::runtime_error( "Error: hoCuOFConebeamProjectionOperator::mult_M: displacement field not set");
		}

		if( phase_ < 0 ){
			throw std::runtime_error( "Error: hoCuOFConebeamProjectionOperator::mult_M: phase not set");
		}

		if( projections->get_size(2) != acquisition_->get_geometry()->get_angles().size() ||
				projections->get_size(2) != acquisition_->get_geometry()->get_offsets().size() ){
			throw std::runtime_error("Error: hoCuOFConebeamProjectionOperator::mult_M: inconsistent sizes of input arrays/vectors");
		}

		//
		// Forwards projection
		//

		floatd2 ps_dims_in_mm = acquisition_->get_geometry()->get_FOV();
		float SDD = acquisition_->get_geometry()->get_SDD();
		float SAD = acquisition_->get_geometry()->get_SAD();

		// Project all phases except phase_
		//

		hoCuNDArray<float> *moving_image;
		boost::shared_ptr< hoCuNDArray<float> > moving_image_boost;

		unsigned int num_downsamples = log2l(image->get_size(0)/reg_is_dims_4d_[0]);

		if( image->get_size(0) == reg_is_dims_4d_[0] )
			moving_image = image;
		else{
			// Downsampling required
			std::vector<size_t> tmp_dims = *image->get_dimensions();
			cuNDArray<float> tmp_in(image);
			for( unsigned int d=0; d<num_downsamples; d++ ){
				for( unsigned int i=0; i<tmp_dims.size(); i++ ) tmp_dims[i] /= 2;
				cuDownsampleOperator<float,3> D;
				cuNDArray<float> tmp_out(&tmp_dims);
				D.mult_M( &tmp_in, &tmp_out );
				tmp_in = tmp_out;
			}
			moving_image_boost = boost::shared_ptr< hoCuNDArray<float> >( new hoCuNDArray<float>(&tmp_dims) );
			moving_image = moving_image_boost.get();
			*moving_image = *tmp_in.to_host();
		}

		hoCuNDArray<float> image_4d(&reg_is_dims_4d_);
		R_->mult_M( moving_image, &image_4d );

		std::vector<size_t> is_dims_3d = reg_is_dims_4d_;
		is_dims_3d.pop_back();

		int num_3d_elements = is_dims_3d[0]*is_dims_3d[1]*is_dims_3d[2];

		hoCuNDArray<float>* projections2 = projections;
		if (accumulate)
			projections2 = new hoCuNDArray<float>(projections->get_dimensions());


		for (unsigned int b=0, bin=0; b<this->binning_->get_number_of_bins(); b++) {

			if( b==phase_ )
				continue;

			hoCuNDArray<float> image_3d(&is_dims_3d, image_4d.get_data_ptr()+bin*num_3d_elements);

			conebeam_forwards_projection( projections2, &image_3d,
					acquisition_->get_geometry()->get_angles(),
					acquisition_->get_geometry()->get_offsets(),
					binning_->get_bin(b),
					projections_per_batch_, samples_per_pixel_,
					is_dims_in_mm_, ps_dims_in_mm,
					SDD, SAD);

			bin++;
		}

		if (use_offset_correction_)
			this->offset_correct(projections2);

		if (accumulate){
			*projections += *projections2;
			delete projections2;
		}
	}

	virtual void
	mult_MH( hoCuNDArray<float> *projections, hoCuNDArray<float> *image, bool accumulate )
	{
		// Validate the input
		//

		if( image == 0x0 || projections == 0x0 ){
			throw std::runtime_error("Error: hoCuOFConebeamProjectionOperator::mult_MH:: illegal array pointer provided");
		}

		if( image->get_number_of_dimensions() != 3 ){
			throw std::runtime_error("Error: hoCuOFConebeamProjectionOperator::mult_MH: image array must be three-dimensional");
		}

		if( projections->get_number_of_dimensions() != 3 ){
			throw std::runtime_error("Error: hoCuOFConebeamProjectionOperator::mult_MH: projections array must be three-dimensional");
		}

		if( !preprocessed_ ){
			throw std::runtime_error( "Error: hoCuOFConebeamProjectionOperator::mult_MH: setup not performed");
		}

		if( !displacements_set_ ){
			throw std::runtime_error( "Error: hoCuOFConebeamProjectionOperator::mult_MH: displacement field not set");
		}

		if( phase_ < 0 ){
			throw std::runtime_error( "Error: hoCuOFConebeamProjectionOperator::mult_MH: phase not set");
		}

		if( projections->get_size(2) != acquisition_->get_geometry()->get_angles().size() ||
				projections->get_size(2) != acquisition_->get_geometry()->get_offsets().size() ){
			throw std::runtime_error("Error: hoCuOFConebeamProjectionOperator::mult_MH: inconsistent sizes of input arrays/vectors");
		}

		//
		// Backwards projection
		//

		floatd2 ps_dims_in_mm = acquisition_->get_geometry()->get_FOV();
		float SDD = acquisition_->get_geometry()->get_SDD();
		float SAD = acquisition_->get_geometry()->get_SAD();

		// Backproject all phases but phase_
		//

		hoCuNDArray<float> image_4d(&reg_is_dims_4d_);

		std::vector<size_t> is_dims_3d = reg_is_dims_4d_;
		is_dims_3d.pop_back();

		intd3 is_dims_lores( is_dims_3d[0], is_dims_3d[1], is_dims_3d[2]);

		int num_3d_elements = is_dims_3d[0]*is_dims_3d[1]*is_dims_3d[2];

		for (unsigned int b=0, bin=0; b < this->binning_->get_number_of_bins(); b++) {

			if( b==phase_ )
				continue;

			hoCuNDArray<float> image_3d(&is_dims_3d, image_4d.get_data_ptr()+bin*num_3d_elements);

			conebeam_backwards_projection<false>
			( projections, &image_3d,
					acquisition_->get_geometry()->get_angles(),
					acquisition_->get_geometry()->get_offsets(),
					binning_->get_bin(b),
					projections_per_batch_, is_dims_lores, is_dims_in_mm_, ps_dims_in_mm,
					SDD, SAD, false, use_offset_correction_, false );

			// Gaussian weighing of result according to the cyclic distance to the reference frame
			//

			const float overall_scale = float(1)/this->gaussian(float(0));
			int dist = (phase_-b);
			float scale;
			if( abs(dist) <= (this->binning_->get_number_of_bins()>>1) )
				scale = overall_scale*this->gaussian(float(dist));
			else
				scale = overall_scale*this->gaussian(float(this->binning_->get_number_of_bins())-float(abs(dist)));

			image_3d *= scale;

			bin++;
		}

		// Spply the adjoint of the resampling matrix
		// - upsampling required if the vector field was initially downsampled
		//

		if( image_4d.get_size(0) == image->get_size(0) )
			R_->mult_MH( &image_4d, image, accumulate );
		else{
			// Upsampling required
			unsigned int num_upsamples = log2l(image->get_size(0)/reg_is_dims_4d_[0]);
			std::vector<size_t> dims_3d = reg_is_dims_4d_;
			dims_3d.pop_back();
			hoCuNDArray<float> image_lowres(&dims_3d);
			R_->mult_MH( &image_4d, &image_lowres );
			cuNDArray<float> tmp_in(&image_lowres);
			for( unsigned int d=0; d<num_upsamples; d++ ){
				for( unsigned int i=0; i<3; i++ ) dims_3d[i]*=2;
				cuNDArray<float> tmp_out(&dims_3d);
				cuUpsampleOperator<float,3> U;
				U.mult_M( &tmp_in, &tmp_out ); // this is the transpose of the downsampling by design
				tmp_in = tmp_out;
			}
			if( accumulate )
				*image += *tmp_in.to_host();
			else
				*image = *tmp_in.to_host();
		}
	}


protected:
	boost::shared_ptr< hoLinearResampleOperator_eigen<float,3> > R_;
	boost::shared_ptr<CBCT_acquisition> acquisition_;
	boost::shared_ptr<CBCT_binning> binning_;
	std::vector<size_t> reg_is_dims_4d_;
	int phase_;
	floatd3 is_dims_in_mm_;
	float samples_per_pixel_;
	unsigned int projections_per_batch_;
	bool use_offset_correction_;
	bool allow_offset_correction_override_;
	bool displacements_set_;
	bool preprocessed_;

private:

	inline float gaussian( float x )
	{
		const float sigma = std::sqrt(float(5.0));
		const float pi = float(4)*std::atan(float(1));
		const float a = float(1)/(sigma*std::sqrt(float(2)*pi));
		return a*std::exp(-float(0.5)*(x/sigma)*(x/sigma));
	}
};
}
