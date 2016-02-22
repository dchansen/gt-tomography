/*
 * cuCTProjectionOperator.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: u051747
 */

#include "cuCTProjectionOperator.h"
#include "conebeam_projection.h"
#include <boost/math/constants/constants.hpp>
#include "hoNDArray_math.h"
using boost::math::constants::pi;
namespace Gadgetron {


cuCTProjectionOperator::cuCTProjectionOperator() {
	// TODO Auto-generated constructor stub

	samples_per_pixel_ = 1.5f;

}
void cuCTProjectionOperator
::offset_correct(cuNDArray<float>* projections){

	if( !preprocessed_ ){
		throw std::runtime_error( "Error: cuCTProjectionOperator::offset_correct: setup not performed");
	}
	float SDD = acquisition_->get_geometry()->get_SDD();
	float SAD = acquisition_->get_geometry()->get_SAD();
	floatd2 ps_dims_in_mm = acquisition_->get_geometry()->get_FOV();
}

cuCTProjectionOperator::~cuCTProjectionOperator() {
	// TODO Auto-generated destructor stub
}

void cuCTProjectionOperator::mult_M(cuNDArray<float>* input,
		cuNDArray<float>* output, bool accumulate) {

	auto dims = *input->get_dimensions();
	std::vector<size_t> dims3d(dims);

	if (dims3d.size() ==4) dims3d.pop_back();
	auto outdims = *output->get_dimensions();
	std::vector<size_t> outbindims(outdims);
	float* input_ptr = input->get_data_ptr();
	float* output_ptr = output->get_data_ptr();
	for (int bin = 0; bin < binning_->get_number_of_bins(); bin++){
		//Check for empty bins
		if (binning_->get_bin(bin).size() == 0)
			continue;
		cuNDArray<float> input_view(dims3d,input_ptr);
		outbindims.back() = angles[bin].size();
		auto  output_view = boost::make_shared<cuNDArray<float>>(outbindims,output_ptr);
		auto output_view2 = output_view;
		if (use_offset_correction_ && accumulate){
			output_view2 = boost::make_shared<cuNDArray<float>>(outbindims);
			clear(output_view2.get());
		}

		conebeam_forwards_projection(output_view2.get(),&input_view,angles[bin],offsets[bin],samples_per_pixel_,is_dims_in_mm_,acquisition_->get_geometry()->get_FOV(),acquisition_->get_geometry()->get_SDD(),acquisition_->get_geometry()->get_SAD(),accumulate);

		if (use_offset_correction_){
			apply_offset_correct(output_view2.get(),offsets[bin],acquisition_->get_geometry()->get_FOV(),acquisition_->get_geometry()->get_SDD(),acquisition_->get_geometry()->get_SAD());
			if (accumulate)
				*output_view += *output_view2;
		}

		input_ptr += input_view.get_number_of_elements();
		output_ptr += output_view->get_number_of_elements();


	}

}

void cuCTProjectionOperator::mult_MH(cuNDArray<float>* input,
		cuNDArray<float>* output, bool accumulate) {
	auto dims = *output->get_dimensions();
	std::vector<size_t> dims3d = dims;
	if (dims3d.size() ==4) dims3d.pop_back();

	auto indims = *input->get_dimensions();
	std::vector<size_t> inbindims(indims);
	float* input_ptr = input->get_data_ptr();
	float* output_ptr = output->get_data_ptr();
	for (int bin = 0; bin < binning_->get_number_of_bins(); bin++){
		//Check for empty bins
		if (binning_->get_bin(bin).size() == 0)
			continue;

		cuNDArray<float> output_view(dims3d,output_ptr);
		inbindims.back() = angles[bin].size();
		auto input_view = boost::make_shared<cuNDArray<float>>(inbindims,input_ptr);
		auto input_view2 = input_view;
		if(use_offset_correction_)
			input_view2 = boost::make_shared<cuNDArray<float>>(*input_view);

		vector_td<int,3> is_dims_in_pixels{dims3d[0], dims3d[1],dims3d[2]};

		conebeam_backwards_projection(input_view2.get(),&output_view, angles[bin],offsets[bin],is_dims_in_pixels, is_dims_in_mm_,acquisition_->get_geometry()->get_FOV(),acquisition_->get_geometry()->get_SDD(),acquisition_->get_geometry()->get_SAD(),use_offset_correction_,accumulate);

		input_ptr += input_view->get_number_of_elements();
		output_ptr += output_view.get_number_of_elements();

	}

	if (mask)
		apply_mask(output,mask.get());

}
void Gadgetron::cuCTProjectionOperator::setup( boost::shared_ptr<CT_acquisition> acquisition,
		floatd3 is_dims_in_mm)
{
	acquisition_ = acquisition;
	is_dims_in_mm_ = is_dims_in_mm;


	// Are we in a short scan setup?
	// - we say yes if we have covered less than PI+3*delta radians
	//

	float angle_span = *std::max_element(all_angles.begin(), all_angles.end() );
	floatd2 ps_dims_in_mm = acquisition_->get_geometry()->get_FOV();
	float SDD = acquisition_->get_geometry()->get_SDD();
	float delta = std::atan(ps_dims_in_mm[0]/(2.0f*SDD)); // Fan angle

	if( angle_span*pi<float>()/180.0f > pi<float>()+3.0f*delta )
		short_scan_ = false;
	else
		short_scan_ = true;

	std::vector<floatd2> all_offsets = acquisition_->get_geometry()->get_offsets();

	preprocessed_ = true;
	if (!binning_){
		std::vector<unsigned int> bins(angles.size());
		std::iota(bins.begin(),bins.end(),0);
		binning_ = boost::make_shared<CBCT_binning>(std::vector<std::vector<unsigned int>>(1,bins));
	}
}
	void Gadgetron::cuCTProjectionOperator::calculate_slice_indices() {

		floatd2 detectorSize = acquisition_->geometry.detectorSize;
		auto & centralElements = acquisition_->geometry.detectorCentralElement;
		auto & axialPosition = acquisition_->geometry.detectorFocalCenterAxialPosition;

		auto image_dims = *this->get_domain_dimensions();
		auto proj_dims = *this->get_codomain_dimensions();


		std::vector<size_t> slice_indices(image_dims[2]);
        std::vector<float> start_point(centralElements.size());

		for (int i = 0; i < centralElements.size(); i++){
			start_point[i] = axialPosition[i][1]+(centralElements[i][1]/float(proj_dims[1])-0.5)*detectorSize[1]-detectorSize[1]/2;
		}

		int projection_start = 0;
		int projection_stop = 0;
		for (int i = 0; i < slice_indices.size(); i++){
			float slice_start = is_dims_in_mm_[2]/image_dims[2]*(i-0.5f)-is_dims_in_mm_[2]/2-offset_;
			float slice_stop = is_dims_in_mm_[2]/image_dims[2]*(i+0.5f)-is_dims_in_mm_[2]/2-offset_;
			while (slice_start > (start_point[projection_start]+detectorSize[1] ))
				projection_start++;
			while (slice_stop < start_point[projection_start])
				projection_stop++;

			slice_indices[i][0] = projection_start;
			slice_indices[i][1] = projection_stop;
		}
		return slice_indices;

	}
void Gadgetron::cuCTProjectionOperator::setup( boost::shared_ptr<CT_acquisition> acquisition,
		boost::shared_ptr<CBCT_binning> binning,floatd3 is_dims_in_mm)

{

	binning_ = binning;
	setup( acquisition, is_dims_in_mm,transform_angles );
}




} /* namespace Gadgetron */
