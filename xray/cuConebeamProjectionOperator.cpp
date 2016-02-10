/*
 * cuConebeamProjectionOperator.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: u051747
 */

#include "cuConebeamProjectionOperator.h"
#include "conebeam_projection.h"
#include <boost/math/constants/constants.hpp>
#include "hoNDArray_math.h"
using boost::math::constants::pi;
namespace Gadgetron {

cuConebeamProjectionOperator::cuConebeamProjectionOperator() {
	// TODO Auto-generated constructor stub

	samples_per_pixel_ = 1.5f;
	allow_offset_correction_override_=true;

}
void cuConebeamProjectionOperator
::offset_correct(cuNDArray<float>* projections){

	if( !preprocessed_ ){
		throw std::runtime_error( "Error: cuConebeamProjectionOperator::offset_correct: setup not performed");
	}
	float SDD = acquisition_->get_geometry()->get_SDD();
	float SAD = acquisition_->get_geometry()->get_SAD();
	floatd2 ps_dims_in_mm = acquisition_->get_geometry()->get_FOV();
	apply_offset_correct( projections,acquisition_->get_geometry()->get_offsets(),ps_dims_in_mm, SDD, SAD);
}

cuConebeamProjectionOperator::~cuConebeamProjectionOperator() {
	// TODO Auto-generated destructor stub
}

void cuConebeamProjectionOperator::mult_M(cuNDArray<float>* input,
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

void cuConebeamProjectionOperator::mult_MH(cuNDArray<float>* input,
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

boost::shared_ptr<cuNDArray<bool>> cuConebeamProjectionOperator::calculate_mask( cuNDArray<float>* projections,float limit)
{
	auto dims = *this->get_domain_dimensions();
	std::vector<size_t> dims3d(dims.begin(),dims.end()-1);
	auto indims = *projections->get_dimensions();
	std::vector<size_t> inbindims(indims);
	float* input_ptr = projections->get_data_ptr();
	auto mask = boost::make_shared<cuNDArray<bool>>(dims);
	bool* mask_ptr = mask->get_data_ptr();
	for (int bin = 0; bin < binning_->get_number_of_bins(); bin++){
		//Check for empty bins
		if (binning_->get_bin(bin).size() == 0)
			continue;

		cuNDArray<bool> mask_view(dims3d,mask_ptr);
		inbindims.back() = angles[bin].size();
		auto input_view = boost::make_shared<cuNDArray<float>>(inbindims,input_ptr);

		vector_td<int,3> is_dims_in_pixels{dims3d[0], dims3d[1],dims3d[2]};

		conebeam_spacecarver(input_view.get(),&mask_view, angles[bin],offsets[bin],is_dims_in_pixels, is_dims_in_mm_,acquisition_->get_geometry()->get_FOV(),acquisition_->get_geometry()->get_SDD(),acquisition_->get_geometry()->get_SAD(),limit);

		input_ptr += input_view->get_number_of_elements();
		mask_ptr += mask_view.get_number_of_elements();

	}

	return mask;

}
void Gadgetron::cuConebeamProjectionOperator::setup( boost::shared_ptr<CBCT_acquisition> acquisition,
		floatd3 is_dims_in_mm, bool transform_angles )
{
	acquisition_ = acquisition;
	is_dims_in_mm_ = is_dims_in_mm;

	// Determine the minimum and maximum angles scanned and transform array angles from [0;max_angle_].
	//

	std::vector<float> &all_angles = acquisition->get_geometry()->get_angles();
	if (transform_angles){
		float min_value = *std::min_element(all_angles.begin(), all_angles.end() );
		transform(all_angles.begin(), all_angles.end(), all_angles.begin(), bind2nd(std::minus<float>(), min_value));
	}
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
	floatd2 mean_offset = std::accumulate(all_offsets.begin(),all_offsets.end(),floatd2(0,0))/float(all_offsets.size());
	if( allow_offset_correction_override_ && std::abs(mean_offset[0]) > ps_dims_in_mm[0]*0.1f )
		use_offset_correction_ = true;

	std::cout << "Mean offset " << mean_offset << " Use offset correct: " << (use_offset_correction_ ? "true" : "false") << std::endl;
	preprocessed_ = true;
	if (!binning_){
		std::vector<unsigned int> bins(angles.size());
		std::iota(bins.begin(),bins.end(),0);
		binning_ = boost::make_shared<CBCT_binning>(std::vector<std::vector<unsigned int>>(1,bins));
	}

	angles = std::vector<std::vector<float>>();
	offsets = std::vector<std::vector<floatd2>>();
	for (int bin =0; bin < binning_->get_number_of_bins(); bin++){
		auto binvec = binning_->get_bin(bin);
		angles.push_back(std::vector<float>());
		offsets.push_back(std::vector<floatd2>());
		for (auto index : binvec){
			angles.back().push_back(all_angles[index]);
			offsets.back().push_back(all_offsets[index]);
		}

	}
	auto permutations = new_order(binning_->get_bins());
	auto proj = acquisition->get_projections();
	if (proj)
		*proj =	*permute_projections(proj,permutations);

}

void Gadgetron::cuConebeamProjectionOperator::setup( boost::shared_ptr<CBCT_acquisition> acquisition,
		boost::shared_ptr<CBCT_binning> binning,floatd3 is_dims_in_mm, 		bool transform_angles)

{

	binning_ = binning;
	setup( acquisition, is_dims_in_mm,transform_angles );
}

std::vector<unsigned int> Gadgetron::cuConebeamProjectionOperator::new_order(std::vector<std::vector<unsigned int>> bins){
	std::vector<unsigned int> result;
	for (auto & b : bins)
		for (auto p : b)
			result.push_back(p);
	return result;
}

boost::shared_ptr<hoCuNDArray<float> > Gadgetron::cuConebeamProjectionOperator::permute_projections(
		boost::shared_ptr<hoCuNDArray<float> > projections,
		std::vector<unsigned int>  & permutations) {

	std::vector<size_t> new_proj_dims = {projections->get_size(0),projections->get_size(1),permutations.size()};

	auto result = boost::make_shared<hoCuNDArray<float>>(new_proj_dims);

	size_t nproj = permutations.size();
	size_t proj_size = projections->get_size(0)*projections->get_size(1);

	float * res_ptr = result->get_data_ptr();
	float * proj_ptr = projections->get_data_ptr();

	for (unsigned int i = 0; i < nproj; i++){
		cudaMemcpy(res_ptr+i*proj_size,proj_ptr+proj_size*permutations[i],proj_size*sizeof(float),cudaMemcpyHostToHost);
	}
	return result;


}



} /* namespace Gadgetron */
