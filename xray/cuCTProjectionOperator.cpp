/*
 * cuCTProjectionOperator.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: u051747
 */

#include "cuCTProjectionOperator.h"
#include <boost/math/constants/constants.hpp>
#include "ct_projection.h"

using namespace boost::math::float_constants;
namespace Gadgetron {


cuCTProjectionOperator::cuCTProjectionOperator() {
    // TODO Auto-generated constructor stub

    samples_per_pixel_ = 1.5f;

}

cuCTProjectionOperator::~cuCTProjectionOperator() {
    // TODO Auto-generated destructor stub
}

void cuCTProjectionOperator::mult_M(cuNDArray<float>* input,
                                    cuNDArray<float> *output, bool accumulate) {

    auto dims = *input->get_dimensions();
    std::vector<size_t> dims3d(dims);

    if (dims3d.size() == 4) dims3d.pop_back();
    auto outdims = *output->get_dimensions();
    std::vector<size_t> outbindims(outdims);
    float *input_ptr = input->get_data_ptr();
    float *output_ptr = output->get_data_ptr();
    for (int bin = 0; bin < binning->get_number_of_bins(); bin++) {
        //Check for empty bins
        if (binning->get_bin(bin).size() == 0)
            continue;
        cuNDArray<float> input_view(dims3d, input_ptr);
        outbindims.back() = detector_focal_cyls[bin].size();
        auto output_view = boost::make_shared<cuNDArray<float>>(outbindims, output_ptr);
        auto output_view2 = output_view;

        //conebeam_forwards_projection(output_view2.get(),&input_view,angles[bin],offsets[bin],samples_per_pixel_,is_dims_in_mm_,acquisition_->get_geometry()->get_FOV(),acquisition_->get_geometry()->get_SDD(),acquisition_->get_geometry()->get_SAD(),accumulate);

        input_ptr += input_view.get_number_of_elements();
        output_ptr += output_view->get_number_of_elements();


    }

}

void cuCTProjectionOperator::mult_MH(cuNDArray<float>* input,
                                     cuNDArray<float> *output, bool accumulate) {
    auto dims = *output->get_dimensions();
    std::vector<size_t> dims3d = dims;
    if (dims3d.size() == 4) dims3d.pop_back();

    auto indims = *input->get_dimensions();
    std::vector<size_t> inbindims(indims);
    float *input_ptr = input->get_data_ptr();
    float *output_ptr = output->get_data_ptr();
    for (int bin = 0; bin < binning->get_number_of_bins(); bin++) {
        //Check for empty bins
        if (binning->get_bin(bin).size() == 0)
            continue;

        cuNDArray<float> output_view(dims3d, output_ptr);
        inbindims.back() = detector_focal_cyls[bin].size();
        auto input_view = cuNDArray<float>(inbindims, input_ptr);

        vector_td<int, 3> is_dims_in_pixels{dims3d[0], dims3d[1], dims3d[2]};

        ct_backwards_projection(&input_view, &output_view, detector_focal_cyls[bin], focal_offset_cyls[bin],
                                central_elements[bin], proj_indices[bin], is_dims_in_pixels, is_dims_in_mm,
                                ps_dims_in_pixels, ps_spacing, SDD, accumulate);
        input_ptr += input_view.get_number_of_elements();
        output_ptr += output_view.get_number_of_elements();

    }


}

    void Gadgetron::cuCTProjectionOperator::setup(boost::shared_ptr<CT_acquisition> acquisition,
                                                  floatd3 is_dims_in_mm)
{
    std::vector<unsigned int> bins(acquisition->projections.get_size(2));
    std::iota(bins.begin(), bins.end(), 0);
    auto tmp_binning = boost::make_shared<CBCT_binning>(std::vector<std::vector<unsigned int>>(1, bins));
    this->setup(acquisition, tmp_binning, is_dims_in_mm);
}

    std::vector<intd2> Gadgetron::cuCTProjectionOperator::calculate_slice_indices(CT_acquisition &acquisition) {

        floatd2 detectorSize = acquisition.geometry.detectorSize;
        auto &centralElements = acquisition.geometry.detectorCentralElement;
        auto &axialPosition = acquisition.geometry.detectorFocalCenterAxialPosition;

        auto image_dims = *this->get_domain_dimensions();
        auto proj_dims = *this->get_codomain_dimensions();


        std::vector<intd2> slice_indices(image_dims[2]);
        std::vector<float> start_point(centralElements.size());

        for (int i = 0; i < centralElements.size(); i++) {
            start_point[i] = axialPosition[i] + (centralElements[i][1] / float(proj_dims[1]) - 0.5) * detectorSize[1] -
                             detectorSize[1] / 2;
        }

        int projection_start = 0;
        int projection_stop = 0;
        for (int i = 0; i < slice_indices.size(); i++) {
            float slice_start = is_dims_in_mm[2] / image_dims[2] * (i - 0.5f) - is_dims_in_mm[2] / 2 - offset;
            float slice_stop = is_dims_in_mm[2] / image_dims[2] * (i + 0.5f) - is_dims_in_mm[2] / 2 - offset;
            while (slice_start > (start_point[projection_start] + detectorSize[1]))
                projection_start++;
            while (slice_stop < start_point[projection_start])
                projection_stop++;

            slice_indices[i][0] = projection_start;
            slice_indices[i][1] = projection_stop;
        }
        return slice_indices;

    }

    void Gadgetron::cuCTProjectionOperator::setup(boost::shared_ptr<CT_acquisition> acquisition,
                                                  boost::shared_ptr<CBCT_binning> binning, floatd3 is_dims_in_mm) {
        this->binning = binning;
        auto bins = binning->get_bins();
        this->is_dims_in_mm = is_dims_in_mm;

        //Variables needed for differewnt
        detector_focal_cyls = std::vector<std::vector<floatd3>>(bins.size());
        focal_offset_cyls = std::vector<std::vector<floatd3>>(bins.size());
        central_elements = std::vector<std::vector<floatd2>>(bins.size());
        proj_indices = std::vector<std::vector<intd2>>(bins.size());

        auto all_proj_indices = calculate_slice_indices(*acquisition);
        CT_geometry &geometry = acquisition->geometry;

        for (size_t b = 0; b < bins.size(); b++) {
            for (auto i : bins[b]) {
                detector_focal_cyls[b].emplace_back(geometry.detectorFocalCenterAngularPosition[i] * pi / 360,
                                                    geometry.detectorFocalCenterAxialPosition[i],
                                                    geometry.detectorFocalRadialDistance[i]);

                focal_offset_cyls[b].emplace_back(geometry.sourceAngularPositionShift[i] * pi / 360,
                                                  geometry.sourceAxialPositionShift[i],
                                                  geometry.sourceRadialDistanceShift[i]);

                central_elements[b].emplace_back(geometry.detectorCentralElement[i]);
                proj_indices[b].push_back(all_proj_indices[i]);
            }

        }
}




} /* namespace Gadgetron */
