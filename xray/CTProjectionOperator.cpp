/*
 * cuCTProjectionOperator.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: u051747
 */

#include "CTProjectionOperator.h"
#include <boost/math/constants/constants.hpp>
#include "ct_projection.h"
#include "hoNDArray_math.h"

using namespace boost::math::float_constants;
namespace Gadgetron {


    template<template<class> class ARRAY>
    CTProjectionOperator<ARRAY>::CTProjectionOperator() {
        // TODO Auto-generated constructor stub

        samples_per_pixel_ = 1.5f;

    }

    template<template<class> class ARRAY>
    CTProjectionOperator<ARRAY>::~CTProjectionOperator() {
        // TODO Auto-generated destructor stub
    }

    template<template<class> class ARRAY>
    void CTProjectionOperator<ARRAY>::mult_M(ARRAY<float> *input,
                                             ARRAY<float> *output, bool accumulate) {

        auto dims = *input->get_dimensions();
        std::vector<size_t> dims3d(dims);

        if (dims3d.size() == 4) dims3d.pop_back();
        auto outdims = *output->get_dimensions();
        std::vector<size_t> outbindims(outdims);
        float *input_ptr = input->get_data_ptr();
        ARRAY<float> *tmp_out = output;
        if (accumulate)
            tmp_out = new ARRAY<float>(output->get_dimensions());

        float *output_ptr = tmp_out->get_data_ptr();
        for (int bin = 0; bin < binning->get_number_of_bins(); bin++) {
            //Check for empty bins
            if (binning->get_bin(bin).size() == 0)
                continue;
            ARRAY<float> input_view(dims3d, input_ptr);
            outbindims.back() = detector_focal_cyls[bin].size();
            auto output_view = ARRAY<float>(outbindims, output_ptr);


            ct_forwards_projection(&output_view, &input_view, detector_focal_cyls[bin], focal_offset_cyls[bin],
                                   central_elements[bin], is_dims_in_mm, ps_spacing, ADD, samples_per_pixel_, false);
            //conebeam_forwards_projection(output_view2.get(),&input_view,angles[bin],offsets[bin],samples_per_pixel_,is_dims_in_mm_,acquisition_->get_geometry()->get_FOV(),acquisition_->get_geometry()->get_SDD(),acquisition_->get_geometry()->get_SAD(),accumulate);

            input_ptr += input_view.get_number_of_elements();
            output_ptr += output_view.get_number_of_elements();


        }
        if (weights)
            *tmp_out *= *weights;

        if (accumulate) {
            *output += *tmp_out;
            delete tmp_out;
        }


    }



    template<template<class> class ARRAY>
    void CTProjectionOperator<ARRAY>::mult_MH(ARRAY<float> *input,
                                              ARRAY<float> *output, bool accumulate) {

        ARRAY<float> *tmp_in = input;
        if (weights) {
            tmp_in = new ARRAY<float>(input);
            *tmp_in *= *weights;
        }
        auto dims = *output->get_dimensions();
        std::vector<size_t> dims3d = dims;
        if (dims3d.size() == 4) dims3d.pop_back();

        auto indims = *input->get_dimensions();
        std::vector<size_t> inbindims(indims);
        float *input_ptr = tmp_in->get_data_ptr();
        float *output_ptr = output->get_data_ptr();
        if (!accumulate)
            clear(output);
        for (int bin = 0; bin < binning->get_number_of_bins(); bin++) {
            //Check for empty bins
            if (binning->get_bin(bin).size() == 0)
                continue;

            ARRAY<float> output_view(dims3d, output_ptr);
            inbindims.back() = detector_focal_cyls[bin].size();
            auto input_view = ARRAY<float>(inbindims, input_ptr);

            vector_td<int, 3> is_dims_in_pixels{dims3d[0], dims3d[1], dims3d[2]};

            ct_backwards_projection(&input_view, &output_view, detector_focal_cyls[bin], focal_offset_cyls[bin],
                                    central_elements[bin], proj_indices[bin], is_dims_in_mm, ps_spacing, ADD,
                                    accumulate);
            input_ptr += input_view.get_number_of_elements();
            output_ptr += output_view.get_number_of_elements();
        }

        if (weights) {
            delete tmp_in;
        }


    }

    template<template<class> class ARRAY>
    void Gadgetron::CTProjectionOperator<ARRAY>::setup(boost::shared_ptr<CT_acquisition> acquisition,
                                                       floatd3 is_dims_in_mm) {
        std::vector<unsigned int> bins(acquisition->geometry.detectorFocalCenterAngularPosition.size());
        std::iota(bins.begin(), bins.end(), 0);
        auto tmp_binning = boost::make_shared<CBCT_binning>(std::vector<std::vector<unsigned int>>(1, bins));
        this->setup(acquisition, tmp_binning, is_dims_in_mm);
    }

    template<template<class> class ARRAY>
    std::vector<intd2> Gadgetron::CTProjectionOperator<ARRAY>::calculate_slice_indices(CT_acquisition &acquisition) {

        floatd2 detectorSize = acquisition.geometry.detectorSize;
        auto &centralElements = acquisition.geometry.detectorCentralElement;
        auto &axialPosition = acquisition.geometry.detectorFocalCenterAxialPosition;

        auto image_dims = *this->get_domain_dimensions();
        auto proj_dims = *this->get_codomain_dimensions();


        std::vector<intd2> slice_indices(image_dims[2]);
        std::fill(slice_indices.begin(), slice_indices.end(), intd2(0, 0));
        std::vector<float> start_point(centralElements.size() + 1);

        for (int i = 0; i < centralElements.size(); i++) {
            start_point[i] = axialPosition[i] + (centralElements[i][1] - float(proj_dims[1]) / 2) * detectorSize[1] -
                             detectorSize[1] * proj_dims[1] / 2;
            //std::cout << "Start point " << start_point[i] << std::endl;
        }
        start_point.back() = 2 * start_point[centralElements.size() - 1] - start_point[centralElements.size() - 2];

        int projection_start = 0;
        int projection_stop = 0;
        for (int i = 0; i < slice_indices.size(); i++) {
            float slice_start = is_dims_in_mm[2] / image_dims[2] * (i - 0.5f) - is_dims_in_mm[2] / 2;
            float slice_stop = is_dims_in_mm[2] / image_dims[2] * (i + 0.5f) - is_dims_in_mm[2] / 2;
            //std::cout << "Slice start " << slice_start << " slice end " << slice_stop << std::endl;
            while (slice_start > (start_point[projection_start] + detectorSize[1] * proj_dims[1])) {
                projection_start++;
                if (projection_start >= centralElements.size()) {
                    projection_start = centralElements.size();
                    break;
                }
            }
            while (slice_stop > start_point[projection_stop]) {

                projection_stop++;
                if (projection_stop >= centralElements.size()) {
                    projection_stop = centralElements.size();
                    break;
                }
            }


            slice_indices[i][0] = projection_start;
            slice_indices[i][1] = projection_stop;
            //std::cout << "Projection start stop " << projection_start << " " << projection_stop << " start point " << start_point[projection_start] << " " << start_point[projection_stop] << " " << detectorSize[1] << std::endl;
        }

        std::cout << "Indices size " << slice_indices.size() << std::endl;
        return slice_indices;

    }

    template<template<class> class ARRAY>
    void Gadgetron::CTProjectionOperator<ARRAY>::setup(boost::shared_ptr<CT_acquisition> acquisition,
                                                       boost::shared_ptr<ARRAY<float>> weights, floatd3 is_dims_in_mm) {
        this->weights = weights;
        setup(acquisition, is_dims_in_mm);
    }

    template<template<class> class ARRAY>
    void Gadgetron::CTProjectionOperator<ARRAY>::setup(boost::shared_ptr<CT_acquisition> acquisition,
                                                       boost::shared_ptr<CBCT_binning> binning, floatd3 is_dims_in_mm) {
        this->binning = binning;
        auto bins = binning->get_bins();
        this->is_dims_in_mm = is_dims_in_mm;

        //Variables needed for differewnt
        detector_focal_cyls = std::vector<std::vector<floatd3>>(bins.size());
        focal_offset_cyls = std::vector<std::vector<floatd3>>(bins.size());
        central_elements = std::vector<std::vector<floatd2>>(bins.size());
        proj_indices = std::vector<std::vector<intd2>>(bins.size());

        proj_indices[0] = calculate_slice_indices(*acquisition);
        CT_geometry &geometry = acquisition->geometry;
        ps_spacing = acquisition->geometry.detectorSize;
        ADD = acquisition->geometry.constantRadialDistance[0];

        if (bins.size() != 1) throw std::runtime_error("CT reconstruction does not fully support 4D data yet");
        for (size_t b = 0; b < bins.size(); b++) {
            for (auto i : bins[b]) {
                detector_focal_cyls[b].emplace_back(geometry.detectorFocalCenterAngularPosition[i],
                                                    geometry.detectorFocalRadialDistance[i],
                                                    geometry.detectorFocalCenterAxialPosition[i]);

                focal_offset_cyls[b].emplace_back(geometry.sourceAngularPositionShift[i],
                                                  geometry.sourceRadialDistanceShift[i],
                                                  geometry.sourceAxialPositionShift[i]);

                central_elements[b].push_back(geometry.detectorCentralElement[i]);
                //proj_indices[b].push_back(all_proj_indices[i]);
            }

        }
    }


    template
    class CTProjectionOperator<cuNDArray>;

    template
    class CTProjectionOperator<hoCuNDArray>;


} /* namespace Gadgetron */
