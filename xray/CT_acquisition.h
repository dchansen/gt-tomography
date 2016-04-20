#ifndef CT_ACQUISITION_H_
#define CT_ACQUISITION_H_
#pragma once
#include "hoCuNDArray.h"
#include <gdcmReader.h>
#include <gdcmFile.h>
#include <gdcmFileMetaInformation.h>
#include <gdcmAttribute.h>
#include <gdcmImageReader.h>

#include <boost/make_shared.hpp>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <boost/math/constants/constants.hpp>
namespace Gadgetron {
    struct CT_geometry {
        CT_geometry() : detectorFocalCenterAngularPosition(), detectorFocalCenterAxialPosition(),
                        constantRadialDistance(), detectorCentralElement(), sourceAngularPositionShift(),
                        sourceAxialPositionShift(), sourceRadialDistanceShift(){
        }


        floatd2 detectorSize;

        std::vector<float> detectorFocalCenterAngularPosition;
        std::vector<float> detectorFocalCenterAxialPosition;
        std::vector<float> detectorFocalRadialDistance;
        std::vector<float> constantRadialDistance;
        std::vector<floatd2> detectorCentralElement;
        std::vector<float> sourceAngularPositionShift;
        std::vector<float> sourceAxialPositionShift;
        std::vector<float> sourceRadialDistanceShift;
    };

    class CT_acquisition {
    public:
        CT_acquisition() : projections(), geometry(){

        }
        hoCuNDArray<float> projections;
        CT_geometry geometry;
        float calibration_factor;
        std::string StudyInstanceUID;
        std::string SeriesInstanceUID;
        size_t SeriesNumber;

    };


    boost::shared_ptr<CT_acquisition> read_dicom_projections(std::vector<std::string> files);


}
#endif