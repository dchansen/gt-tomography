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
        CT_geometry() : angles(), detectorFocalCenterAngularPosition(), detectorFocalCenterAxialPosition(),
                        constantRadialDistance(), detectorCentralElement(), sourceAngularPositionShift(),
                        sourceAxialPositionShift(), sourceRadialDistanceShift(){
        }

        std::vector<float> angles;
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


    };

    std::vector<double> extract_axial_position(std::vector<std::string> files){
        std::vector<double> result;
        for (auto file : files){
            gdcm::Reader reader;
            reader.SetFileName(file.c_str());
            if (!reader.Read()) throw std::runtime_error("Could not read dicom file");
            auto & dcmFile = reader.GetFile();

            gdcm::Attribute<0x7031,0x1002> detectorAxialPosition;
            auto & ds = dcmFile.GetDataSet();
            detectorAxialPosition.Set(ds);
            result.push_back(detectorAxialPosition.GetValue());

        }
        return result;
    }

    std::vector<std::string> sort_dicoms(std::vector<std::string> files){
        auto axial_pos = extract_axial_position(files);

        thrust::sort_by_key(thrust::host,axial_pos.begin(),axial_pos.end(),files.begin());
        return files;

    }

    boost::shared_ptr<CT_acquisition> read_dicom_projections(std::vector<std::string> files) {

        using namespace boost::math::double_constants;
        files = sort_dicoms(files);
        boost::shared_ptr<CT_acquisition> result = boost::make_shared<CT_acquisition>();
        {
            gdcm::Reader reader;
            reader.SetFileName(files[0].c_str());
            if (!reader.Read()) throw std::runtime_error("Could not read dicom file");
            gdcm::File &file = reader.GetFile();
            gdcm::FileMetaInformation &header = file.GetHeader();
            const gdcm::DataSet &ds = file.GetDataSet();

            std::vector<size_t> projection_dims;

            gdcm::Attribute<0x7029, 0x1010> num_rows;
            num_rows.Set(ds);
            gdcm::Attribute<0x7029, 0x1011> num_cols;
            num_cols.Set(ds);

            projection_dims.push_back(num_cols.GetValue());
            projection_dims.push_back(num_rows.GetValue());
            projection_dims.push_back(files.size());

            result->projections = hoCuNDArray<float>(projection_dims);


            gdcm::Attribute<0x7029, 0x1002> detectorsizeY;
            detectorsizeY.Set(ds);
            gdcm::Attribute<0x7029, 0x1006> detectorsizeX;
            detectorsizeX.Set(ds);

            result->geometry.detectorSize = floatd2(detectorsizeX.GetValue(), detectorsizeY.GetValue());

            gdcm::Attribute<0x7029, 0x100B> detectorShape;
            detectorShape.Set(ds);
            if (detectorShape.GetValue() != "CYLINDRICAL ") {
                std::cout << "Detector shape: " << detectorShape.GetValue() << "X" << std::endl;
                throw std::runtime_error("Detector not cylindrical!");
            }
        }
        CT_geometry * geometry = &result->geometry;
        float* projectionPtr = result->projections.get_data_ptr();
        for (auto fileStr : files){
            gdcm::Reader reader;

            reader.SetFileName(fileStr.c_str());
            if (!reader.Read()) throw std::runtime_error("Could not read dicom file");
            gdcm::File &file = reader.GetFile();
            const gdcm::DataSet &ds = file.GetDataSet();

            gdcm::Attribute<0x7031,0x1001> detectorAngularPosition;
            detectorAngularPosition.Set(ds);
            //geometry->detectorFocalCenterAngularPosition.push_back((fmod(detectorAngularPosition.GetValue()+pi,2*pi)-pi));
           geometry->detectorFocalCenterAngularPosition.push_back(detectorAngularPosition.GetValue());

            gdcm::Attribute<0x7031,0x1002> detectorAxialPosition;
            detectorAxialPosition.Set(ds);
            geometry->detectorFocalCenterAxialPosition.push_back(detectorAxialPosition.GetValue());

            gdcm::Attribute<0x7031,0x1003> detectorRadialDistance;
            detectorRadialDistance.Set(ds);
            geometry->detectorFocalRadialDistance.push_back(detectorRadialDistance.GetValue());

            gdcm::Attribute<0x7031,0x1031> constantRadialDistance;
            constantRadialDistance.Set(ds);
            geometry->constantRadialDistance.push_back(constantRadialDistance.GetValue());

            gdcm::Attribute<0x7031,0x1033> centralElement;
            centralElement.Set(ds);
            geometry->detectorCentralElement.push_back(floatd2(centralElement.GetValue(0),centralElement.GetValue(1)));

            gdcm::Attribute<0x7033,0x100B> sourceAngularPositionShift;
            sourceAngularPositionShift.Set(ds);
            //geometry->sourceAngularPositionShift.push_back(sourceAngularPositionShift.GetValue());
            geometry->sourceAngularPositionShift.push_back(0);

            gdcm::Attribute<0x7033,0x100C>  sourceAxialPositionShift;
            sourceAxialPositionShift.Set(ds);
            geometry->sourceAxialPositionShift.push_back(0);
            //geometry->sourceAxialPositionShift.push_back(sourceAxialPositionShift.GetValue());

            gdcm::Attribute<0x7033,0x100D> sourceRadialDistanceShift;
            sourceRadialDistanceShift.Set(ds);
            //geometry->sourceRadialDistanceShift.push_back(sourceRadialDistanceShift.GetValue());
            geometry->sourceRadialDistanceShift.push_back(0);

            gdcm::ImageReader imReader;
            imReader.SetFileName(fileStr.c_str());
            if (!imReader.Read()) throw std::runtime_error("Unable to read image data");
            auto image = imReader.GetImage();

            auto imdims = image.GetDimensions();
            size_t elements = image.GetBufferLength()/sizeof(uint16_t);
            std::vector<uint16_t> buffer(elements);
            uint16_t * bufferptr = buffer.data();

            image.GetBuffer((char*)bufferptr);

            gdcm::Attribute<0x0028,0x1052> rescaleIntercept;
            rescaleIntercept.Set(ds);
            double intercept = rescaleIntercept.GetValue();

            gdcm::Attribute<0x0028,0x1053> rescaleSlope;
            rescaleSlope.Set(ds);
            double slope = rescaleSlope.GetValue();

            const size_t xdim = result->projections.get_size(0);
            const size_t ydim = result->projections.get_size(1);
            for (size_t y = 0; y <ydim; y++) {
                for (size_t x = 0; x < xdim; x++) {
                    projectionPtr[x+y*xdim] = double(bufferptr[y+x*ydim]) * slope + intercept;
                }
            }

            projectionPtr += elements;

        }

        return result;

    }

}