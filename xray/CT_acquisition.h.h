#pragma once
#include "hoCuNDArray.h"
#include <gdcm/gdcmReader.h>
#include <gdcm/gdcmFile.h>
#include <gdcm/gdcmFileMetaInformation.h>

namespace Gadgetron {
    struct CT_geometry {
        std::vector<float> angles;
        floatd2 detectorsize;

        std::vector<float> detectorFocalCenterAngularPosition;
        std::vector<float> detectorFocalCenterAxialPosition;
        std::vector<float> detectorFocalRadialDistance;
        std::vector<float> constantRadialDistance;
        std::vector<uintd2> detectorCentralElement;
        std::vector<float> sourceAngularPositionShift;
        std::vector<float> sourceAxialPositionShift;
        std::vector<float> sourceRadialDistanceShift;
    };

    class CT_acquisition {
    public:
        hoCuNDArray<float> projections;
        CT_geometry geometry;


    };


    boost::shared_ptr<CT_acquisition> read_dicom_projections(std::vector<std::string> files) {

        boost::shared_ptr<CT_acquisition> result = boost::make_shared<CT_acquisition>();
        {
            gdcm::Reader reader;
            reader.SetFileName(files[0]);
            if (!reader.Read()) throw std::runtime_error("Could not read dicom file");
            gdcm::File &file = reader.GetFile();
            gdcm::FileMetaInformation &header = file.getHeader();
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
            if (detectorShape.GetValue() != "CYLINDRICAL") throw std::runtime_error("Detector not cylindrical!");
        }
        CT_geometry & geometry = result->geometry;
        for (auto file : files){
            gdcm::Reader reader;
            reader.SetFileName(file);
            if (!reader.Read()) throw std::runtime_error("Could not read dicom file");
            gdcm::File &file = reader.GetFile();
            const gdcm::DataSet &ds = file.GetDataSet();

            gdcm::Attribute<0x7031,1001> detectorAngularPosition;
            detectorAngularPosition.Set(ds);
            geometry.detectorFocalCenterAngularPosition.push_back(detectorAngularPosition.GetValue());

            gdcm::Attribute<0x7031,1002> detectorAxialPosition;
            detectorAxialPosition.Set(ds);
            geometry.detectorFocalCenterAxialPosition.push_back(detectorAxialPosition.GetValue());

            gdcm::Attribute<0x7031,1003> detectorRadialDistance;
            detectorRadialDistance.Set(ds);
            geometry.detectorFocalRadialDistance.push_back(detectorRadialDistance.GetValue());

            gdcm::Attribute<0x7031,1031> constantRadialDistance;
            constantRadialDistance.Set(ds);
            geometry.constantRadialDistance.push_back(constantRadialDistance.GetValue());

            gdcm::Attribute<0x7031,1033> centralElement;
            centralElement.Set(ds);
            geometry.detectorCentralElement.push_back(centralElement.GetValue());

            gdcm::Attribute<0x7033,0x100B> sourceAngularPositionShift;
            sourceAngularPositionShift.Set(ds);
            geometry.sourceAngularPositionShift.push_back(sourceAngularPositionShift.GetValue());

            gdcm::Attribute<0x7033,0x100C>  sourceAxialPositionShift;
            sourceAxialPositionShift.Set(ds);
            geometry.sourceAxialPositionShift.push_back(sourceAxialPositionShift.GetValue());

            gdcm::Attribute<0x7033,0x100D> sourceRadialDistanceShift;
            sourceRadialDistanceShift.Set(ds);
            geometry.sourceRadialDistanceShift.push_back(sourceRadialDistanceShift.GetValue());



        }


    }

}