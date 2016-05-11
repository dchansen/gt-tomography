#include "CT_acquisition.h"
#include "vector_td_io.h"
using namespace Gadgetron;
static std::vector<double> extract_axial_position(std::vector<std::string> files){
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

static std::vector<std::string> sort_dicoms(std::vector<std::string> files){
    auto axial_pos = extract_axial_position(files);

    thrust::sort_by_key(thrust::host,axial_pos.begin(),axial_pos.end(),files.begin());
    return files;

}

boost::shared_ptr<CT_acquisition> Gadgetron::read_dicom_projections(std::vector<std::string> files_in, unsigned int remove_projs) {

    using namespace boost::math::double_constants;
    auto files = sort_dicoms(files_in);

    if (remove_projs > 0){
        files = std::vector<std::string>(files.begin()+remove_projs,files.end()-remove_projs);
    }

    std::cout << "Number of dicoms " << files.size() << std::endl;



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
        result->photonStatistics = hoCuNDArray<float>({projection_dims[0],1,files.size()});


        gdcm::Attribute<0x7029, 0x1002> detectorsizeX;
        detectorsizeX.Set(ds);
        gdcm::Attribute<0x7029, 0x1006> detectorsizeY;
        detectorsizeY.Set(ds);

        result->geometry.detectorSize = floatd2(detectorsizeX.GetValue(), detectorsizeY.GetValue());
        std::cout << "Detector element size " << result->geometry.detectorSize << std::endl;


        gdcm::Attribute<0x7029, 0x100B> detectorShape;
        detectorShape.Set(ds);
        if (detectorShape.GetValue() != "CYLINDRICAL ") {
            std::cout << "Detector shape: " << detectorShape.GetValue() << "X" << std::endl;
            throw std::runtime_error("Detector not cylindrical!");
        }

        gdcm::Attribute<0x0018,0x0061> calibrationFactor;
        calibrationFactor.Set(ds);
        result->calibration_factor = calibrationFactor.GetValue();
        std::cout << "Calibration factor " << result->calibration_factor << std::endl;

        gdcm::Attribute<0x0020,0x000D> StudyInstanceUID;
        StudyInstanceUID.Set(ds);
        result->StudyInstanceUID = StudyInstanceUID.GetValue();

        gdcm::Attribute<0x0020,0x000E> SeriesInstanceUID;
        SeriesInstanceUID.Set(ds);
        result->SeriesInstanceUID = SeriesInstanceUID.GetValue();

        gdcm::Attribute<0x0020,0x0011> SeriesNumber;
        SeriesNumber.Set(ds);
        result->SeriesNumber = SeriesNumber.GetValue();


        gdcm::Attribute<0x0018,0x0090> FOV;
        FOV.Set(ds);
        std::cout << "Scan FOV " << FOV.GetValue() << std::endl;
    }

    CT_geometry * geometry = &result->geometry;
    float* projectionPtr = result->projections.get_data_ptr();

    float* weightsPtr = result->photonStatistics.get_data_ptr();

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
        geometry->detectorCentralElement.push_back(floatd2(centralElement.GetValue(0)-1,centralElement.GetValue(1)-1)); //Real men start at index 0


        gdcm::Attribute<0x7033,0x100C> sourceAngularPositionShift;
        sourceAngularPositionShift.Set(ds);
        geometry->sourceAngularPositionShift.push_back(sourceAngularPositionShift.GetValue());

        //geometry->sourceAngularPositionShift.push_back(0);

        gdcm::Attribute<0x7033,0x100B>  sourceAxialPositionShift;
        sourceAxialPositionShift.Set(ds);
        //geometry->sourceAxialPositionShift.push_back(0);
        geometry->sourceAxialPositionShift.push_back(sourceAxialPositionShift.GetValue());


        gdcm::Attribute<0x7033,0x100D> sourceRadialDistanceShift;
        sourceRadialDistanceShift.Set(ds);
        geometry->sourceRadialDistanceShift.push_back(sourceRadialDistanceShift.GetValue());

        //geometry->sourceRadialDistanceShift.push_back(0);


        gdcm::Attribute<0x7033,0x1065> photonStatistics;
        photonStatistics.Set(ds);

        for (int i = 0; i < result->photonStatistics.get_size(0); i++){
            weightsPtr[i] = photonStatistics[i];
        }

        weightsPtr += result->photonStatistics.get_size(0);


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
        #pragma omp parallel for
        for (size_t y = 0; y <ydim; y++) {
            for (size_t x = 0; x < xdim; x++) {
                projectionPtr[x+y*xdim] = float(bufferptr[y+x*ydim]) * slope + intercept;
            }
        }

        projectionPtr += elements;

    }

    return result;

}

void Gadgetron::downscale_projections(boost::shared_ptr<CT_acquisition> acq) {

    hoCuNDArray<float>& projections = acq->projections;



    const size_t nrows = projections.get_size(0);
    const size_t ncols = projections.get_size(1);
    const size_t nprojs = projections.get_size(2);
    hoCuNDArray<float> proj2({nrows,ncols/2,nprojs});

    float* proj2_ptr = proj2.get_data_ptr();
    float* proj_ptr= projections.get_data_ptr();
    #pragma omp parallel for
    for (int proj = 0; proj < nprojs; proj++)
        for (int col = 0; col < ncols/2; col++ )
            for (int row = 0; row < nrows; row++) {
                proj2_ptr[row+col*nrows+proj*nrows*ncols/2] = log(exp(proj_ptr[row+2*col*nrows+proj*nrows*ncols])+exp(proj_ptr[row+(2*col+1)*nrows+proj*nrows*ncols]))/2;
            }


    acq->projections = std::move(proj2);

    acq->geometry.detectorSize[1] /= 2;

    std::vector<floatd2>& centralElements = acq->geometry.detectorCentralElement;

    for (auto & f : centralElements ){
        f[1] /= 2.0f;
    }

}