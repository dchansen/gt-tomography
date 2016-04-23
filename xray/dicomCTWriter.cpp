/*
 * dicomWriter.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: dch
 */
#include "dicomCTWriter.h"


#include <chrono>
#include "hoNDArray_math.h"
#include <hoNDArray_utils.h>
#include "vector_td.h"
#include <gdcmImage.h>
#include <gdcmImageWriter.h>
#include <gdcmFileDerivation.h>
#include <gdcmUIDGenerator.h>
#include <gdcmAttribute.h>
using namespace Gadgetron;



static hoNDArray<float> rollingAverageCT(hoNDArray<float> * image){

    const auto dims = *image->get_dimensions();
    auto newdims = dims;
    //1 pixels per new slice

    const int nslices = 3;
    const int nskip = 2;
    newdims[2] = (dims[2]-nslices+nskip)/nskip;



    hoNDArray<float> result(newdims);


    float* in_ptr = image->get_data_ptr();
    float* out_ptr = result.get_data_ptr();
    #pragma omp parallel for
    for (int z = 0; z < newdims[2]; z++)
        for (int y = 0; y < dims[1]; y++)
            for (int x = 0; x < dims[0]; x++){
                float val = 0;
                for (int offset = 0; offset < nslices; offset++)
                    val +=  in_ptr[x+y*dims[0]+(z*nskip+offset)*dims[1]*dims[0]];

                out_ptr[x+y*dims[0]+z*dims[1]*dims[0]]= val/nslices;
            }


    return result;




}


static hoNDArray<float> transpose(hoNDArray<float> * image){
	auto dims = *image->get_dimensions();


	hoNDArray<float> result(dims);

	float* in_ptr = image->get_data_ptr();
	float* out_ptr = result.get_data_ptr();

	for (int z = 0; z < dims[2]; z++)
		for (int y = 0; y < dims[1]; y++)
			for (int x = 0; x < dims[0]; x++){
				out_ptr[x+y*dims[0]+z*dims[1]*dims[0]]= in_ptr[dims[0]-x-1+(dims[1]-y-1)*dims[0]+z*dims[1]*dims[0]];
			}

	return result;
}

void Gadgetron::write_dicomCT(std::string dcmDir,hoNDArray<float>* input_image, floatd3 imageDimensions, CT_acquisition * acquisition, float offset){





    hoNDArray<float> cropped({512,512,input_image->get_size(2)});
    if (input_image->get_size(0) == 512 && input_image->get_size(1) == 512) {
        cropped = *input_image;
    } else {
        auto cropped_dim = uint64d3{512,512,input_image->get_size(2)};
        crop(cropped_dim,input_image,&cropped);
    }


    //Create moving average

    auto image = rollingAverageCT(&cropped);
	image = transpose(&image);

    std::vector<size_t > imgSize = * image.get_dimensions();

	uint16_t * pixel_data = new uint16_t[image.get_number_of_elements()];
	size_t len = image.get_number_of_elements();


    float* data = image.get_data_ptr();
	float calibration = acquisition->calibration_factor;
#pragma omp parallel for
	for (size_t i = 0; i < len; i++)
		pixel_data[i] = 1000*(data[i]-calibration)/calibration+1024;


	uint16_t * data_ptr = pixel_data;

	for (size_t slice = 0; slice < image.get_size(2); slice++) {
		gdcm::ImageWriter w;
		gdcm::Image &im = w.GetImage();
		gdcm::File &file = w.GetFile();
		gdcm::DataSet &ds = file.GetDataSet();
		im.SetNumberOfDimensions(2);
		im.SetDimension(0, imgSize[0]);
		im.SetDimension(1, imgSize[1]);

		im.SetSpacing(0, image.get_size(0)/imageDimensions[0]);
		im.SetSpacing(1, image.get_size(1)/imageDimensions[1]);


		gdcm::PixelFormat pixelFormat = gdcm::PixelFormat::UINT16;
		im.SetPixelFormat(pixelFormat);
		im.SetPhotometricInterpretation(gdcm::PhotometricInterpretation::MONOCHROME2);
		im.SetTransferSyntax(gdcm::TransferSyntax::ExplicitVRLittleEndian);

		im.SetIntercept(-1024);

		gdcm::Attribute<0x0008, 0x0060> CT = {"CT"};
		ds.Replace(CT.GetAsDataElement());

		gdcm::Attribute<0x0018, 0x0050> sliceSpacing = {3};
		ds.Replace(sliceSpacing.GetAsDataElement());
        im.SetDataElement(sliceSpacing.GetAsDataElement());
		//Dicom has two seperate tags for slice spacing?
		gdcm::Attribute<0x0018, 0x0088> sliceSpace = {2};
		ds.Replace(sliceSpace.GetAsDataElement());

        im.SetDataElement(sliceSpace.GetAsDataElement());
		size_t sliceSize = imgSize[0] * imgSize[1];

		std::cout << "Writing slice " << slice << std::endl;

		gdcm::DataElement pixeldata(gdcm::Tag(0x7fe0, 0x0010));
		pixeldata.SetByteValue((char *) (data_ptr), sliceSize * sizeof(uint16_t));
		im.SetDataElement(pixeldata);


		gdcm::Attribute<0x0020, 0x4000> imageComment = {"Site 3"};
		ds.Replace(imageComment.GetAsDataElement());
		std::stringstream ss;
		ss << dcmDir << "/DICOM_" << std::setfill('0') << std::setw(3) << slice+1;
		w.SetFileName(ss.str().c_str());


		gdcm::Attribute<0x0020, 0x000D> SeriesInstanceUID = {acquisition->SeriesInstanceUID};
		ds.Replace(SeriesInstanceUID.GetAsDataElement());

		gdcm::Attribute<0x0020, 0x000E> StudyInstanceUID = {acquisition->StudyInstanceUID};
		ds.Replace(StudyInstanceUID.GetAsDataElement());

		gdcm::Attribute<0x0020, 0x0011> SeriesNumber = {acquisition->SeriesNumber};
		ds.Replace(SeriesNumber.GetAsDataElement());


		gdcm::Attribute<0x0020,0x1041> sliceLocation = {slice*3-imageDimensions[2]/2+offset};
		ds.Replace(sliceLocation.GetAsDataElement());

		gdcm::Attribute<0x0020,0x0032> imagePosition = {0.0,0.0,sliceLocation.GetValue()};
		ds.Replace(imagePosition.GetAsDataElement());


		if (!w.Write())
			throw std::runtime_error("Failed to write dicom data");

		data_ptr += sliceSize;
	}

	delete[] pixel_data;

}


