/*
 * dicomWriter.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: dch
 */


#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmdata/dcostrmb.h"
#include <chrono>
#include "hoNDArray_math.h"
#include "vector_td.h"
#include "dicomWriter.h"
#include <gdcmImage.h>
#include <gdcmImageWriter.h>
#include <gdcmFileDerivation.h>
#include <gdcmUIDGenerator.h>
#include <gdcmAttribute.h>
using namespace Gadgetron;




static void write_dicomCT(hoNDArray<float>* image, floatd3 imageDimensions, CT_acquisition * acquisition){
	std::vector<size_t > imgSize = * image->get_dimensions();




	uint16_t * pixel_data = new uint16_t[image->get_number_of_elements()];
	size_t len = image->get_number_of_elements();


	float scaling = calculate_scaling(image);
	float calibration = acquisition->calibration_factor;
	for (size_t i = 0; i < len; i++)
		pixel_data[i] = 1000*std::max((data[i]-calibration)/calibration,0.0f)+1024;


	uint16_t * data_ptr = pixel_data;

	for (size_t slice = 0; slice < image->get_size(2); slice++) {
		gdcm::ImageWriter w;
		gdcm::Image &im = w.GetImage();
		gdcm::File &file = w.GetFile();
		gdcm::DataSet &ds = file.GetDataSet();
		im.SetNumberOfDimensions(2);
		im.SetDimension(0, imgSize[0]);
		im.SetDimension(1, imgSize[1]);

		im.SetSpacing(0, 1.0);
		im.SetSpacing(1, 1.0);


		gdcm::PixelFormat pixelFormat = gdcm::PixelFormat::UINT16;
		im.SetPixelFormat(pixelFormat);
		im.SetPhotometricInterpretation(gdcm::PhotometricInterpretation::MONOCHROME2);
		im.SetTransferSyntax(gdcm::TransferSyntax::ExplicitVRLittleEndian);

		im.SetIntercept(-1024);

		gdcm::Attribute<0x0008, 0x0060> CT = {"CT"};
		ds.Replace(CT.GetAsDataElement());

		gdcm::Attribute<0x0018, 0x0050> sliceSpacing = {3};
		ds.Replace(sliceSpacing.GetAsDataElement());
		//Dicom has two seperate tags for slice spacing?
		gdcm::Attribute<0x0018, 0x0088> sliceSpace = {2};
		ds.Replace(sliceSpace.GetAsDataElement());
		im.SetDataElement(sliceSpacing.GetAsDataElement());
		size_t sliceSize = imgSize[0] * imgSize[1];

		std::cout << "Writing slice " << slice << std::endl;

		gdcm::DataElement pixeldata(gdcm::Tag(0x7fe0, 0x0010));
		pixeldata.SetByteValue((char *) (data_ptr), sliceSize * sizeof(uint16_t));
		im.SetDataElement(pixeldata);


		gdcm::Attribute<0x0020, 0x4000> imageComment = {"Site 3"};
		ds.Replace(imageComment.GetAsDataElement());
		std::stringstream ss;
		ss << "DICOM_" << std::setfill('0') << std::setw(3) << slice+1;
		w.SetFileName(ss.str().c_str());


		gdcm::Attribute<0x0020, 0x000D> SeriesInstanceUID = acquisition->SeriesInstanceUID;
		ds.Replace(SeriesInstanceUID.GetAsDataElement());

		gdcm::Attribute<0x0020, 0x000E> StudyInstanceUID = acquisition->StudyInstanceUID;
		ds.Replace(StudyInstanceUID.GetAsDataElement());

		gdcm::Attribute<0x0020, 0x0011> SeriesNumber = acquisition->SeriesNumber;
		ds.Replace(SeriesNumber.GetAsDataElement());


		gdcm::Attribute<0x0020,0x1041> sliceLocation = {0.0};
		ds.Replace(sliceLocation.GetAsDataElement());

		gdcm::Attribute<0x0020,0x0032> imagePosition = {0.0,0.0,0.0};
		ds.Replace(imagePosition.GetAsDataElement());


		if (!w.Write())
			throw std::runtime_error("Failed to write dicom data");

		data_ptr += sliceSize;
	}

	delete[] pixel_data;

}

void Gadgetron::write_dicom(hoNDArray<float>* image, const std::string& command_line, floatd3 imageDimensions){
	for ( unsigned int i = 0; i < image->get_size(3); i++)
		write_dicom_frame(image,command_line,imageDimensions,i);




}



