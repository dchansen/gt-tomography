/*
 * dicomWriter.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: dch
 */


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



static float calculate_scaling(hoNDArray<float>* image){
	//float med = median(image);
	float med = 0.02;
	std::cout << "Median " << med  << std::endl;
	return 1000/med;
}


static void write_dicom_frame(hoNDArray<float>* image, const std::string& command_line, floatd3 imageDimensions, unsigned int frame){
	std::vector<size_t > imgSize = * image->get_dimensions();


	gdcm::ImageWriter w;
	gdcm::Image& im = w.GetImage();
	gdcm::File& file = w.GetFile();
	gdcm::DataSet& ds = file.GetDataSet();

	uint16_t * pixel_data = new uint16_t[image->get_number_of_elements()/image->get_size(3)];
	size_t len = image->get_number_of_elements()/image->get_size(3);

	float * data = image->get_data_ptr()+len*frame;
	float scaling = calculate_scaling(image);
	for (size_t i = 0; i < len; i++)
		pixel_data[i] = std::max(data[i]*scaling,0.0f);

	im.SetNumberOfDimensions(3);
	im.SetDimension(0,imgSize[0]);
	im.SetDimension(1,imgSize[1]);
	im.SetDimension(2,imgSize[2]);

	im.SetSpacing(0,imageDimensions[0]/imgSize[0]);
	im.SetSpacing(1,imageDimensions[1]/imgSize[1]);
	im.SetSpacing(2,imageDimensions[2]/imgSize[2]);


	gdcm::PixelFormat pixelFormat = gdcm::PixelFormat::UINT16;
	im.SetPixelFormat(pixelFormat);
	im.SetPhotometricInterpretation(gdcm::PhotometricInterpretation::MONOCHROME2);
	im.SetTransferSyntax(gdcm::TransferSyntax::ExplicitVRLittleEndian);

	im.SetIntercept(-1000);

	gdcm::Attribute<0x0008,0x0060> CT={"CT"};
	ds.Replace(CT.GetAsDataElement());

	gdcm::Attribute<0x0018,0x0050> sliceSpacing= {imageDimensions[2]/imgSize[2]};
	ds.Replace(sliceSpacing.GetAsDataElement());
	//Dicom has two seperate tags for slice spacing?
	gdcm::Attribute<0x0018,0x0088> sliceSpace= {imageDimensions[2]/imgSize[2]};
	ds.Replace(sliceSpace.GetAsDataElement());
	im.SetDataElement(sliceSpacing.GetAsDataElement());
	size_t frameSize = imgSize[0]*imgSize[1]*imgSize[2];

	gdcm::Attribute<0x0020,0x0105> num_temporal_positions = {image->get_size(3)};
	ds.Replace(num_temporal_positions.GetAsDataElement());

	std::cout << "Writing frame " << frame << std::endl;

	gdcm::DataElement pixeldata(gdcm::Tag(0x7fe0,0x0010));
	pixeldata.SetByteValue((char*)(pixel_data),frameSize*sizeof(uint16_t));
	im.SetDataElement(pixeldata);

	std::stringstream commentS;
	commentS << std::setfill('0') << std::setw(2) << (frame*100)/image->get_size(3) << "\%A";
	gdcm::Attribute<0x0020,0x4000> imageComment = {commentS.str().c_str() };
	ds.Replace(imageComment.GetAsDataElement());
	std::stringstream ss;
	ss << "fdk-" << std::setfill('0') << std::setw(4) << frame+1 << ".dcm";
	w.SetFileName(ss.str().c_str());
	gdcm::Attribute<0x0020,0x0100> temporalNumber = {frame+1};
	ds.Replace(temporalNumber.GetAsDataElement());

	gdcm::Attribute<0x0020,0x0011> seriesNumber = {frame+1};
	ds.Replace(seriesNumber.GetAsDataElement());

	if (!w.Write())
		throw std::runtime_error("Failed to write dicom data");



}

void Gadgetron::write_dicom(hoNDArray<float>* image, const std::string& command_line, floatd3 imageDimensions){
	for ( unsigned int i = 0; i < image->get_size(3); i++)
		write_dicom_frame(image,command_line,imageDimensions,i);




}



