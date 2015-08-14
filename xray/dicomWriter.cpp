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
#include <gdcm/gdcmImage.h>
#include <gdcm/gdcmImageWriter.h>
#include <gdcm/gdcmFileDerivation.h>
#include <gdcm/gdcmUIDGenerator.h>
#include <gdcm/gdcmAttribute.h>
using namespace Gadgetron;


static void CALL_DCMTK(OFCondition s){
	if (!s.good())
		throw std::runtime_error(s.text());
}
static float calculate_scaling(hoNDArray<float>* image){
	//float med = median(image);
	float med = 0.11;
	std::cout << "Median " << med  << std::endl;
	return 1000/med;
}
void Gadgetron::write_dicom(hoNDArray<float>* image, const std::string& command_line, floatd3 imageDimensions){


	std::vector<size_t > imgSize = * image->get_dimensions();


	gdcm::ImageWriter w;
	gdcm::Image& im = w.GetImage();
	gdcm::File& file = w.GetFile();
	gdcm::DataSet& ds = file.GetDataSet();

	uint16_t * pixel_data = new uint16_t[image->get_number_of_elements()];
	size_t len = image->get_number_of_elements();

	float * data = image->get_data_ptr();
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

	gdcm::Attribute<0x0008,0x0060> CT={"CT"};
	ds.Replace(CT.GetAsDataElement());

	gdcm::Attribute<0x0018,0x0050> sliceSpacing= {imageDimensions[2]/imgSize[2]};
	ds.Replace(sliceSpacing.GetAsDataElement());
	//Dicom has two seperate tags for slice spacing?
	gdcm::Attribute<0x0018,0x0088> sliceSpace= {imageDimensions[2]/imgSize[2]};
	ds.Replace(sliceSpace.GetAsDataElement());
	im.SetDataElement(sliceSpacing.GetAsDataElement());
	size_t frameSize = imgSize[0]*imgSize[1]*imgSize[2];

	for (auto i = 0u; i < image->get_size(3); i++){
		std::cout << "Writing frame " << i << std::endl;

		gdcm::DataElement pixeldata(gdcm::Tag(0x7fe0,0x0010));
		pixeldata.SetByteValue((char*)(pixel_data+i*frameSize),frameSize*sizeof(uint16_t));
		im.SetDataElement(pixeldata);

		std::stringstream commentS;
		commentS << std::setfill('0') << std::setw(2) << (i*100)/image->get_size(3) << "\%A";
		gdcm::Attribute<0x0020,0x4000> imageComment = {commentS.str().c_str() };
		ds.Replace(imageComment.GetAsDataElement());
		std::stringstream ss;
		ss << "fdk-" << std::setfill('0') << std::setw(4) << i+1 << ".dcm";
		w.SetFileName(ss.str().c_str());
		gdcm::Attribute<0x0020,0x0011> seriesNumber = {i+1};
		ds.Replace(seriesNumber.GetAsDataElement());

		if (!w.Write())
			throw std::runtime_error("Failed to write dicom data");


	}
	//--> Set the Image Data
	// ( Casting as 'unsigned char *' is just to avoid warnings.
	// It doesn't change the values. )
	//-> Write !


	/*
	std::vector<size_t> imageSize = *image->get_dimensions();
	OFCondition status;
	DcmTagKey key;
	DcmFileFormat dcmFile;
	DcmDataset *dataset = dcmFile.getDataset();

	//SOP class UID
	key.set(0x0008,0x0016);
	CALL_DCMTK(dataset->putAndInsertString(key,"1.2.840.10008.5.1.4.1.1.2"));


	//Study description
	key.set(0x0008,0x1010);
	CALL_DCMTK(dataset->putAndInsertString(key,command_line.c_str()));


	//Modality
	key.set(0x0008,0x0060);
	CALL_DCMTK(dataset->putAndInsertString(key,"CT"));

	//Content Qualification
	key.set(0x0018,0x9004);
	CALL_DCMTK(dataset->putAndInsertString(key,"RESEARCH"));

	//Reconstruction Algorithm
	key.set(0x0018,0x9315);
	CALL_DCMTK(dataset->putAndInsertString(key,"ITERATIVE"));

	//Rescale Intercept
	key.set(0x0028,0x1052);
	CALL_DCMTK(dataset->putAndInsertString(key,"-1000"));

	//Rescale slope
	key.set(0x0028,0x1053);
	CALL_DCMTK(dataset->putAndInsertString(key,"1"));


	{
		//Slice Thickness
		key.set(0x0018,0x0050);
		std::stringstream ss;
		ss << imageDimensions[2]/imageSize[2];
		CALL_DCMTK(dataset->putAndInsertString(key,ss.str().c_str()));
	}

	{
		//Pixel spacing
		key.set(0x0028,0x0030);
		std::stringstream ss;
		ss << imageDimensions[0]/imageSize[0] << "\\" << imageDimensions[1]/imageSize[1];
		CALL_DCMTK(dataset->putAndInsertString(key,ss.str().c_str()));
	}


	{
		//Bits allocated
		key.set(0x0028,0x0100);
		CALL_DCMTK(dataset->putAndInsertString(key,"16"));
		//Bits stored
		key.set(0x0028,0x0101);
		CALL_DCMTK(dataset->putAndInsertString(key,"16"));
		//High bit
		key.set(0x00028,0x0102);
		CALL_DCMTK(dataset->putAndInsertString(key,"15"));
		// Pixel representation
		key.set(0x00028,0x0103);
		CALL_DCMTK(dataset->putAndInsertString(key,"1"));

	}

	{
		//Columns
		key.set(0x0028,0x0010);
		std::stringstream ss;
		ss << imageSize[0];
		CALL_DCMTK(dataset->putAndInsertString(key,ss.str().c_str()));
	}
	{
		//Columns
		key.set(0x0028,0x0011);
		std::stringstream ss;
		ss << imageSize[1];
		CALL_DCMTK(dataset->putAndInsertString(key,ss.str().c_str()));
	}
	{
		//Slices
		key.set(0x0028,0x0008);
		std::stringstream ss;
		ss << imageSize[2];
		CALL_DCMTK(dataset->putAndInsertString(key,ss.str().c_str()));
	}


	{
		// Series Instance UID
		key.set(0x0020,0x000E);
		char newuid[96];
		dcmGenerateUniqueIdentifier(newuid);
		CALL_DCMTK(dataset->putAndInsertString(key,newuid));
	}

	{
		std::vector<uint16_t> pixel_data(image->get_number_of_elements());
		std::cout << "Pixel size " << pixel_data.size() << std::endl;
		float * data = image->get_data_ptr();
		float scaling = calculate_scaling(image);
		for (size_t i = 0; i < pixel_data.size(); i++)
			pixel_data[i] = data[i]*scaling;
		key.set(0x7fe0,0x0010);
		CALL_DCMTK(dataset->putAndInsertUint16Array(key,pixel_data.data(),pixel_data.size()));

	}


	{
		//Date
		key.set(0x0008,0x0023);
		auto timepoint = std::chrono::system_clock::now();
		auto time = std::chrono::system_clock::to_time_t(timepoint);
		std::stringstream ss;
		ss << std::put_time(std::localtime(&time),"%Y%m%d");
		CALL_DCMTK(dataset->putAndInsertString(key,ss.str().c_str()));
		key.set(0x0008,0x0033);
		ss.clear();
		auto fractional = timepoint-std::chrono::system_clock::from_time_t(time);
		auto fracSecs  = std::chrono::duration_cast<std::chrono::microseconds>(fractional).count()/100;
		ss << std::put_time(std::localtime(&time),"%H%M%S") << "." <<  std::setfill('0') << std::setw(4) <<  fracSecs;
		CALL_DCMTK(dataset->putAndInsertString(key,ss.str().c_str()));

	}
	//CALL_DCMTK(dcmFile.saveFile("fdk.dcm"));
	std::string tmp = "fdk.dcm";
	CALL_DCMTK(dataset->saveFile(tmp.c_str()));
	 */


}



