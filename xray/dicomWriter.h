#pragma once

#include "hoNDArray.h"
#include "cuNDArray.h"
#include "vector_td.h"

namespace Gadgetron {
void write_dicom(hoNDArray<float>* image, const std::string& command_line, floatd3 imageDimensions);


void write_dicom(cuNDArray<float>* image, const std::string& command_line, floatd3 imageDimensions){
	auto array = image->to_host();
	write_dicom(array.get(),command_line,imageDimensions);
}

}
