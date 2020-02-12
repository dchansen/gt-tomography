#pragma once

#include <hoNDArray.h>

#include <vector_td.h>
#include "CT_acquisition.h"

namespace Gadgetron {

    void write_dicomCT(std::string dicomDir,hoNDArray<float>* image, floatd3 imageDimensions, CT_acquisition * acquisition, float offset,int skip_proj=0);
    void write_binaryCT(std::string dicomDir,hoNDArray<float>* image, floatd3 imageDimensions, CT_acquisition * acquisition, float offset,int skip_proj=0);
}