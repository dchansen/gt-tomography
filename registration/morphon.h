#pragma once

#include "cuNDArray.h"

boost::shared_ptr<Gadgetron::cuNDArray<float>> morphon(Gadgetron::cuNDArray<float>* moving, Gadgetron::cuNDArray<float>* fixed);