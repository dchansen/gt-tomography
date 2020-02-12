//
// Created by dch on 19/12/16.
//
#pragma once
#ifndef GT_TOMOGRAPHY_PROJECTION_DOWNSAMPLER_H_H
#define GT_TOMOGRAPHY_PROJECTION_DOWNSAMPLER_H_H

#include <npp.h>

#include <cuNDArray.h>

namespace Gadgetron {

    cuNDArray<float> downsample_projections(cuNDArray<float> *image, float factorX, float factorY);

}
#endif //GT_TOMOGRAPHY_PROJECTION_DOWNSAMPLER_H_H


