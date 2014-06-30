#pragma once
#include "cuNDArray.h"

#include "protonDataset.h"
namespace Gadgetron{

void rotate_splines(cuNDArray<floatd3> * splines,float angle);


/**
 *
 * @param[in] image Input image to project
 * @param[out] projections Output array to contain projections
 * @param[in] splines Array containing the information for the cubic splines
 * @param[in] phys_dims Physical dimensions of the image in cm
 */
template<template<class> class ARRAY> void protonProjection(ARRAY<float>* image,ARRAY<float>* projections, ARRAY<floatd3>* splines, floatd3 phys_dims);

/**
 *
 * @param[out] image Image to backproject into
 * @param[in] projections Proton projections to backproject
 * @param[in] splines Array containing the information for the cubic splines
 * @param[in] phys_dims Physical dimensions of the image in cm
 */
template<template<class> class ARRAY> void protonBackprojection(ARRAY<float>* image,ARRAY<float>* projections, ARRAY<floatd3>* splines, floatd3 phys_dims);

}
