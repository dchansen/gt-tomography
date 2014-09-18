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
 * @param[in] exterior_path_lengths Array containing length from starting position to hull
 */
template<template<class> class ARRAY> void protonProjection(ARRAY<float>* image,ARRAY<float>* projections, ARRAY<floatd3>* splines, floatd3 phys_dims,ARRAY<float>* exterior_path_lengths=NULL );

/**
 *
 * @param[out] image Image to backproject into
 * @param[in] projections Proton projections to backproject
 * @param[in] splines Array containing the information for the cubic splines
 * @param[in] phys_dims Physical dimensions of the image in cm
 * @param[in] exterior_path_lengths Array containing length from starting position to hull
 */
template<template<class> class ARRAY> void protonBackprojection(ARRAY<float>* image,ARRAY<float>* projections, ARRAY<floatd3>* splines, floatd3 phys_dims,ARRAY<float>* exterior_path_lengths=NULL);


template<template<class> class ARRAY> void countProtonsPerVoxel(ARRAY<float>* counts,ARRAY<floatd3>* splines, floatd3 phys_dims,ARRAY<float>* exterior_path_lengths=NULL);

template<template<class> class ARRAY> void protonPathNorm(std::vector<size_t> img_dims, ARRAY<float>* projections, ARRAY<floatd3>* splines, floatd3 phys_dims,ARRAY<float>* exterior_path_lengths=NULL );

template<class T, unsigned int D> void pad_nearest( cuNDArray<T> *in, cuNDArray<T> *out );
}

