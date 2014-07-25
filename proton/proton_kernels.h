#pragma once
#include "vector_td.h"

namespace Gadgetron{
/**
 * Calculates the forwards projection (Ax) of the image by integrating the cubic splines through.
 * @param[in] image
 * @param[out] projections
 * @param[in] splines
 * @param[in] dims
 * @param[in] ndims
 * @param[in] proj_dim
 * @param[in] offset
 */
template <class REAL> __global__ void forward_kernel(const REAL* __restrict__ image, REAL* __restrict__ projections,
		const vector_td<REAL,3> * __restrict__ splines,  const vector_td<REAL,3> dims,
		const typename intd<3>::Type ndims, const int proj_dim, const int offset);

/**
 * Calculates the forwards projection (Ax) of the image by integrating the cubic splines through.
 * Uses straight line segments for the first and last parts, based on the space_lengths
 * @param image
 * @param projections
 * @param splines
 * @param space_lengths
 * @param dims
 * @param ndims
 * @param proj_dim
 * @param offset
 */
template <class REAL> __global__ void forward_kernel2(const REAL* __restrict__ image, REAL* __restrict__ projections,
		const vector_td<REAL,3> * __restrict__ splines, const REAL * __restrict__ space_lengths, const vector_td<REAL,3> dims,
		const intd3 ndims, const int proj_dim, const int offset);

template <class REAL> __global__ void backwards_kernel(const REAL* __restrict__ projections, REAL* __restrict__ image,
		const vector_td<REAL,3> * __restrict__ splines,  const vector_td<REAL,3> dims,
		const typename intd<3>::Type ndims, const int proj_dim, const int offset);

template <class REAL> __global__ void backwards_kernel2(const REAL* __restrict__ projections, REAL* __restrict__ image,
		const vector_td<REAL,3> * __restrict__ splines, const REAL * __restrict__ space_lengths, const vector_td<REAL,3> dims,
		const typename intd<3>::Type ndims, const int proj_dim, const int offset);

template <class REAL> __global__ void space_carver_kernel(const REAL* __restrict__ projections, REAL* __restrict__ image,
		const vector_td<REAL,3> * __restrict__ splines,  const vector_td<REAL,3> dims, REAL cutoff,
		const typename intd<3>::Type ndims, const int proj_dim, const int offset);

/**
 * Calculates how long the straight line segments at the end and te beginning should be.
 * @param splines
 * @param space_lengths
 * @param hull_mask
 * @param ndims
 * @param dims
 * @param proj_dim
 * @param offset
 */
template <class REAL> __global__ void calc_spaceLengths_kernel(const vector_td<REAL,3> * __restrict__ splines, REAL* __restrict__ space_lengths,const REAL*  __restrict__ hull_mask,const vector_td<int,3> ndims, const  vector_td<REAL,3>  dims, const int proj_dim,int offset);

template <class REAL> __global__ void move_origin_kernel(vector_td<REAL,3> * splines,  const vector_td<REAL,3> origin,const unsigned int proj_dim, const unsigned int offset);

template <class REAL> __global__ void rotate_splines_kernel(vector_td<REAL,3> * splines, REAL angle, unsigned int total, unsigned int offset);
template <class REAL> __global__ void crop_splines_kernel(vector_td<REAL,3> * splines, REAL* projections, const  vector_td<REAL,3>  dims, const int proj_dim,const REAL background,int offset);
template <class REAL> __global__ void crop_splines_hull_kernel(vector_td<REAL,3> * splines, REAL* projections, REAL* hull_mask, const vector_td<int,3> ndims, const  vector_td<REAL,3>  dims, const int proj_dim,const REAL background,int offset);
template <class REAL> __global__ void rescale_directions_kernel(vector_td<REAL,3> * splines, REAL* projections, const  vector_td<REAL,3>  dims,  const int proj_dim, const int offset);


template <class REAL> __global__ void points_to_coefficients(vector_td<REAL,3> * splines, int dim,int offset);

template <class REAL> __global__ void spline_trapz_kernel(vector_td<REAL,3> * splines, REAL* lengths, int dim, int offset);
template <class REAL> __global__ void length_correction_kernel(vector_td<REAL,3> * splines, REAL* projections, int dim, int offset);
}
