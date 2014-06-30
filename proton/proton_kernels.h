#pragma once
#include "vector_td.h"

namespace Gadgetron{
template <class REAL> __global__ void forward_kernel(const REAL* __restrict__ image, REAL* __restrict__ projections,
		const vector_td<REAL,3> * __restrict__ splines,  const vector_td<REAL,3> dims,
		const typename intd<3>::Type ndims, const int proj_dim, const int offset);

template <class REAL> __global__ void backwards_kernel(const REAL* __restrict__ projections, REAL* __restrict__ image,
		const vector_td<REAL,3> * __restrict__ splines,  const vector_td<REAL,3> dims,
		const typename intd<3>::Type ndims, const int proj_dim, const int offset);

template <class REAL> __global__ void space_carver_kernel(const REAL* __restrict__ projections, REAL* __restrict__ image,
		const vector_td<REAL,3> * __restrict__ splines,  const vector_td<REAL,3> dims, REAL cutoff,
		const typename intd<3>::Type ndims, const int proj_dim, const int offset);

template <class REAL> __global__ void move_origin_kernel(vector_td<REAL,3> * splines,  const vector_td<REAL,3> origin,const unsigned int proj_dim, const unsigned int offset);

template <class REAL> __global__ void rotate_splines_kernel(vector_td<REAL,3> * splines, REAL angle, unsigned int total, unsigned int offset);
template <class REAL> __global__ void crop_splines_kernel(vector_td<REAL,3> * splines, REAL* projections, const  vector_td<REAL,3>  dims, const int proj_dim,const REAL background,int offset);
template <class REAL> __global__ void crop_splines_hull_kernel(vector_td<REAL,3> * splines, REAL* projections, REAL* hull_mask, const vector_td<int,3> ndims, const  vector_td<REAL,3>  dims, const int proj_dim,const REAL background,int offset);
template <class REAL> __global__ void rescale_directions_kernel(vector_td<REAL,3> * splines, REAL* projections, const  vector_td<REAL,3>  dims,  const int proj_dim, const int offset);


template <class REAL> __global__ void points_to_coefficients(vector_td<REAL,3> * splines, int dim,int offset);

template <class REAL> __global__ void spline_trapz_kernel(vector_td<REAL,3> * splines, REAL* lengths, int dim, int offset);
template <class REAL> __global__ void length_correction_kernel(vector_td<REAL,3> * splines, REAL* projections, int dim, int offset);
}
