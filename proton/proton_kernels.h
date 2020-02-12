#pragma once
#include "vector_td.h"

namespace Gadgetron{



template<class REAL> struct forward_functor {
	forward_functor(const REAL* __restrict__ input_image, REAL* __restrict__ input_projections): image(input_image), projection(input_projections), res(0){};

	__device__  __inline__ void begin(const int proj_idx){};
	/**
	 *
	 * @param idx Image id
	 * @param length
	 */
	__device__ __inline__ void operator() (const int img_idx, const REAL length ){
		res += image[img_idx]*length;
	}

	__device__  __inline__ void final(const int proj_idx){
		projection[proj_idx] = res;
	}
	REAL res;
	const REAL* __restrict__ image;
	REAL* __restrict__ projection;
};

template<class REAL> struct backward_functor {
	backward_functor(REAL* __restrict__ input_image, REAL* __restrict__ input_projections): image(input_image), projection(input_projections){};
	/**
	 *
	 * @param idx Image id
	 * @param length
	 */
	__device__  __inline__ void begin(const int proj_idx){
		proj = projection[proj_idx];
	}

	__device__ __inline__ void operator() (const int img_idx, const REAL length ){
		atomicAdd(&(image[img_idx]),length*proj);
	}

	__device__  __inline__ void final(const int proj_idx){
	}
	REAL proj;
	REAL* __restrict__ image;
	const REAL* __restrict__ projection;
};


template<class REAL> struct backward_counting_functor {
	backward_counting_functor(REAL* __restrict__ input_image): image(input_image){};
	/**
	 *
	 * @param idx Image id
	 * @param length
	 */
	__device__  __inline__ void begin(const int proj_idx){
	}

	__device__ __inline__ void operator() (const int img_idx, const REAL length ){
		atomicAdd(&(image[img_idx]),REAL(1));
	}

	__device__  __inline__ void final(const int proj_idx){
	}
	REAL* __restrict__ image;

};

template<class REAL> struct forward_norm_functor {
	forward_norm_functor(REAL* __restrict__ input_projections): projection(input_projections), res(0){};

	__device__  __inline__ void begin(const int proj_idx){};
	/**
	 *
	 * @param idx Image id
	 * @param length
	 */
	__device__ __inline__ void operator() (const int img_idx, const REAL length ){
		//res += image[img_idx]*length;
		res += length*length;
	}

	__device__  __inline__ void final(const int proj_idx){
		projection[proj_idx] = res;
	}
	REAL res;
	REAL* __restrict__ projection;
};



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

template <class REAL, class OP> __global__ void path_kernel(OP op,
		const vector_td<REAL,3> * __restrict__ splines,  const vector_td<REAL,3> dims,
		const typename intd<3>::Type ndims, const int proj_dim, const int offset);

template <class REAL, class OP> __global__ void path_kernel2(OP op, const vector_td<REAL,3> * __restrict__ splines,  const REAL* __restrict__ space_lengths, const vector_td<REAL,3> dims,
		const intd3 ndims, const int proj_dim, const int offset);

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
