#include "splineBackprojectionOperator.h"
#include "vector_td_utilities.h"
#include "vector_td_io.h"
#include "cuNDArray_math.h"
#include "cuNDArray_reductions.h"
#include "cuGaussianFilterOperator.h"
#include "check_CUDA.h"

#include <vector>

#include <stdio.h>
#include "hoNDArray_fileio.h"
#include "hoCuNDArray_math.h"


#include "proton_utils.h"

#define MAX_THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 65535
#define MAX_BLOCKS 4096*4

namespace Gadgetron{



template<template<class> class ARRAY> void splineBackprojectionOperator<ARRAY>
::mult_M( ARRAY<float>* in_orig, ARRAY<float>* out_orig, bool accumulate ) {
	if( !in_orig || !out_orig){
		throw std::runtime_error( "cuOperatorPathBackprojection: mult_M empty data pointer");
	}
	ARRAY<float>* out = out_orig;
	if (accumulate) out = new ARRAY<float>(out_orig->get_dimensions());
	clear(out);

	ARRAY<float>* in = in_orig;
	/*
	if (data->get_hull()){
		in = new ARRAY<float>(*in_orig);
		*in *= *data->get_hull();
	}
*/
	protonProjection(in,out,data->get_splines().get(),physical_dims,data->get_EPL().get());

	if (data->get_weights() && use_weights){
		*out *= *data->get_weights();
		std::cout << "Using weights" << std::endl;
	}
 	if (accumulate){
		*out_orig += *out;
		delete out;
	}
 	/*
	if (data->get_hull())
		delete in;
		*/

}

template<template<class> class ARRAY> void splineBackprojectionOperator<ARRAY>
::mult_MH( ARRAY<float>* in_orig, ARRAY<float>* out_orig, bool accumulate ) {
	if( !in_orig || !out_orig){
		throw std::runtime_error("cuOperatorPathBackprojection: mult_MH empty data pointer");
	}
	ARRAY<float>* out = out_orig;
	if (accumulate) out = new ARRAY<float>(out_orig->get_dimensions());

	clear(out);

	ARRAY<float>* in = in_orig;
	if (data->get_weights() && use_weights){
		in = new ARRAY<float>(*in_orig);
		*in *= *data->get_weights();
	}
	protonBackprojection(out,in,data->get_splines().get(),physical_dims,data->get_EPL().get());
	//if (data->get_hull()) *out *= *data->get_hull();
	if (accumulate){
		*out_orig += *out;
	}
	if (accumulate) delete out;

	if (data->get_weights() && use_weights) delete in;
}



// Instantiations
template class splineBackprojectionOperator<cuNDArray>;
template class splineBackprojectionOperator<hoCuNDArray>;
}
