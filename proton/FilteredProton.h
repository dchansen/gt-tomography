#pragma once
#include "subsetOperator.h"
#include "hoCuNDArray.h"
#include "splineBackprojectionOperator.h"
#include "hdf5.h"
#include "hdf5_hl.h"
#include <numeric>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include "hoCuParallelProjection.h"
#include "hoNDFFT.h"
#include "cuNDFFT.h"
#include "cuNDArray_math.h"

#include "hoNDArray_fileio.h"
#include "cuNDArray_fileio.h"

#include <boost/math/constants/constants.hpp>

#include "proton_utils.h"
#include <boost/type_traits.hpp>
#include <boost/make_shared.hpp>

#define MAX_THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 65535
#define MAX_BLOCKS 4096*4

namespace Gadgetron{




template<template<class>  class ARRAY> class FilteredProton {

public:
	FilteredProton() {

	}


	//Terrible terrible name. Penguin_sauce would be as good...or better, 'cos penguin
	boost::shared_ptr<ARRAY<float> >  calculate(std::vector<size_t> dims,vector_td<float,3> physical_dims, boost::shared_ptr<protonDataset<ARRAY> > data, bool estimate_missing=true,float oversamplingWidth = 2.0f, float undersamplingDepth = 6.0f){

		boost::shared_ptr<cuNDArray<float> > image(new cuNDArray<float>(dims));
		clear(image.get());

		std::vector<size_t> dims_proj = { size_t(dims[0]*oversamplingWidth),size_t(dims[1]/undersamplingDepth),dims[2]};
		//std::vector<size_t> dims_proj;

		vector_td<float,3> physical_dims_proj = physical_dims;
		physical_dims_proj[0] *= std::sqrt(2.0f);
		//physical_dims_proj[1] *= std::sqrt(2.0f);
/*
		for (int i  = 0; i < dims.size(); i++)
		{
			if (dims[i] == 1) dims_proj.push_back(1);
			else{
				dims_proj.push_back(dims[i]*2);
				//dims_proj.push_back(dims[i]);
				physical_dims_proj[i] *= std::sqrt(2.0f);
				//physical_dims_proj[i] *= 0.5;
			}
		}*/


		unsigned int oversampling = 2;

		auto ramp = *calc_filter<float>(dims_proj[0]*oversampling,physical_dims[0]/(dims[0]));
		//fill(&ramp,complext<float>(1.0f,0));
		//ramp *= 2.0f*oversampling;
		unsigned int ngroups = data->get_number_of_groups();

		cuNDArray<float>* hull = 0x0;
		if (data->get_hull()) hull = new cuNDArray<float>(*data->get_hull());

		for (unsigned int group = 0; group < ngroups; group++){
			//std::cout << "Penguin processing group " << group << std::endl;
			boost::shared_ptr<cuNDArray<float> > cu_paths = to_cundarray(data->get_projection_group(group));
			boost::shared_ptr<cuNDArray<floatd3> >  cu_splines = to_cundarray(data->get_spline_group(group));
			rotate_splines(cu_splines.get(), data->get_angle(group));
			//std::cout << "Setup done " << std::endl;
			cuNDArray<float> projection(dims_proj);
			clear(&projection);

			boost::shared_ptr< cuNDArray<float> > EPL;
			if (data->get_EPL()){
				EPL = to_cundarray(data->get_EPL_group(group));
			}

			{
				cuNDArray<float> projection_nrm(dims_proj);
				clear(&projection_nrm);
				cuNDArray<float> normalization(cu_paths->get_dimensions());

				if (data->get_weights())
					normalization = *data->get_weights_group(group);
				else
					fill(&normalization,1.0f);


				protonBackprojection(&projection,cu_paths.get(),cu_splines.get(),physical_dims_proj,EPL.get());
				protonBackprojection(&projection_nrm,&normalization,cu_splines.get(),physical_dims_proj,EPL.get());

				clamp(&projection_nrm,1e-6f,1e8f,1.0f,1.0f);
				projection /= projection_nrm;
				if (estimate_missing)
					interpolate_missing(&projection);
				CHECK_FOR_CUDA_ERROR();
			}
			std::vector<size_t> batch_dims = *projection.get_dimensions();
			{
				//boost::shared_ptr<cuNDArray<double> > double_proj = convert_to<float,double>(&projection);
				uint64d3 pad_dims(batch_dims[0]*oversampling, batch_dims[1], batch_dims[2]);
				boost::shared_ptr<cuNDArray<float_complext> > proj_complex;
				{
					boost::shared_ptr< cuNDArray<float> > padded_projection = pad<float,3>( pad_dims, &projection );
					//std::cout << "Spline projection done " << std::endl;
					proj_complex = real_to_complex<float_complext>(padded_projection.get());
				}
				cuNDFFT<float>::instance()->fft(proj_complex.get(),0u);
				*proj_complex *= ramp;
				cuNDFFT<float>::instance()->ifft(proj_complex.get(),0u);
				//*proj_complex /= (float)ramp.get_size(0);
				//*proj_complex /= (float)ramp.get_size(0);
				*proj_complex *= float(2*boost::math::constants::pi<float>()*std::sqrt((float)ramp.get_size(0))/(physical_dims_proj[0]*oversampling));
				//ramp *= physical_dims_proj[0];


				uint64d3 crop_offsets(batch_dims[0]*(oversampling-1)/2, 0, 0);
				crop<float,3>( crop_offsets, real(proj_complex.get()).get(), &projection);
				//convert_to<double,float>(double_proj.get(),&projection);

			}

			//CHECK_FOR_CUDA_ERROR();
			//std::cout << "Filtering done " << std::endl;
			parallel_backprojection(&projection,image.get(),data->get_angle(group),physical_dims,physical_dims_proj);
			//CHECK_FOR_CUDA_ERROR();
			//std::cout << "Backprojection done " << std::endl;

		}


		//if (hull) image *= *hull;
		//image *= float(dims_proj[0])/(float(ngroups)*2*boost::math::constants::pi<float>());
		*image *= 1.0f/float(ngroups);

		if (hull) delete hull;
		//*out *= float(dims_proj[0]);
		//*out *= 2*4*boost::math::constants::pi<float>()*boost::math::constants::pi<float>()/float(ngroups);

		//*out *= *data->get_hull();
		return from_cundarray(image);

	}

protected:

	/*
	boost::shared_ptr<cuNDArray<double_complext> > calc_filter(size_t width,float spacing){
		std::vector<size_t > filter_size;
		filter_size.push_back(width);
		boost::shared_ptr< hoCuNDArray<double_complext> >  filter(new hoCuNDArray<double_complext>(filter_size));

		double_complext* filter_ptr = filter->get_data_ptr();

		const float A2 = width*width;

		for( int i=0; i<width/2; i++ ) {
			double k = double(i);
			filter_ptr[i+width/2]=k;
			//filter_ptr[i+width/2] = k*A2/(A2-k*k)*std::exp(-A2/(A2-k*k)); // From Guo et al, Journal of X-Ray Science and Technology 2011, doi: 10.3233/XST-2011-0294
		}
		for( int i=1; i<=width/2; i++ ) {
			double k = double(i);
			filter_ptr[width/2-i] = k;
			//filter_ptr[width/2-i] = k*A2/(A2-k*k)*std::exp(-A2/(A2-k*k)); // From Guo et al, Journal of X-Ray Science and Technology 2011, doi: 10.3233/XST-2011-0294
		}
		//float sum = asum(filter.get());
		//*filter *= (width/sum/4);
		//*filter /= float(width);
		//float norm = width/asum(filter.get());
		//*filter /= norm;
		//std::cout << "Filter scaling" << norm <<std::endl;
		boost::shared_ptr< cuNDArray<double_complext> >  cufilter(new cuNDArray<double_complext>(*filter));
		return cufilter;


	}*/

	template<class T> boost::shared_ptr<cuNDArray<complext<T> > > calc_filter(size_t width,float spacing){
		std::vector<size_t > filter_size;
		filter_size.push_back(width);
		boost::shared_ptr< hoCuNDArray<complext<T> > >  filter(new hoCuNDArray<complext<T> >(filter_size));

		clear(filter.get());
		complext<T>* filter_ptr = filter->get_data_ptr();

		double pi2  = boost::math::constants::pi<double>()*boost::math::constants::pi<double>();
		for (size_t i = 1; i < width/2; i+=2){
			filter_ptr[i+width/2] = complext<T>(-1.0f/(float(i*i)*pi2));
			filter_ptr[width/2-i] = complext<T>(-1.0f/(float(i*i)*pi2));
		}


		//*filter *= 0.25f/asum(filter.get());
		filter_ptr[width/2] = 0.25;
		//filter_ptr[width/2] = 1000.0f;
		//filter_ptr[0] = 0.0f;
		std::cout << sum(filter.get()) << std::endl;

		//float norm = width/asum(filter.get());
		//*filter /= spacing*spacing;
		//*filter /= spacing;

		*filter *= T(width);

		//std::cout << "Filter scaling" << norm <<std::endl;
		boost::shared_ptr< cuNDArray<complext<T> > >  cufilter(new cuNDArray<complext<T> >(*filter));
		cuNDFFT<T>::instance()->fft(cufilter.get(),0u);
		auto hofilter = cufilter->to_host();
		auto data = hofilter->get_data_ptr();
		size_t elements = hofilter->get_number_of_elements();
		size_t ncut = width/2*0.5;
		//size_t ncut = width/2*1.0;
		for ( size_t i = 1; i < ncut; i++){
			data[i] *= 0.5*(1+std::cos(boost::math::constants::pi<T>()*T(i)/ncut));
			data[width-i] *= 0.5*(1+std::cos(boost::math::constants::pi<T>()*T(i)/ncut));
		}
		for ( size_t i = ncut; i < width/2; i++){
			data[i] = complext<T>(0);
			data[width-i] = complext<T>(0);
		}

		*cufilter = cuNDArray<complext<T> >(*hofilter);


		//float norm = asum(cufilter.get());
		//*cufilter *= norm;

		return cufilter;


	}


	template<class T> typename boost::enable_if<boost::is_same<hoCuNDArray<T>,ARRAY<T> >, boost::shared_ptr<hoCuNDArray<T> > >::type from_cundarray(boost::shared_ptr<cuNDArray<T> >  in ){
		boost::shared_ptr<hoCuNDArray<T> > res(new hoCuNDArray<T>);
		in->to_host(res.get());
		return res;
	}
	template<class T> typename boost::enable_if<boost::is_same<cuNDArray<T>,ARRAY<T> >, boost::shared_ptr<cuNDArray<T> > >::type from_cundarray(boost::shared_ptr<cuNDArray<T> >  in ){
			return in;
	}

	template<class T> boost::shared_ptr<cuNDArray<T> > to_cundarray(boost::shared_ptr<cuNDArray<T> > in){
		return in;
	}
	template<class T> boost::shared_ptr<cuNDArray<T> > to_cundarray(boost::shared_ptr<hoCuNDArray<T> > in){
		return boost::make_shared<cuNDArray<T> >(*in);

	}
};
}
