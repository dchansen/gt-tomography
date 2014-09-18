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

#include "hoNDArray_fileio.h"

#include <boost/math/constants/constants.hpp>

#include "proton_utils.h"

#define MAX_THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 65535
#define MAX_BLOCKS 4096*4

namespace Gadgetron{
class hoCuFilteredProton {

public:
	hoCuFilteredProton() {

	}


	//Terrible terrible name. Penguin_sauce would be as good...or better, 'cos penguin
	boost::shared_ptr<hoCuNDArray<float> >  calculate(std::vector<size_t> dims,vector_td<float,3> physical_dims, boost::shared_ptr<protonDataset<hoCuNDArray> > data){

		cuNDArray<float> image(dims);
		clear(&image);

		std::vector<size_t> dims_proj;

		vector_td<float,3> physical_dims_proj = physical_dims;
		for (int i  = 0; i < dims.size(); i++)
		{
			if (dims[i] == 1) dims_proj.push_back(1);
			else{
				dims_proj.push_back(dims[i]*2);
				//dims_proj.push_back(dims[i]);
				physical_dims_proj[i] *= std::sqrt(2.0f);
				//physical_dims_proj[i] *= 0.5;
			}
		}

		unsigned int oversampling = 2;

		cuNDArray<double_complext> ramp = *calc_filter(dims_proj[0]*oversampling,physical_dims[0]/(dims[0]));
		//ramp *= 2.0f*oversampling;
		unsigned int ngroups = data->get_number_of_groups();

		cuNDArray<float>* hull = 0x0;
		if (data->get_hull()) hull = new cuNDArray<float>(*data->get_hull());
		for (unsigned int group = 0; group < ngroups; group++){
			//std::cout << "Penguin processing group " << group << std::endl;
			boost::shared_ptr<cuNDArray<float> > cu_paths(new cuNDArray<float>(*data->get_projection_group(group)));
			cuNDArray<floatd3>  cu_splines(*data->get_spline_group(group));
			rotate_splines(&cu_splines, data->get_angle(group));
			//std::cout << "Setup done " << std::endl;
			cuNDArray<float> projection(dims_proj);
			clear(&projection);

			boost::shared_ptr< cuNDArray<float> > EPL;
			if (data->get_EPL()){
				EPL = boost::shared_ptr< cuNDArray<float> >(new cuNDArray<float>(*data->get_EPL_group(group)));
			}

			{
				cuNDArray<float> projection_nrm(dims_proj);
				clear(&projection_nrm);
				cuNDArray<float> normalization(cu_paths->get_dimensions());

				if (data->get_weights())
					normalization = *data->get_weights_group(group);
				else
					fill(&normalization,1.0f);


				protonBackprojection(&projection,cu_paths.get(),&cu_splines,physical_dims_proj,EPL.get());
				protonBackprojection(&projection_nrm,&normalization,&cu_splines,physical_dims_proj,EPL.get());
				clamp(&projection_nrm,1e-6f,1e8f,1.0f,1.0f);
				projection /= projection_nrm;
				CHECK_FOR_CUDA_ERROR();
			}
			std::vector<size_t> batch_dims = *projection.get_dimensions();
			{
				boost::shared_ptr<cuNDArray<double> > double_proj = convert_to<float,double>(&projection);
				uint64d3 pad_dims(batch_dims[0]*oversampling, batch_dims[1], batch_dims[2]);
				boost::shared_ptr<cuNDArray<double_complext> > proj_complex;
				{
					boost::shared_ptr< cuNDArray<double> > padded_projection = pad<double,3>( pad_dims, double_proj.get() );
					//std::cout << "Spline projection done " << std::endl;
					proj_complex = real_to_complex<double_complext>(padded_projection.get());
				}
				cuNDFFT<double>::instance()->fft(proj_complex.get(),0u);
				*proj_complex *= ramp;
				cuNDFFT<double>::instance()->ifft(proj_complex.get(),0u);
				//*proj_complex /= (float)ramp.get_size(0);
				//*proj_complex /= (float)ramp.get_size(0);
				*proj_complex *= 2*boost::math::constants::pi<double>()/(physical_dims_proj[0]*oversampling);
				//ramp *= physical_dims_proj[0];


				uint64d3 crop_offsets(batch_dims[0]*(oversampling-1)/2, 0, 0);
				crop<double,3>( crop_offsets, real(proj_complex.get()).get(), double_proj.get());
				convert_to<double,float>(double_proj.get(),&projection);
			}
			//projection = *real(proj_complex.get());
			//write_nd_array(&projection,"projection.real");
			//CHECK_FOR_CUDA_ERROR();
			//std::cout << "Filtering done " << std::endl;
			parallel_backprojection(&projection,&image,data->get_angle(group),physical_dims,physical_dims_proj);
			//CHECK_FOR_CUDA_ERROR();
			//std::cout << "Backprojection done " << std::endl;

		}
		//if (hull) image *= *hull;
		//image *= float(dims_proj[0])/(float(ngroups)*2*boost::math::constants::pi<float>());
		image *= 1.0f/float(ngroups);

		if (hull) delete hull;
		//*out *= float(dims_proj[0]);
		//*out *= 2*4*boost::math::constants::pi<float>()*boost::math::constants::pi<float>()/float(ngroups);
		boost::shared_ptr<hoCuNDArray<float> > out(new hoCuNDArray<float>);
		image.to_host(out.get());
		//*out *= *data->get_hull();
		return out;

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

	boost::shared_ptr<cuNDArray<double_complext> > calc_filter(size_t width,float spacing){
		std::vector<size_t > filter_size;
		filter_size.push_back(width);
		boost::shared_ptr< hoCuNDArray<double_complext> >  filter(new hoCuNDArray<double_complext>(filter_size));

		clear(filter.get());
		double_complext* filter_ptr = filter->get_data_ptr();

		double pi2  = boost::math::constants::pi<double>()*boost::math::constants::pi<double>();
		size_t i,j;
		for ( i = 1, j = width-1; i < width/2; i+=2, j-=2){
				filter_ptr[i+width/2] = double_complext(-1.0f/(float(i*i)*pi2));
				filter_ptr[width/2-i] = double_complext(-1.0f/(float(i*i)*pi2));
		}


		//*filter *= 0.25f/asum(filter.get());
		filter_ptr[width/2] = 0.25;
		//filter_ptr[width/2] = 1000.0f;
		//filter_ptr[0] = 0.0f;
		std::cout << sum(filter.get()) << std::endl;

		//float norm = width/asum(filter.get());
	  //*filter /= spacing*spacing;
		//*filter /= spacing;

	 *filter *= double(width);

		//std::cout << "Filter scaling" << norm <<std::endl;
		boost::shared_ptr< cuNDArray<double_complext> >  cufilter(new cuNDArray<double_complext>(*filter));
		cuNDFFT<double>::instance()->fft(cufilter.get(),0u);

		//float norm = asum(cufilter.get());
		//*cufilter *= norm;

		return cufilter;


	}

};
}
