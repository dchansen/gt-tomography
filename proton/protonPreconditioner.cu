#include "protonPreconditioner.h"
#include "vector_td_utilities.h"

#include "cuCgSolver.h"
#include "vector_td.h"
#include "cuNDFFT.h"
#include "radial_utilities.h"
#include "cuNFFT.h"
#include "cuNFFTOperator.h"
#include "cuNDArray_math.h"

#include "hoNDArray_fileio.h"
#include <thrust/sort.h>
#include <thrust/transform.h>

using namespace Gadgetron;

void protonPreconditioner::apply(cuNDArray<float> * in, cuNDArray<float> * out){

	boost::shared_ptr<cuNDArray<float_complext> > complex_in =real_to_complex<float_complext>(in);

	uint64d2 dims = from_std_vector<size_t,2>(*in->get_dimensions());
	complex_in = pad<float_complext,2>(dims*size_t(2),complex_in.get());
	cuNDFFT<float>::instance()->fft(complex_in.get());
	*complex_in *= *kernel_;
	cuNDFFT<float>::instance()->ifft(complex_in.get());

	*out = *crop<float,2>(dims/size_t(2),dims,real<float_complext>(complex_in.get()).get());

	if (hull_.get()) *out *= *hull_;

}

static float find_percentile(cuNDArray<float_complext>* arr,float fraction){
	boost::shared_ptr<cuNDArray<float> > absarr = abs(arr);
	thrust::sort(absarr->begin(),absarr->end());

	return absarr->at((size_t)(absarr->get_number_of_elements()*fraction));


}


struct precon_cutoff : public thrust::unary_function<float_complext,float_complext>
{

	precon_cutoff(float cutoff){
		_cutoff = cutoff;
	}
  __device__ float_complext operator()(const float_complext &x) const {
  	float ax = abs(x);
  	//if (ax < _cutoff) return x*exp(-(_cutoff-ax)*(_cutoff-ax)/(_cutoff*_cutoff));
  	if (ax < _cutoff) return float_complext(0);
  	else return x;
  }

  float _cutoff;

};


boost::shared_ptr<cuNDArray<float_complext> > protonPreconditioner::calcKernel(uint64d2 dims, int angles){

		boost::shared_ptr< cuNDArray<floatd2> > traj =
				compute_radial_trajectory_fixed_angle_2d<float>(dims[0],angles,1);


		boost::shared_ptr< cuNDArray<float> > dcw =
				compute_radial_dcw_fixed_angle_2d<float>(dims[0],angles,2.0f,1.0f);



		cuCgSolver<float_complext> solver;
		 solver.set_output_mode( cuCgSolver<float>::OUTPUT_VERBOSE );
		boost::shared_ptr<cuNFFTOperator<float,2>  > E (new cuNFFTOperator<float,2>);

		E->setup( uint64d2(dims[0], dims[1]),
				uint64d2(dims[0], dims[1])<<1, // !! <-- alpha_
				5.5f );
		E->set_dcw( dcw );

		E->preprocess( traj.get() );



		std::vector<size_t> data_dims;
		data_dims.push_back(dims[0]);
		data_dims.push_back(dims[1]);



		std::vector<size_t > kernel_size;
		kernel_size.push_back(dims[0]);
		hoNDArray<float_complext> kernel(kernel_size);
		float A2 = dims[0]*dims[0]/4;

		for (size_t i = 0; i < dims[0]/2; i++)
			kernel[i] = (dims[0]/2-float(i))/float(dims[0]/2);
		for (size_t i = 0; i < dims[0]/2; i++)
			kernel[i+dims[0]/2] = (float(i))/float(dims[0]/2);

/*
		for (size_t k = 0; k < dims[0]/2; k++)
			kernel[dims[0]/2-k-1] = k*A2/(A2-k*k)*std::exp(-A2/(A2-k*k));
		for (size_t k = 0; k < dims[0]/2; k++)
			kernel[dims[0]-k-1] = kernel[k];
*/




		cuNDArray<float_complext> cu_kernel(kernel);


		boost::shared_ptr<cuNDArray<float_complext> > ekernel = expand(&cu_kernel,angles);

		 boost::shared_ptr< hoNDArray<float> > host_kernel = abs(ekernel.get())->to_host();
			  write_nd_array<float>( host_kernel.get(), "filter.real" );

		E->set_domain_dimensions(&data_dims);
		E->set_codomain_dimensions(ekernel->get_dimensions().get());

		solver.set_encoding_operator(E);
		solver.set_max_iterations(30);
		solver.set_tc_tolerance(1e-5);
		boost::shared_ptr<cuNDArray<float_complext> > result = solver.solve(ekernel.get());



	  boost::shared_ptr< hoNDArray<float> > host_norm = abs(result.get())->to_host();
	  write_nd_array<float>( host_norm.get(), "kernel.real" );

	  result = pad<float_complext,2>(dims*size_t(2),result.get());
		cuNDFFT<float>::instance()->fft(result.get());
		float cutoff = find_percentile(result.get(),0.05);
		std::cout << "Cutoff: " << cutoff << std::endl;

		//thrust::transform(result->begin(),result->end(),result->begin(),precon_cutoff(cutoff));
		sqrt_inplace(result.get());
		return result;
	}
