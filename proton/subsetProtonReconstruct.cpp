/*
 * protonReconstruct.cpp
 *
 *  Created on: Dec 22, 2011
 *      Author: u051747
 */
#include <iostream>
#include "parameterparser.h"
#include "cuNDArray.h"
#include "hoCuNDArray.h"

#include "cuNDArray.h"
#include "cuNDArray_math.h"
#include "cuNDArray_fileio.h"

#include "protonSubsetOperator.h"
#include "hoNDArray_fileio.h"
#include "check_CUDA.h"

#include "hoCuGPBBSolver.h"
#include "hoCuPartialDerivativeOperator.h"
#include "hoCuNDArray_blas.h"


#include "osSARTSolver.h"
#include "protonDROPSolver.h"
#include "osSPSSolver.h"
#include "hoOSGPBBSolver.h"
#include "hoOSCGSolver.h"
#include "hoOSCGPBBSolver.h"
#include "hoCuBILBSolver.h"
//#include "hoABIBBSolver.h"
#include "hoCuNCGSolver.h"
#include "cuNCGSolver.h"
#include "hdf5_utils.h"
#include "BILBSolver.h"
#include "cuSolverUtils.h"
#include "osLALMSolverD.h"
#include "osMOMSolverD.h"

#include "encodingOperatorContainer.h"
#include "hoCuOperator.h"
#include "hoImageOperator.h"
#include "identityOperator.h"
#include <boost/program_options.hpp>
#include "vector_td_io.h"

#include "hoCuTvOperator.h"
#include "hoCuTvPicsOperator.h"
#include "projectionSpaceOperator.h"

using namespace std;
using namespace Gadgetron;
typedef float _real;

typedef solver<hoCuNDArray<_real>, hoCuNDArray<_real> > baseSolver;
namespace po = boost::program_options;
int main( int argc, char** argv)
{

  //
  // Parse command line
  //

  _real background =  0.00106;
  std::string dataName;
	std::string outputFile;
	vector_td<int,3> dimensions;
	vector_td<float,3> physical_dims;
	vector_td<float,3> origin;
	float beta,gamma;
	float huber;
	int iterations;
	int device;
	int subsets;
	bool use_hull,use_weights, use_non_negativity;
	float tv_weight;
	po::options_description desc("Allowed options");
	desc.add_options()
			("help", "produce help message")
			("data,D", po::value<std::string>(&dataName)->default_value("data.hdf5"), "HDF5 file containing projections and splines")
			("dimensions,d", po::value<vector_td<int,3> >(&dimensions)->default_value(vector_td<int,3>(512,512,1)), "Pixel dimensions of the image")
			("size,S", po::value<vector_td<float,3> >(&physical_dims)->default_value(vector_td<float,3>(20,20,5)), "Dimensions of the image")
			("center,c", po::value<vector_td<float,3> >(&origin)->default_value(vector_td<float,3>(0,0,0)), "Center of the reconstruction")
			("iterations,i", po::value<int>(&iterations)->default_value(10), "Dimensions of the image")
			("output,f", po::value<std::string>(&outputFile)->default_value("image.hdf5"), "Output filename")
			("prior,P", po::value<std::string>(),"Prior image filename")
			("prior-weight,k",po::value<float>(),"Weight of the prior image")
			("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
			("TV,T",po::value<float>(&tv_weight)->default_value(0),"TV Weight ")
			("subsets,n", po::value<int>(&subsets)->default_value(10), "Number of subsets to use")
			("beta",po::value<float>(&beta)->default_value(1),"Step size for SART")
			("gamma",po::value<float>(&gamma)->default_value(1e-1),"Relaxation Gamma")
			("huber",po::value<float>(&huber)->default_value(0),"Huber value")
			("use_hull",po::value<bool>(&use_hull)->default_value(true),"Estimate convex hull of object")
			("use_weights",po::value<bool>(&use_weights)->default_value(false),"Use weights if available. ")
			("use_non_negativity",po::value<bool>(&use_non_negativity)->default_value(true),"Use non-negativity constraint. ")

	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		cout << desc << "\n";
		return 1;
	}
	std::cout << "Command line options:" << std::endl;
	for (po::variables_map::iterator it = vm.begin(); it != vm.end(); ++it){
		boost::any a = it->second.value();
		std::cout << it->first << ": ";
		if (a.type() == typeid(std::string)) std::cout << it->second.as<std::string>();
		else if (a.type() == typeid(int)) std::cout << it->second.as<int>();
		else if (a.type() == typeid(float)) std::cout << it->second.as<float>();
		else if (a.type() == typeid(bool)) std::cout << it->second.as<bool>();
		else if (a.type() == typeid(vector_td<float,3>)) std::cout << it->second.as<vector_td<float,3> >();
		else if (a.type() == typeid(vector_td<int,3>)) std::cout << it->second.as<vector_td<int,3> >();
		else std::cout << "Unknown type" << std::endl;
		std::cout << std::endl;
	}
	cudaSetDevice(device);
	//cudaDeviceReset();


/*
  hoCuGPBBSolver< _real> solver;

  solver.set_max_iterations( iterations);

  solver.set_output_mode( hoCuGPBBSolver< _real>::OUTPUT_VERBOSE );*/
	//osSARTSolver<cuNDArray<_real> > solver;
	//osSPSSolver<cuNDArray<_real> > solver;
	//protonDROPSolver<cuNDArray > solver;
	//BILBSolver<cuNDArray<float> > solver;
	//osSPSSolver<cuNDArray<_real> > solver;
	//cuNCGSolver<_real> solver;
	//hoOSCGPBBSolver<hoCuNDArray<_real> > solver;
	//hoABIBBSolver<hoCuNDArray<_real> > solver;
	//hoOSCGSolver<hoCuNDArray<_real> > solver;
	//hoCuBILBSolver<hoCuNDArray<_real> > solver;
	osMOMSolverD<cuNDArray<float>> solver;
	//osLALMSolverD<cuNDArray<float>> solver;

	//solver.set_m(24);
	//solver.set_beta(1.9f);
	//solver.set_beta(beta);
	//solver.set_gamma(gamma);

  solver.set_non_negativity_constraint(use_non_negativity);
  solver.set_max_iterations(iterations);
  solver.set_huber(huber);
  solver.set_reg_steps(5);
  solver.set_dump(false);
  //solver.set_tau(gamma);
  //solver.set_regularization_iterations(1);
  //solver.set_damping(beta);

  //solver.set_tc_tolerance(1e-10f);
  std::vector<size_t> rhs_dims(&dimensions[0],&dimensions[3]); //Quick and dirty vector_td to vector

  boost::shared_ptr<protonDataset<cuNDArray> >  data(new protonDataset<cuNDArray>(dataName,use_weights));

  data = protonDataset<cuNDArray>::shuffle_dataset(data,subsets);

  data->preprocess(rhs_dims,physical_dims,use_hull);

  std::cout << "Size: " << data->get_projections()->get_number_of_elements() << std::endl;

  boost::shared_ptr< protonSubsetOperator<cuNDArray> > E (new protonSubsetOperator<cuNDArray>(data->get_subsets(), physical_dims) );

  E->set_domain_dimensions(&rhs_dims);
  E->set_codomain_dimensions(data->get_projections()->get_dimensions().get());

  if (tv_weight > 0){

  	auto Dx = boost::make_shared<cuPartialDerivativeOperator<float,2>>(0);
  	Dx->set_weight(tv_weight);
  	Dx->set_domain_dimensions(&rhs_dims);
  	Dx->set_codomain_dimensions(&rhs_dims);

  	auto Dy = boost::make_shared<cuPartialDerivativeOperator<float,2>>(1);
  	Dy->set_weight(tv_weight);
  	Dy->set_domain_dimensions(&rhs_dims);
  	Dy->set_codomain_dimensions(&rhs_dims);

  	//solver.add_regularization_operator(Dx);
  	//solver.add_regularization_operator(Dy);
  	solver.add_regularization_group({Dx,Dy});
  }


/*
  auto precon = boost::make_shared<cuNDArray<float>>(rhs_dims);
  fill(precon.get(),1.0f);

  solver.set_preconditioning_image(precon);
  solver.set_tau(0.001);
  solver.set_damping(1e-4);
  */
/*
  boost::shared_ptr<hoCuNDArray<_real > > prior;
  if (vm.count("prior")){
 	  std::cout << "Prior image regularization in use" << std::endl;
		prior = boost::static_pointer_cast<hoCuNDArray<_real > >(read_nd_array<_real >(vm["prior"].as<std::string>().c_str()));

		prior->reshape(&rhs_dims);
		_real offset = _real(0.01);
		//cuNDA_add(offset,prior.get());


		if (vm.count("prior-weight")){

		//boost::shared_ptr<hoImageOperator<_real> > I (new hoImageOperator<_real > ());
		//I->compute(prior.get());
			boost::shared_ptr<identityOperator<hoCuNDArray<_real> > > Itmp ( new identityOperator<hoCuNDArray<_real> >);

			boost::shared_ptr<projectionSpaceOperator<hoCuNDArray<_real> > > I (new projectionSpaceOperator<hoCuNDArray<_real> >(Itmp));

		I->set_weight(vm["prior-weight"].as<float>());

		I->set_codomain_dimensions(&rhs_dims);
		I->set_domain_dimensions(&rhs_dims);
		I->set_projections(prior);
		//hoCuNDArray<_real> tmp = *prior;


		//I->mult_M(prior.get(),&tmp);
		solver.add_regularization_operator(I);

		}
		solver.set_x0(prior);
  }
  */
/*
  if (vm.count("TV")){
	  std::cout << "Total variation regularization in use" << std::endl;
	  boost::shared_ptr<cuTvOperator<float,3> > tv(new cuTvOperator<float,3>);
	  //tv->set_weight(vm["TV"].as<float>());
	  //solver.add_nonlinear_operator(tv);
	  solver.set_reg_op(tv);
  }
*/

  solver.set_encoding_operator(E);
  solver.set_output_mode(baseSolver::OUTPUT_VERBOSE);
	//hoCuNDA_clear(projections.get());
	//CHECK_FOR_CUDA_ERROR();

/*
  {
  	cuNDArray<float> precon(rhs_dims);
  	fill(&precon,1.0f);
  	linearOperator<cuNDArray<float>>* E2 = E.get();
  	E2->mult_MH_M(&precon,&precon);
  	clamp_min(&precon,1e-6);
  	reciprocal_inplace(&precon);


  	auto old = boost::make_shared<cuNDArray<float>>(rhs_dims);
  	fill(old.get(),1.0f);

  	auto eigen_vec = boost::make_shared<cuNDArray<float>>(rhs_dims);
  	for (int i = 0; i < 20; i ++){
  		E2->mult_MH_M(old.get(),eigen_vec.get());
  		//*eigen_vec *= precon;

  		std::cout << "Val: " << dot(old.get(),eigen_vec.get()) << std::endl;
  		*eigen_vec /= nrm2(eigen_vec.get());
  		std::swap(eigen_vec,old);

  	}

  }
*/
  auto precon = boost::make_shared<cuNDArray<float>>(rhs_dims);
  fill(precon.get(),1.0f);
  //solver.set_preconditioning_image(precon);

	//float res = dot(projections.get(),projections.get());

  if (data->get_weights()) *data->get_projections() *= *data->get_weights();

  boost::shared_ptr<cuNDArray<float> > result;
  {
  	  GPUTimer timer("Reconstruction time");
	result = solver.solve(data->get_projections().get());
  }


	boost::shared_ptr< hoNDArray<float> > host_result = result->to_host();
	//write_nd_array<_real>(result.get(), (char*)parms.get_parameter('f')->get_string_value());
	std::stringstream ss;
	for (int i = 0; i < argc; i++){
		ss << argv[i] << " ";
	}
	saveNDArray2HDF5<3>(host_result.get(),outputFile,physical_dims,origin,ss.str(), solver.get_max_iterations());
}



