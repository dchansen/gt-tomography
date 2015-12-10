/*
 * split_denoise.cpp

 *
 *  Created on: Dec 10, 2015
 *      Author: dch
 */


#include "hoNDArray_fileio.h"
#include "hoCuNDArray_math.h"
#include "vector_td_utilities.h"
#include "hoNDArray_utils.h"
#include "GPUTimer.h"
#include "cuATrousOperator.h"
#include "cuEdgeATrousOperator.h"
#include "cuSbCgSolver.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <boost/program_options.hpp>
#include "cuDCTOperator.h"
#include "osSPSSolver.h"
#include "osMOMSolver.h"
#include "osMOMSolverD.h"
#include "osMOMSolverD2.h"
#include "hoCuNCGSolver.h"
#include "osMOMSolverD3.h"
#include "osMOMSolverF.h"
#include "osAHZCSolver.h"
#include "ADMMSolver.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <math_constants.h>
#include <boost/make_shared.hpp>
#include "cuSolverUtils.h"
#include "osPDsolver.h"
#include "osLALMSolver.h"
#include "osLALMSolver2.h"
#include "cuATrousOperator.h"
#include "hdf5_utils.h"
#include "cuEdgeATrousOperator.h"
#include "cuDCTOperator.h"
#include "cuDCTDerivativeOperator.h"
#include "weightingOperator.h"
#include "hoNDArray_math.h"
#include "cuNCGSolver.h"
#include "subsetAccumulateOperator.h"
#include "accumulateOperator.h"
#include "subselectionOperator.h"
#include "subsetConverter.h"
using namespace std;
using namespace Gadgetron;
using namespace std;

namespace po = boost::program_options;
int main(int argc, char** argv){

	po::options_description desc("Allowed options");
	unsigned int iterations;
	int device;
	float tv_weight,wavelet_weight,huber,dct_weight;
	string outputFile;
	desc.add_options()
    				("help", "produce help message")
    				("input,a", po::value<string>(), "Input filename")
    				("samples,n",po::value<unsigned int>(),"Number of samples per ray")
    				("output,f", po::value<string>(&outputFile)->default_value("reconstruction.real"), "Output filename")
    				("iterations,i",po::value<unsigned int>(&iterations)->default_value(10),"Number of iterations")
    				("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
    				("TV",po::value<float>(&tv_weight)->default_value(0),"Total variation weight")
    				("Wavelet,W",po::value<float>(&wavelet_weight)->default_value(0),"Weight of the wavelet operator")
    				("Huber",po::value<float>(&huber)->default_value(0),"Huber weight")
    				("DCT",po::value<float>(&dct_weight)->default_value(0),"DCT regularization")
    				;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		cout << desc << "\n";
		return 1;
	}

	std::stringstream command_line_string;
	std::cout << "Command line options:" << std::endl;
	for (po::variables_map::iterator it = vm.begin(); it != vm.end(); ++it){
		boost::any a = it->second.value();
		command_line_string << it->first << ": ";
		if (a.type() == typeid(std::string)) command_line_string << it->second.as<std::string>();
		else if (a.type() == typeid(int)) command_line_string << it->second.as<int>();
		else if (a.type() == typeid(unsigned int)) command_line_string << it->second.as<unsigned int>();
		else if (a.type() == typeid(float)) command_line_string << it->second.as<float>();
		else command_line_string << "Unknown type" << std::endl;
		command_line_string << std::endl;
	}
	std::cout << command_line_string.str();

	cudaSetDevice(device);
	cudaDeviceReset();

	string filename = vm["input"].as<string>();

	auto hoInput = read_nd_array<float>(filename.c_str());

	auto is_dims = *hoInput->get_dimensions();
	auto double_dims = is_dims;
	double_dims.push_back(2);
/*
	cuSbCgSolver<float> solver;
	solver.set_max_outer_iterations(iterations);
	solver.get_inner_solver()->set_max_iterations(10);
	solver.set_output_mode(cuSbCgSolver<float>::OUTPUT_VERBOSE);
*/
	osMOMSolverD3<cuNDArray<float>> solver;
	solver.set_max_iterations(iterations);

 if (tv_weight > 0){

  	auto Dx = boost::make_shared<cuPartialDerivativeOperator<float,4>>(0);
  	Dx->set_weight(tv_weight);

  	Dx->set_domain_dimensions(&is_dims);
  	Dx->set_codomain_dimensions(&is_dims);
/*
  	Dx->set_domain_dimensions(&double_dims);
  	Dx->set_codomain_dimensions(&double_dims);
*/
  	auto Dy = boost::make_shared<cuPartialDerivativeOperator<float,4>>(1);
  	Dy->set_weight(tv_weight);
  	Dy->set_domain_dimensions(&is_dims);
  	Dy->set_codomain_dimensions(&is_dims);

  	auto Dx1 = boost::make_shared<subselectionOperator<cuNDArray<float>>>(Dx,0);
  	Dx1->set_domain_dimensions(&double_dims);
  	Dx1->set_codomain_dimensions(&is_dims);
  	auto Dy1 = boost::make_shared<subselectionOperator<cuNDArray<float>>>(Dy,0);
  	Dy1->set_domain_dimensions(&double_dims);
  	Dy1->set_codomain_dimensions(&is_dims);

  	Dx1->set_weight(tv_weight);
  	Dy1->set_weight(tv_weight);

  	solver.add_regularization_group({Dx1,Dy1});
  	/*
  	solver.add_regularization_group_operator(Dx1);
  	solver.add_regularization_group_operator(Dy1);
  	solver.add_group();
*/
 }


	if (dct_weight > 0){
		auto dctOp = boost::make_shared<identityOperator<cuNDArray<float>>>();
		//auto dctOp = boost::make_shared<cuDCTOperator<float>>();
		dctOp->set_domain_dimensions(&is_dims);
		dctOp->set_codomain_dimensions(&is_dims);
		dctOp->set_weight(dct_weight);

		auto dctOp1 = boost::make_shared<subselectionOperator<cuNDArray<float>>>(dctOp,1);
		dctOp1->set_domain_dimensions(&double_dims);
		dctOp1->set_codomain_dimensions(dctOp->get_codomain_dimensions().get());
		dctOp1->set_weight(dct_weight);
		solver.add_regularization_operator(dctOp1);


	}

	auto E = boost::make_shared<identityOperator<cuNDArray<float> > >();


	//E->setup(ps,binning,imageDimensions);
	E->set_domain_dimensions(&is_dims);
	E->set_codomain_dimensions(&is_dims);

	auto B = boost::make_shared<accumulateOperator<cuNDArray<float>>>(E);
	B->set_domain_dimensions(&double_dims);
	B->set_codomain_dimensions(&is_dims);
	B->set_weight(1);


	auto B2 = boost::make_shared<subsetConverter<cuNDArray<float>>>(B);
	B2->set_domain_dimensions(&double_dims);
	solver.set_encoding_operator(B2);




	cuNDArray<float> cuInput (*hoInput);

	auto result = solver.solve(&cuInput);


	write_nd_array(result.get(),outputFile);

}

