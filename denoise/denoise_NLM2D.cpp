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
#include "subsetAccumulateOperator.h"

#include <iostream>
#include <algorithm>
#include <sstream>
#include <math_constants.h>
#include <boost/make_shared.hpp>
#include <operators/cuGaussianFilterOperator.h>
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
#include "accumulateOperator.h"
#include "subselectionOperator.h"
#include "subsetConverter.h"
#include "nonlocalMeans.h"
using namespace std;
using namespace Gadgetron;
using namespace std;

namespace po = boost::program_options;
int main(int argc, char** argv){

	po::options_description desc("Allowed options");
	unsigned int iterations;
	int device;
	float noise;
	string outputFile;
	desc.add_options()
    				("help", "produce help message")
    				("input,a", po::value<string>(), "Input filename")
    				("output,f", po::value<string>(&outputFile)->default_value("denoised.real"), "Output filename")
    				("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
    				("noise",po::value<float>(&noise)->default_value(1),"noise level")

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


	auto input = cuNDArray<float>(*hoInput);

	auto output = cuNDArray<float>(input.get_dimensions());
	clear(&output);

//	auto output = input;
	nonlocal_means2D(&input,&output,noise);

//	cuGaussianFilterOperator<float,2> E;
//	std::vector<float> host_kernel= {1.0/16,1.0/4, 3.0/8, 1.0/4,1.0/16};
//	auto kernel = thrust::device_vector<float>(host_kernel);
//
//	E.set_sigma(noise);
/*
	std::vector<size_t> dims2D = {input.get_size(0),input.get_size(1)};
	float* input_ptr = input.get_data_ptr();
	float* output_ptr = output.get_data_ptr();
	for (int i =0; i < input.get_size(2); i++){


		cuNDArray<float> input_view(dims2D,input_ptr);
		hoCuNDArray<float> output_view(dims2D,output_ptr);
		cuNDArray<float> tmp(input_view);

		//E.mult_M(&input_view,&tmp);
		EdgeWavelet(&input_view,&tmp,&kernel,1,0,noise,false);
		EdgeWavelet(&tmp,&input_view,&kernel,1,1,noise,false);

		output_view = input_view;

		input_ptr += input_view.get_number_of_elements();
		output_ptr += input_view.get_number_of_elements();

	}
*/
//	E.mult_M(&input,&output);
	write_nd_array(&output,outputFile.c_str());

}

