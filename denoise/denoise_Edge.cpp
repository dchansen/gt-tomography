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
#include "cuEdgeATrousOperator.h"

#include "cuNDArray_math.h"
using namespace std;
using namespace Gadgetron;
using namespace std;

namespace po = boost::program_options;
int main(int argc, char** argv){

	po::options_description desc("Allowed options");
	unsigned int iterations;
	int device;
	float noise,alpha;
	string outputFile;
	desc.add_options()
    				("help", "produce help message")
    				("input,a", po::value<string>(), "Input filename")
    				("output,f", po::value<string>(&outputFile)->default_value("denoised.real"), "Output filename")
    				("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
    				("noise",po::value<float>(&noise)->default_value(1),"noise level")
			        ("alpha",po::value<float>(&alpha)->default_value(1),"noise level")

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




	auto dims = *input.get_dimensions();
	cuEdgeATrousOperator<float> E;
    E.set_sigma(noise);
	E.set_domain_dimensions(&dims);
    int levels = 3;
	E.set_levels({levels,levels,levels});
	auto wavelets = cuNDArray<float>(E.get_codomain_dimensions());
    clear(&wavelets);
	E.mult_M(&input,&wavelets);


	auto d0 = cuNDArray<float>(dims,wavelets.get_data_ptr()); //Highest detail level

	auto d0Host = abs(&d0)->to_host();
	float sigma = median(d0Host.get())/0.6745;
	float sigma2 = sigma*sigma;

    std::cout << "Sigma 2" << sigma2 << " " << sigma <<std::endl;


	for (int i = 0; i< levels; i++){
		auto view = cuNDArray<float>(dims,wavelets.get_data_ptr()+input.get_number_of_elements()*i);
		float view_norm = nrm2(&view);
		float sigmaY2 = (view_norm*view_norm)/view.get_number_of_elements();

		float sigmaX2 = sigma2*std::pow(std::pow(2.0f,-i),2);
		float T = sigma2/std::sqrt(std::max(0.0f,sigmaY2-sigmaX2));
        std::cout << "T " << T << " " << sigmaY2 << " " << sigmaX2 << std::endl;
		shrink1(&view,alpha);



	}

    //write_nd_array(&wavelets,"wavelets.real");
	E.inverse(&wavelets,&input);



	write_nd_array(&input,outputFile);

}

