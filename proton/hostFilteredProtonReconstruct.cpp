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


#include "splineBackprojectionOperator.h"
#include "hoNDArray_fileio.h"
#include "check_CUDA.h"

#include "hoCuGPBBSolver.h"
#include "hoCuPartialDerivativeOperator.h"
#include "hoCuNDArray_blas.h"
#include "hoCuNDArray_operators.h"

#include "osSARTSolver.h"
#include "hoOSGPBBSolver.h"
#include "hoOSCGSolver.h"
#include "hoOSCGPBBSolver.h"
//#include "hoABIBBSolver.h"
#include "hdf5_utils.h"

#include "encodingOperatorContainer.h"
#include "hoCuOperator.h"
#include "hoImageOperator.h"
#include "identityOperator.h"
#include <boost/program_options.hpp>
#include "vector_td_io.h"

#include "hoCuTvOperator.h"
#include "hoCuTvPicsOperator.h"
#include "projectionSpaceOperator.h"
#include "hoCuNCGSolver.h"
#include "hoCuFilteredProton.h"

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
	int iterations;
	int device;
	int subsets;
	bool use_hull,use_weights;
	po::options_description desc("Allowed options");
	desc.add_options()
					("help", "produce help message")
					("data,D", po::value<std::string>(&dataName)->default_value("data.hdf5"), "HDF5 file containing projections and splines")
					("dimensions,d", po::value<vector_td<int,3> >(&dimensions)->default_value(vector_td<int,3>(512,512,1)), "Pixel dimensions of the image")
					("size,S", po::value<vector_td<float,3> >(&physical_dims)->default_value(vector_td<float,3>(20,20,5)), "Dimensions of the image")
					("center,c", po::value<vector_td<float,3> >(&origin)->default_value(vector_td<float,3>(0,0,0)), "Center of the reconstruction")
					("output,f", po::value<std::string>(&outputFile)->default_value("image.hdf5"), "Output filename")
					("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
					("use_hull",po::value<bool>(&use_hull)->default_value(true),"Use hull estimate")
					("use_weights",po::value<bool>(&use_weights)->default_value(false),"Use weights if available. Always true if variance is set")
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
		else if (a.type() == typeid(bool)) std::cout << it->second.as<bool>();
		else if (a.type() == typeid(float)) std::cout << it->second.as<float>();
		else if (a.type() == typeid(vector_td<float,3>)) std::cout << it->second.as<vector_td<float,3> >();
		else if (a.type() == typeid(vector_td<int,3>)) std::cout << it->second.as<vector_td<int,3> >();
		else std::cout << "Unknown type" << std::endl;
		std::cout << std::endl;
	}
	cudaSetDevice(device);
	//cudaDeviceReset();

	std::vector<size_t> rhs_dims(&dimensions[0],&dimensions[3]); //Quick and dirty vector_td to vector

	boost::shared_ptr< protonDataset<hoCuNDArray> > data(new protonDataset<hoCuNDArray>(dataName,use_weights) );
	if (use_hull) //If we don't estimate the hull, we should use a larger volume
		data->preprocess(rhs_dims,physical_dims,use_hull,background);
	else {
		floatd3 physical_dims2 = physical_dims*sqrt(2.0f);
		data->preprocess(rhs_dims,physical_dims2,use_hull,background);
	}
	hoCuFilteredProton E;


	if (data->get_weights())
		*data->get_projections() *= *data->get_weights(); //Have to scale projection data by weights before handing it to the solver. Write up the cost function and see why.

	boost::shared_ptr< hoCuNDArray<_real> > result;
	{
		GPUTimer tim("Reconstruction time:");
		result = E.calculate(rhs_dims,physical_dims,data);
	}


	splineBackprojectionOperator<hoCuNDArray> op(data,physical_dims);

	hoCuNDArray<float> tmp(*data->get_projections());
	op.mult_M(result.get(),&tmp);
	tmp -= *data->get_projections();

	std::cout << "Residual: " << dot(&tmp,&tmp) << std::endl;

	//Calculate correct scaling factor because someone cannot be bother to calculate it by hand...
	//float s = dot(data->get_projections().get(),&tmp)/dot(&tmp,&tmp);
	//*result *= s;

	std::cout << "Calculation done, saving " << std::endl;
	//write_nd_array<_real>(result.get(), (char*)parms.get_parameter('f')->get_string_value());
	std::stringstream ss;
	for (int i = 0; i < argc; i++){
		ss << argv[i] << " ";
	}
	saveNDArray2HDF5<3>(result.get(),outputFile,physical_dims,origin,ss.str(), -1);

	std::cout << "Mean: " << sum(result.get())/result->get_number_of_elements() << std::endl;
}



