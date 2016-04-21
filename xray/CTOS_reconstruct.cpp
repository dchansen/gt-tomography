#include "hoCuNDArray_utils.h"
#include "radial_utilities.h"
#include "cuNDArray_fileio.h"
#include "cuNDArray_math.h"
#include "imageOperator.h"
#include "identityOperator.h"
#include "hoPartialDerivativeOperator.h"


#include "hoCuNDArray_blas.h"
#include "hoCuNDArray_elemwise.h"
#include "hoCuNDArray_blas.h"
#include "cgSolver.h"
#include "CT_acquisition.h"
#include "complext.h"
#include "encodingOperatorContainer.h"
#include "vector_td_io.h"
#include "hoPartialDerivativeOperator.h"
#include "hoCuGPBBSolver.h"
#include "hoCuTvOperator.h"
#include "hoCuTvPicsOperator.h"
#include "hoCuNCGSolver.h"
#include "hoCuCgDescentSolver.h"
#include "hoNDArray_utils.h"
#include "hoCuPartialDerivativeOperator.h"
#include "cuTvOperator.h"
#include "cuTv1dOperator.h"
#include "cuTvPicsOperator.h"
#include "CBSubsetOperator.h"
#include "osSPSSolver.h"
#include "osMOMSolver.h"
#include "osMOMSolverD.h"
#include "osMOMSolverD2.h"
#include "hoCuNCGSolver.h"
#include "osMOMSolverD3.h"
#include "osMOMSolverL1.h"
#include "osMOMSolverF.h"
#include "osAHZCSolver.h"
#include "ADMMSolver.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <math_constants.h>
#include <boost/program_options.hpp>
#include <boost/make_shared.hpp>
#include <GPUTimer.h>
#include "cuSolverUtils.h"
#include "osPDsolver.h"
#include "osLALMSolver.h"
#include "osLALMSolver2.h"
#include "cuATrousOperator.h"
#include "hdf5_utils.h"
#include "cuEdgeATrousOperator.h"
#include "cuDCTOperator.h"
#include "cuDCTDerivativeOperator.h"
#include "dicomWriter.h"
#include "conebeam_projection.h"
#include "weightingOperator.h"
#include "hoNDArray_math.h"
#include "cuNCGSolver.h"
#include "CT_acquisition.h"
#include "CTSubsetOperator.h"
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
using namespace std;
using namespace Gadgetron;




namespace po = boost::program_options;
namespace fs = boost::filesystem;

std::vector<string> get_dcm_files(std::string dir){

	fs::directory_iterator end_itr;
	const boost::regex filter(".*\.dcm");

	std::vector<std::string> files;
	for (fs::directory_iterator i (dir); i!= end_itr; i++){
		if (!fs::is_regular(i->status())) continue;
		boost::smatch what;

		if (!boost::regex_match(i->path().filename().string(),what,filter)) continue;

		//std::cout  << i->path().filename() << std::endl;
		files.push_back(i->path().string());
	}
	sort(files.begin(),files.end());

	return files;
}

int main(int argc, char** argv)
{

	string acquisition_filename;
	string outputFile;
	uintd3 imageSize;
	floatd3 voxelSize;
	int device;

	unsigned int iterations;
	unsigned int subsets;
	float rho,tau;
	float tv_weight, wavelet_weight,huber,sigma,dct_weight;

    bool use_non_negativity;
	int reg_iter;

	po::options_description desc("Allowed options");

	desc.add_options()
    				("help", "produce help message")
    				("dir", po::value<string>(), "Dicom directory")
    				("samples,n",po::value<unsigned int>(),"Number of samples per ray")
    				("output,f", po::value<string>(&outputFile)->default_value("reconstruction.hdf5"), "Output filename")
    				("size,s",po::value<uintd3>(&imageSize)->default_value(uintd3(512,512,1)),"Image size in pixels")
    				("voxelSize,v",po::value<floatd3>(&voxelSize)->default_value(floatd3(0.488f,0.488f,1.0f)),"Voxel size in mm")
    				("dimensions,d",po::value<floatd3>(),"Image dimensions in mm. Overwrites voxelSize.")
    				("iterations,i",po::value<unsigned int>(&iterations)->default_value(10),"Number of iterations")
    				("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
    				("subsets,u",po::value<unsigned int>(&subsets)->default_value(10),"Number of subsets to use")
    				("TV",po::value<float>(&tv_weight)->default_value(0),"Total variation weight in spatial dimensions")

    				("Wavelet,W",po::value<float>(&wavelet_weight)->default_value(0),"Weight of the wavelet operator")
    				("Huber",po::value<float>(&huber)->default_value(0),"Huber weight")
    				("use_non_negativity",po::value<bool>(&use_non_negativity)->default_value(true),"Prevent image from having negative attenuation")
    				("sigma",po::value<float>(&sigma)->default_value(0.1),"Sigma for billateral filter")
    				("DCT",po::value<float>(&dct_weight)->default_value(0),"DCT regularization")
							("tau",po::value<float>(&tau)->default_value(1e-5),"Tau value for solver")
							("reg_iter",po::value<int>(&reg_iter)->default_value(2))
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
		else if (a.type() == typeid(vector_td<float,3>)) command_line_string << it->second.as<vector_td<float,3> >();
		else if (a.type() == typeid(vector_td<int,3>)) command_line_string << it->second.as<vector_td<int,3> >();
		else if (a.type() == typeid(vector_td<unsigned int,3>)) command_line_string << it->second.as<vector_td<unsigned int,3> >();
        else if (a.type() == typeid(bool)) command_line_string << it->second.as<bool>();
		else command_line_string << "Unknown type" << std::endl;
		command_line_string << std::endl;
	}
	std::cout << command_line_string.str();

	cudaSetDevice(device);
	cudaDeviceReset();

	//Really weird stuff. Needed to initialize the device?? Should find real bug.
	cudaDeviceManager::Instance()->lockHandle();
	cudaDeviceManager::Instance()->unlockHandle();

	auto files = read_dicom_projections(get_dcm_files(vm["dir"].as<string>()));
	std::vector<float>& axials =  files->geometry.detectorFocalCenterAxialPosition;
	std::cout << "Axials size " << axials.size() << std::endl;
	auto mean_offset = std::accumulate(axials.begin(),axials.end(),0.0f)/axials.size();
	std::cout << "Mean offset " << mean_offset << std::endl;
	for ( auto & z : axials)
		z -= mean_offset;



	floatd3 imageDimensions;
	if (vm.count("dimensions")){
		imageDimensions = vm["dimensions"].as<floatd3>();
		voxelSize = imageDimensions/imageSize;
	}
	else imageDimensions = voxelSize*imageSize;

	float lengthOfRay_in_mm = norm(imageDimensions);
	unsigned int numSamplesPerPixel = 3;
	float minSpacing = min(voxelSize)/numSamplesPerPixel;

	unsigned int numSamplesPerRay;
	if (vm.count("samples")) numSamplesPerRay = vm["samples"].as<unsigned int>();
	else numSamplesPerRay = ceil( lengthOfRay_in_mm / minSpacing );

	float step_size_in_mm = lengthOfRay_in_mm / numSamplesPerRay;
	size_t numProjs = files->projections.get_size(2);
	size_t needed_bytes = 2 * prod(imageSize) * sizeof(float);
	std::vector<size_t> is_dims = to_std_vector((uint64d3)imageSize);

	std::cout << "IS dimensions " << is_dims[0] << " " << is_dims[1] << " " << is_dims[2] << std::endl;
	std::cout << "Image size " << imageDimensions << std::endl;

	//osLALMSolver<hoCuNDArray<float> solver;
	//osMOMSolverD3<hoCuNDArray<float>> solver;
	osSPSSolver<hoCuNDArray<float>> solver;
	//osMOMSolverL1<hoCuNDArray<float> solver;
	//osAHZCSolver<hoCuNDArray<float> solver;
	//osMOMSolverF<hoCuNDArray<float> solver;
	//ADMMSolver<hoCuNDArray<float> solver;
	//solver.set_dump(true);
	//solver.set_stepsize(1);
	//solver.set_beta(0.1);





	//hoCuCgDescentSolver<float> solver;

	//osSPSSolver<hoNDArray<float>> solver;
	//hoCuNCGSolver<float> solver;
	//solver.set_domain_dimensions(&is_dims);
	solver.set_max_iterations(iterations);
	solver.set_output_mode(osSPSSolver<hoCuNDArray<float>>::OUTPUT_VERBOSE);
	//solver.set_tau(tau);
	solver.set_non_negativity_constraint(use_non_negativity);
	//solver.set_huber(huber);
	//solver.set_reg_steps(reg_iter);
	//solver.set_rho(rho);

  if (tv_weight > 0) {

	  auto Dx = boost::make_shared<hoCuPartialDerivativeOperator<float, 4>>(0);
	  Dx->set_weight(tv_weight);
	  Dx->set_domain_dimensions(&is_dims);
	  Dx->set_codomain_dimensions(&is_dims);

	  auto Dy = boost::make_shared<hoCuPartialDerivativeOperator<float, 4>>(1);
	  Dy->set_weight(tv_weight);
	  Dy->set_domain_dimensions(&is_dims);
	  Dy->set_codomain_dimensions(&is_dims);


	  auto Dz = boost::make_shared<hoCuPartialDerivativeOperator<float, 4>>(2);
	  Dz->set_weight(tv_weight);
	  Dz->set_domain_dimensions(&is_dims);
	  Dz->set_codomain_dimensions(&is_dims);

	  //solver.add_regularization_group({Dx, Dy, Dz});
  }


	auto E = boost::make_shared<CTSubsetOperator<hoCuNDArray> >(subsets);

	E->set_domain_dimensions(&is_dims);
	//E->setup(ps,binning,imageDimensions);
	auto host_projections = E->setup(files,imageDimensions);

	E->set_codomain_dimensions(host_projections->get_dimensions().get());

	solver.set_encoding_operator(E);




	boost::shared_ptr<hoCuNDArray<float>> result;
	{
		GPUTimer tim("Solver");
		result = solver.solve(host_projections.get());
	}

	std::cout << "Penguin" << nrm2(result.get()) << std::endl;

	std::cout << "Result sum " << asum(result.get()) << std::endl;

	//apply_mask(result.get(),mask.get());

	std::cout << "Result sum " << asum(result.get()) << std::endl;
	//saveNDArray2HDF5(result.get(),outputFile,imageDimensions,vector_td<float,3>(0),command_line_string.str(),iterations);



	saveNDArray2HDF5(result.get(),outputFile,imageDimensions,floatd3(0,0,0),command_line_string.str(),iterations);
//	write_nd_array(result.get(),"reconstruction.real");
	//write_dicom(result.get(),command_line_string.str(),imageDimensions);



}
