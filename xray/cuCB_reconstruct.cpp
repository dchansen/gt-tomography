#include "hoCuNDArray_utils.h"
#include "radial_utilities.h"
#include "hoNDArray_fileio.h"
#include "cuNDArray.h"
#include "imageOperator.h"
#include "identityOperator.h"
#include "hoPartialDerivativeOperator.h"
#include "cuConebeamProjectionOperator.h"
#include "hoCuConebeamProjectionOperator.h"
#include "cuConvolutionOperator.h"
#include "hoCuNDArray_blas.h"
#include "hoCuNDArray_elemwise.h"
#include "hoCuNDArray_blas.h"
#include "cgSolver.h"
#include "CBCT_acquisition.h"
#include "complext.h"
#include "encodingOperatorContainer.h"
#include "vector_td_io.h"
#include "hoPartialDerivativeOperator.h"
#include "hoCuGPBBSolver.h"
#include "cuTvOperator.h"
#include "cuTvPicsOperator.h"
#include "cuNCGSolver.h"
#include "hoCuCgDescentSolver.h"
#include "hoNDArray_utils.h"
#include "hoCuPartialDerivativeOperator.h"
#include "cuDCTOperator.h"
#include "cuNDArray_fileio.h"
#include "cuATrousOperator.h"
#include "hdf5_utils.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <math_constants.h>
#include <boost/program_options.hpp>
#include <cuTv1dOperator.h>
#include "dicomWriter.h"
#include "cuDWTOperator.h"
#include "cuATvOperator.h"

using namespace std;
using namespace Gadgetron;

namespace po = boost::program_options;

boost::shared_ptr<cuNDArray<float> > calculate_prior(boost::shared_ptr<CBCT_binning>  binning,boost::shared_ptr<CBCT_acquisition> ps, hoCuNDArray<float>& projections, std::vector<size_t> is_dims, floatd3 imageDimensions){
	std::cout << "Calculating FDK prior" << std::endl;
	boost::shared_ptr<CBCT_binning> binning_pics=binning->get_3d_binning();
	std::vector<size_t> is_dims3d = is_dims;
	is_dims3d.pop_back();
	boost::shared_ptr< hoCuConebeamProjectionOperator >
	Ep( new hoCuConebeamProjectionOperator() );
	Ep->setup(ps,binning_pics,imageDimensions);
	Ep->set_codomain_dimensions(ps->get_projections()->get_dimensions().get());
	Ep->set_domain_dimensions(&is_dims3d);
	Ep->set_use_filtered_backprojection(true);
	boost::shared_ptr<hoCuNDArray<float> > prior3d(new hoCuNDArray<float>(&is_dims3d));
	Ep->mult_MH(&projections,prior3d.get());

	hoCuNDArray<float> tmp_proj(*ps->get_projections());
	Ep->mult_M(prior3d.get(),&tmp_proj);
	float s = dot(ps->get_projections().get(),&tmp_proj)/dot(&tmp_proj,&tmp_proj);
	*prior3d *= s;
	boost::shared_ptr<cuNDArray<float> > prior(new cuNDArray<float>(*expand( prior3d.get(), is_dims.back() )));
	std::cout << "Prior complete" << std::endl;
	return prior;
}


int main(int argc, char** argv)
{
	string acquisition_filename;
	string outputFile;
	uintd3 imageSize;
	floatd3 voxelSize;
	int device;
	unsigned int downsamples;
	unsigned int iterations;
	float rho,dct_weight;
	po::options_description desc("Allowed options");

	desc.add_options()
    		("help", "produce help message")
    		("acquisition,a", po::value<string>(&acquisition_filename)->default_value("acquisition.hdf5"), "Acquisition data")
    		("samples,n",po::value<unsigned int>(),"Number of samples per ray")
    		("output,f", po::value<string>(&outputFile)->default_value("reconstruction.hdf5"), "Output filename")
    		("size,s",po::value<uintd3>(&imageSize)->default_value(uintd3(512,512,1)),"Image size in pixels")
    		("binning,b",po::value<string>(),"Binning file for 4d reconstruction")
    		("SAG","Use exact SAG correction if present")
    		("voxelSize,v",po::value<floatd3>(&voxelSize)->default_value(floatd3(0.488f,0.488f,1.0f)),"Voxel size in mm")
    		("dimensions,d",po::value<floatd3>(),"Image dimensions in mm. Overwrites voxelSize.")
    		("iterations,i",po::value<unsigned int>(&iterations)->default_value(10),"Number of iterations")
    		("TV,T",po::value<float>(),"TV Weight ")
					("ATV",po::value<float>(),"TV Weight ")
					("TV4D",po::value<float>(),"Total variation weight in temporal dimensions")
    		("PICS",po::value<float>(),"TV Weight of the prior image (Prior image compressed sensing)")
    		("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
    		("downsample,D",po::value<unsigned int>(&downsamples)->default_value(0),"Downsample projections this factor")
    		("rho",po::value<float>(&rho)->default_value(0.5f),"Rho-value for line search. Must be between 0 and 1. Smaller value means faster runtime, but less stable algorithm.")
    		("DCT",po::value<float>(&dct_weight)->default_value(0),"DCT regularization")
    		("use_prior","Use an FDK prior")
    		("Wavelet,W",po::value<float>(),"Wavelet weight")
    		("3D","Only use binning for selecting valid projections")
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
		else command_line_string << "Unknown type" << std::endl;
		command_line_string << std::endl;
	}

	std::cout << command_line_string.str();

	cudaSetDevice(device);
	cudaDeviceReset();

	//Really weird stuff. Needed to initialize the device?? Should find real bug.
	cudaDeviceManager::Instance()->lockHandle();
	cudaDeviceManager::Instance()->unlockHandle();

	boost::shared_ptr<CBCT_acquisition> ps(new CBCT_acquisition());
	ps->load(acquisition_filename);
	ps->get_geometry()->print(std::cout);
	ps->downsample(downsamples);


	boost::shared_ptr<CBCT_binning> binning(new CBCT_binning());
	if (vm.count("binning")){
		std::cout << "Loading binning data" << std::endl;
		binning->load(vm["binning"].as<string>());
		if (vm.count("3D"))
			binning = binning->get_3d_binning();
	} else binning->set_as_default_3d_bin(ps->get_projections()->get_size(2));
	binning->print(std::cout);

	floatd3 imageDimensions;
	if (vm.count("dimensions")){
		imageDimensions = vm["dimensions"].as<floatd3>();
		voxelSize = imageDimensions/imageSize;
	}
	else imageDimensions = voxelSize*imageSize;

	std::vector<size_t> is_dims = to_std_vector((uint64d3)imageSize);

	std::cout << "IS dimensions " << is_dims[0] << " " << is_dims[1] << " " << is_dims[2] << std::endl;
	std::cout << "Image size " << imageDimensions << std::endl;

	is_dims.push_back(binning->get_number_of_bins());

	cuNCGSolver<float> solver;
	//gpBbSolver<cuNDArray<float>> solver;

	hoCuNDArray<float>* projections = ps->get_projections().get();
	std::cout << "Projection nrm" << nrm2(projections) << std::endl;
	boost::shared_ptr<cuNDArray<float>> prior;
	if (vm.count("use_prior")){
		prior = calculate_prior(binning,ps,*projections,is_dims,imageDimensions);
		solver.set_x0(prior);
	}


	// auto projections2 = ps->get_projections();
	//auto result = calculate_prior(binning,ps,*projections2,is_dims,imageDimensions);
	// Define encoding matrix
	boost::shared_ptr< cuConebeamProjectionOperator >
	E( new cuConebeamProjectionOperator() );

	E->setup(ps,binning,imageDimensions);
	E->set_domain_dimensions(&is_dims);
	E->set_codomain_dimensions(ps->get_projections()->get_dimensions().get());

	//hoCuGPBBSolver<float> solver;
	//hoCuCgDescentSolver<float> solver;

	solver.set_encoding_operator(E);
	solver.set_domain_dimensions(&is_dims);
	solver.set_max_iterations(iterations);
	solver.set_output_mode(hoCuGPBBSolver<float>::OUTPUT_VERBOSE);
	solver.set_non_negativity_constraint(true);
	solver.set_rho(rho);


	cuNDArray<float> cuproj(*projections);
	if (E->get_use_offset_correction())
		E->offset_correct(&cuproj);



	if (vm.count("TV")){
		std::cout << "Total variation regularization in use" << std::endl;
		boost::shared_ptr<cuTvOperator<float,3> > tv(new cuTvOperator<float,3>);
		tv->set_weight(vm["TV"].as<float>());
		solver.add_nonlinear_operator(tv);
		/*
    boost::shared_ptr<hoCuTvOperator<float,4> > tv2(new hoCuTvOperator<float,4>);
    tv2->set_step(2);
    tv2->set_weight(vm["TV"].as<float>());
    solver.add_nonlinear_operator(tv2);
    boost::shared_ptr<hoCuTvOperator<float,4> > tv3(new hoCuTvOperator<float,4>);
    tv3->set_step(3);
    tv3->set_weight(vm["TV"].as<float>());
    solver.add_nonlinear_operator(tv3);
		 */

	}


	if (vm.count("ATV")){
		std::cout << "Advanced Total variation regularization in use" << std::endl;
		boost::shared_ptr<cuATvOperator<float,3> > tv(new cuATvOperator<float,3>);
		tv->set_weight(vm["ATV"].as<float>());
		solver.add_nonlinear_operator(tv);
		/*
    boost::shared_ptr<hoCuTvOperator<float,4> > tv2(new hoCuTvOperator<float,4>);
    tv2->set_step(2);
    tv2->set_weight(vm["TV"].as<float>());
    solver.add_nonlinear_operator(tv2);
    boost::shared_ptr<hoCuTvOperator<float,4> > tv3(new hoCuTvOperator<float,4>);
    tv3->set_step(3);
    tv3->set_weight(vm["TV"].as<float>());
    solver.add_nonlinear_operator(tv3);
		 */

	}

	if (vm.count("TV4D")) {
		std::cout << "Total variation 4d regularization in use" << std::endl;
		boost::shared_ptr<cuTv1DOperator<float, 4> > tv4d(new cuTv1DOperator<float, 4>);
		tv4d->set_weight(vm["TV4D"].as<float>());
		solver.add_nonlinear_operator(tv4d);
	}


	if (vm.count("Wavelet")){
		//auto wave=  boost::make_shared<cuATrousOperator<float>>();
//		wave->set_domain_dimensions(&is_dims);
		//wave->set_levels({2,2,2,2});
		auto wave=  boost::make_shared<cuDWTOperator<float,3>>();
		wave->set_domain_dimensions(&is_dims);
		wave->set_codomain_dimensions(&is_dims);
		wave->set_levels(3);
		wave->set_weight(vm["Wavelet"].as<float>());
		solver.add_regularization_operator(wave,1);
	}

	if (dct_weight > 0){
		auto dctOp = boost::make_shared<cuDCTOperator<float>>();
		dctOp->set_domain_dimensions(&is_dims);
		dctOp->set_weight(dct_weight);
		solver.add_regularization_operator(dctOp);
	}


  if (vm.count("PICS")){
    std::cout << "PICS in use" << std::endl;
    if (!prior) prior = calculate_prior(binning,ps,*projections,is_dims,imageDimensions);
    boost::shared_ptr<cuTvPicsOperator<float,3> > pics (new cuTvPicsOperator<float,3>);
    pics->set_prior(prior);
    pics->set_weight(vm["PICS"].as<float>());
    solver.add_nonlinear_operator(pics);
    solver.set_x0(prior);
  }

	auto result = solver.solve(&cuproj);

	write_dicom(result.get(),command_line_string.str(),imageDimensions);
	//write_nd_array( result.get(), outputFile.c_str());
	saveNDArray2HDF5(result.get(),outputFile,imageDimensions,vector_td<float,3>(0),command_line_string.str(),iterations);

}
