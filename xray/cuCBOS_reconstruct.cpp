#include "hoCuNDArray_utils.h"
#include "radial_utilities.h"
#include "cuNDArray_fileio.h"
#include "cuNDArray_math.h"
#include "imageOperator.h"
#include "identityOperator.h"
#include "hoPartialDerivativeOperator.h"
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


boost::shared_ptr<cuNDArray<float>> calculate_weightImage(boost::shared_ptr<CBCT_binning>  binning,boost::shared_ptr<CBCT_acquisition> ps, hoCuNDArray<float>& ho_projections, std::vector<size_t> is_dims, floatd3 imageDimensions){

	cuNDArray<float> projections(ho_projections);
	boost::shared_ptr<CBCT_binning> binning_pics=binning->get_3d_binning();
	std::vector<size_t> is_dims3d = is_dims;
	is_dims3d.pop_back();
	auto Ep = boost::make_shared<cuConebeamProjectionOperator>();
	auto ps2 = boost::make_shared<CBCT_acquisition>(boost::shared_ptr<hoCuNDArray<float>>(),ps->get_geometry());
	Ep->setup(ps2,binning_pics,imageDimensions);
	Ep->set_codomain_dimensions(ps->get_projections()->get_dimensions().get());
	Ep->set_domain_dimensions(&is_dims3d);
	//Ep->set_use_filtered_backprojection(true);
	Ep->offset_correct(&projections);
	//Ep->mult_MH(&projections,prior3d.get());
	//cgSolver<hoCuNDArray<float>> solv;
	cuNCGSolver<float> solv;
	solv.set_non_negativity_constraint(true);
	solv.set_encoding_operator(Ep);
	solv.set_max_iterations(10);
	auto prior3d = solv.solve(&projections);
	//auto prior3d = boost::make_shared<cuNDArray<float>>(is_dims3d);
	write_nd_array(prior3d.get(),"fdk.real");
	cuNDArray<float> tmp_proj(projections);
	clear(&tmp_proj);
	Ep->mult_M(prior3d.get(),&tmp_proj);
	//float s = dot(ps->get_projections().get(),&tmp_proj)/dot(&tmp_proj,&tmp_proj);
	//std::cout << "Scaling " << s << std::endl;
/*
	 //Ep->offset_correct(&tmp_proj);
	//tmp_proj *= s;
	write_nd_array(&tmp_proj,"projtmp.real");
	tmp_proj -= projections;
	tmp_proj *= float(-1);
	abs_inplace(&tmp_proj);

	write_nd_array(ps->get_projections().get(),"proj.real");
	write_nd_array(&tmp_proj,"projdiff.real");
	std::cout << "Proj size ";
	auto pdims = *tmp_proj.get_dimensions();
	for (auto p : pdims ) std::cout << p << " ";
	std::cout << std::endl;
	//Ep->set_use_filtered_backprojection(false);
	//Ep->mult_MH(&tmp_proj,prior3d.get());
	//solv.set_non_negativity_constraint(false);
	prior3d = solv.solve(&tmp_proj);
	//abs_inplace(prior.get());
	std::cout << "Prior complete" << std::endl;
	*/
	return prior3d;
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
	unsigned int subsets;
	float rho,tau;
	float tv_weight,pics_weight, wavelet_weight,huber,sigma,dct_weight;
	float tv_4d;
    bool use_non_negativity;
	int reg_iter;

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
    				("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
    				("downsample,D",po::value<unsigned int>(&downsamples)->default_value(0),"Downsample projections this factor")
    				("subsets,u",po::value<unsigned int>(&subsets)->default_value(10),"Number of subsets to use")
    				("TV",po::value<float>(&tv_weight)->default_value(0),"Total variation weight in spatial dimensions")
							("TV4D",po::value<float>(&tv_4d)->default_value(0),"Total variation weight in temporal dimensions")
    				("PICS",po::value<float>(&pics_weight)->default_value(0),"PICS weight")
    				("Wavelet,W",po::value<float>(&wavelet_weight)->default_value(0),"Weight of the wavelet operator")
    				("Huber",po::value<float>(&huber)->default_value(0),"Huber weight")
    				("use_prior","Use an FDK prior")
    				("use_non_negativity",po::value<bool>(&use_non_negativity)->default_value(true),"Prevent image from having negative attenuation")
    				("sigma",po::value<float>(&sigma)->default_value(0.1),"Sigma for billateral filter")
    				("DCT",po::value<float>(&dct_weight)->default_value(0),"DCT regularization")
    				("3D","Only use binning for selecting valid projections")
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

	boost::shared_ptr<CBCT_acquisition> ps(new CBCT_acquisition());
	ps->load(acquisition_filename);
	ps->get_geometry()->print(std::cout);
	ps->downsample(downsamples);

	float SDD = ps->get_geometry()->get_SDD();
	float SAD = ps->get_geometry()->get_SAD();

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

	float lengthOfRay_in_mm = norm(imageDimensions);
	unsigned int numSamplesPerPixel = 3;
	float minSpacing = min(voxelSize)/numSamplesPerPixel;

	unsigned int numSamplesPerRay;
	if (vm.count("samples")) numSamplesPerRay = vm["samples"].as<unsigned int>();
	else numSamplesPerRay = ceil( lengthOfRay_in_mm / minSpacing );

	float step_size_in_mm = lengthOfRay_in_mm / numSamplesPerRay;
	size_t numProjs = ps->get_projections()->get_size(2);
	size_t needed_bytes = 2 * prod(imageSize) * sizeof(float);
	std::vector<size_t> is_dims = to_std_vector((uint64d3)imageSize);

	std::cout << "IS dimensions " << is_dims[0] << " " << is_dims[1] << " " << is_dims[2] << std::endl;
	std::cout << "Image size " << imageDimensions << std::endl;

	is_dims.push_back(binning->get_number_of_bins());

	//osLALMSolver<cuNDArray<float>> solver;
	osMOMSolverD<cuNDArray<float>> solver;
	//osMOMSolverL1<cuNDArray<float>> solver;
	//osAHZCSolver<cuNDArray<float>> solver;
	//osMOMSolverF<cuNDArray<float>> solver;
	//ADMMSolver<cuNDArray<float>> solver;
	solver.set_dump(false);


	boost::shared_ptr<cuNDArray<float>> prior;
	if (vm.count("use_prior") || pics_weight > 0) {
		auto projections = *ps->get_projections();
		prior = calculate_prior(binning,ps,projections,is_dims,imageDimensions);
		//prior = calculate_weightImage(binning,ps,projections,is_dims,imageDimensions);
		solver.set_x0(prior);
	}

/*
	if (pics_weight > 0 ){
			auto Dx = boost::make_shared<cuPartialDerivativeOperator<float,4>>(0);
  	Dx->set_weight(pics_weight);
  	Dx->set_domain_dimensions(&is_dims);
  	Dx->set_codomain_dimensions(&is_dims);

  	auto Dy = boost::make_shared<cuPartialDerivativeOperator<float,4>>(1);
  	Dy->set_weight(pics_weight);
  	Dy->set_domain_dimensions(&is_dims);
  	Dy->set_codomain_dimensions(&is_dims);


  	auto Dz = boost::make_shared<cuPartialDerivativeOperator<float,4>>(2);
  	Dz->set_weight(pics_weight);
  	Dz->set_domain_dimensions(&is_dims);
  	Dz->set_codomain_dimensions(&is_dims);
		solver.add_regularization_group({Dx,Dy,Dz},prior);


	}
 */
	//osPDSolver<cuNDArray<float>> solver;
	/*
  {
  hoCuConebeamProjectionOperator op;
  	auto bin3D = boost::make_shared<CBCT_binning>(binning->get_3d_binning());
  	op.setup(ps,bin3D,imageDimensions);
  	hoCuNDArray<float> proj(*ps->get_projections());
  	op.offset_correct(&proj);
  	op.set_use_filtered_backprojection(true);

  	std::vector<size_t> is_dims3D = to_std_vector((uint64d3)imageSize);

  	hoCuNDArray<float> image(is_dims3D);

  	op.mult_MH(&proj,&image,false);

  	auto cuimage = boost::make_shared<cuNDArray<float>>(image);
  	solver.set_x0(expand(cuimage.get(),is_dims.back()));
  }
	 */
	//solver.set_regularization_iterations(1);
	//osSPSSolver<cuNDArray<float>> solver;
	/*
  if (pics_weight > 0){
  	std::cout << "Calculating PICS prior" << std::endl;
  	hoCuConebeamProjectionOperator op;
  	auto bin3D = boost::make_shared<CBCT_binning>(binning->get_3d_binning());
  	op.setup(ps,bin3D,imageDimensions);
  	hoCuNDArray<float> proj(*ps->get_projections());
  	op.offset_correct(&proj);
  	op.set_use_filtered_backprojection(true);

  	std::vector<size_t> is_dims3D = to_std_vector((uint64d3)imageSize);

  	hoCuNDArray<float> image(is_dims3D);

  	op.mult_MH(&proj,&image,false);
  	auto prior = boost::make_shared<cuNDArray<float>>(image);
  	auto PICS = boost::make_shared<cuTvPicsOperator<float,3>>();
  	PICS->set_prior(prior);
  	PICS->set_weight(pics_weight);
  	solver.add_nonlinear_operator(PICS);

  	write_nd_array(prior.get(),"fdk_prior.real");
  }




	 */

	// Define encoding matrix


	/*auto weight_array = boost::make_shared<cuNDArray<float>>(is_dims);
  {
  	boost::shared_ptr<linearOperator<cuNDArray<float>>> E2(E);
  	cuNDArray<float> tmp_proj(*ps->get_projections()->get_dimensions());
  	fill(&tmp_proj,1.0f);
  	E2->mult_MH(&tmp_proj,weight_array.get(),false);
  	clamp_min(weight_array.get(),1.0f);
  	reciprocal_inplace(weight_array.get());
  	//*weight_array *= *weight_array;
  }
  write_nd_array(weight_array.get(),"weights.real");
	 */


	//hoCuCgDescentSolver<float> solver;

	//osSPSSolver<hoNDArray<float>> solver;
	//hoCuNCGSolver<float> solver;
	//solver.set_domain_dimensions(&is_dims);
	solver.set_max_iterations(iterations);
	solver.set_output_mode(osSPSSolver<cuNDArray<float>>::OUTPUT_VERBOSE);
	solver.set_tau(tau);
	solver.set_non_negativity_constraint(use_non_negativity);
	solver.set_huber(huber);
	solver.set_reg_steps(reg_iter);
	//solver.set_rho(rho);

  if (tv_weight > 0){

  	auto Dx = boost::make_shared<cuPartialDerivativeOperator<float,4>>(0);
  	Dx->set_weight(tv_weight);
  	Dx->set_domain_dimensions(&is_dims);
  	Dx->set_codomain_dimensions(&is_dims);

  	auto Dy = boost::make_shared<cuPartialDerivativeOperator<float,4>>(1);
  	Dy->set_weight(tv_weight);
  	Dy->set_domain_dimensions(&is_dims);
  	Dy->set_codomain_dimensions(&is_dims);


  	auto Dz = boost::make_shared<cuPartialDerivativeOperator<float,4>>(2);
  	Dz->set_weight(tv_weight);
  	Dz->set_domain_dimensions(&is_dims);
  	Dz->set_codomain_dimensions(&is_dims);

  	solver.add_regularization_group({Dx,Dy,Dz});
    if (tv_4d > 0) {
        auto Dt = boost::make_shared<cuPartialDerivativeOperator<float, 4>>(3);
        Dt->set_weight(tv_4d);
        Dt->set_domain_dimensions(&is_dims);
        Dt->set_codomain_dimensions(&is_dims);
        solver.add_regularization_operator(Dt);
    }
/*
	auto projections = *ps->get_projections();
  	auto prior_weight = calculate_weightImage(binning,ps,projections,is_dims,imageDimensions);
  	//sqrt_inplace(prior_weight.get());
  	std::cout << "Prior min " << min(prior_weight.get()) << std::endl;
  	//*prior_weight -= min(prior_weight.get());
  	*prior_weight /= asum(prior_weight.get())/prior_weight->get_number_of_elements();
  	*prior_weight -= max(prior_weight.get());
  	*prior_weight *= float(-1);
  	//clamp_min(prior_weight.get(),float(1e-2));
  	//reciprocal_inplace(prior_weight.get());


  	write_nd_array(prior_weight.get(),"prior.real");
  	//cudaDeviceReset();
  	auto Wt = boost::make_shared<weightingOperator<cuNDArray<float>>>(prior_weight,Dt);
	Wt->set_weight(tv_weight);
  	Wt->set_domain_dimensions(&is_dims);
  	Wt->set_codomain_dimensions(&is_dims);
*/
  	//lver.add_regularization_operator(Dt);




  }
  /*
  if (pics_weight > 0){

  	auto Dx = boost::make_shared<cuPartialDerivativeOperator<float,4>>(0);
  	Dx->set_weight(pics_weight);
  	Dx->set_domain_dimensions(&is_dims);
  	Dx->set_codomain_dimensions(&is_dims);

  	auto Dy = boost::make_shared<cuPartialDerivativeOperator<float,4>>(1);
  	Dy->set_weight(pics_weight);
  	Dy->set_domain_dimensions(&is_dims);
  	Dy->set_codomain_dimensions(&is_dims);


  	auto Dz = boost::make_shared<cuPartialDerivativeOperator<float,4>>(2);
  	Dz->set_weight(pics_weight);
  	Dz->set_domain_dimensions(&is_dims);
  	Dz->set_codomain_dimensions(&is_dims);

  	solver.add_regularization_group({Dx,Dy,Dz},prior);


  }*/

/*
	if (tv_weight > 0){

		auto Dx = boost::make_shared<cuDCTDerivativeOperator<float>>(0);
		Dx->set_weight(tv_weight);
		Dx->set_domain_dimensions(&is_dims);
		Dx->set_codomain_dimensions(&is_dims);

		auto Dy = boost::make_shared<cuDCTDerivativeOperator<float>>(1);
		Dy->set_weight(tv_weight);
		Dy->set_domain_dimensions(&is_dims);
		Dy->set_codomain_dimensions(&is_dims);


		auto Dz = boost::make_shared<cuDCTDerivativeOperator<float>>(2);
		Dz->set_weight(tv_weight);
		Dz->set_domain_dimensions(&is_dims);
		Dz->set_codomain_dimensions(&is_dims);



		solver.add_regularization_group({Dx,Dy,Dz});


	}

*/

	if (dct_weight > 0){
		auto dctOp = boost::make_shared<cuDCTOperator<float>>();
		dctOp->set_domain_dimensions(&is_dims);
		dctOp->set_weight(dct_weight);
		solver.add_regularization_operator(dctOp);
	}

	auto E = boost::make_shared<CBSubsetOperator<cuNDArray> >(subsets);


	//E->setup(ps,binning,imageDimensions);
	E->setup(ps,binning,imageDimensions);
	E->set_domain_dimensions(&is_dims);
	E->set_codomain_dimensions(ps->get_projections()->get_dimensions().get());

	solver.set_encoding_operator(E);



	auto projections = boost::make_shared<cuNDArray<float>>(*ps->get_projections());
	std::cout << "Projection norm:" << nrm2(projections.get()) << std::endl;
	//E->set_use_offset_correction(false);

	//boost::shared_ptr<cuNDArray<bool>> mask;
	//mask = E->calculate_mask(projections,0.03f);
	//ps->set_projections(boost::shared_ptr<hoCuNDArray<float>>()); //Clear projections from host memory.

	E->offset_correct(projections.get());
	//E->set_mask(mask);
	std::cout << "Projection norm:" << nrm2(projections.get()) << std::endl;



	//solver.set_damping(1e-6);

	/*
    boost::shared_ptr<hoCuNDArray<float> > prior;

  if (vm.count("use_prior")) {
  	prior = calculate_prior(binning,ps,projections,is_dims,imageDimensions);
  	solver.set_x0(prior);
  }
	 */
	boost::shared_ptr<cuNDArray<float>> result;
	{
		GPUTimer tim("Solver");
		result = solver.solve(projections.get());
	}
//	global_timer.reset();
	std::cout << "Penguin" << nrm2(result.get()) << std::endl;

	std::cout << "Result sum " << asum(result.get()) << std::endl;

	//apply_mask(result.get(),mask.get());

	std::cout << "Result sum " << asum(result.get()) << std::endl;
	//saveNDArray2HDF5(result.get(),outputFile,imageDimensions,vector_td<float,3>(0),command_line_string.str(),iterations);


	if (wavelet_weight > 0){
		osMOMSolverF<cuNDArray<float>> solverF;
		solverF.set_max_iterations(10);
		solverF.set_x0(result);
		solverF.set_encoding_operator(E);
		solverF.set_non_negativity_constraint(true);

		auto wave = boost::make_shared<cuEdgeATrousOperator<float>>();

		wave->set_sigma(sigma);
		wave->set_domain_dimensions(&is_dims);
		if (binning->get_number_of_bins() == 1)
			wave->set_levels({2,2,2});
		else
			wave->set_levels({2,2,2,2});
		wave->set_weight(wavelet_weight);
		solverF.add_regularization_operator(wave);

		result = solverF.solve(projections.get());

	}

	saveNDArray2HDF5(result.get(),outputFile,imageDimensions,floatd3(0,0,0),command_line_string.str(),iterations);
//	write_nd_array(result.get(),"reconstruction.real");
	//write_dicom(result.get(),command_line_string.str(),imageDimensions);



}
