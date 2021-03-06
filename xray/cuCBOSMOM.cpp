#include "hoCuNDArray_utils.h"
#include "radial_utilities.h"
#include "cuNDArray_fileio.h"
#include "cuNDArray.h"
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
#include <iostream>
#include <algorithm>
#include <sstream>
#include <math_constants.h>
#include <boost/program_options.hpp>
#include <boost/make_shared.hpp>
#include "cuSolverUtils.h"
using namespace std;
using namespace Gadgetron;

namespace po = boost::program_options;


boost::shared_ptr<hoCuNDArray<float> > calculate_prior(boost::shared_ptr<CBCT_binning>  binning,boost::shared_ptr<CBCT_acquisition> ps, hoCuNDArray<float>& projections, std::vector<size_t> is_dims, floatd3 imageDimensions){
  std::cout << "Calculating FDK prior" << std::endl;
	boost::shared_ptr<CBCT_binning> binning_pics= binning->get_3d_binning();
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
	    boost::shared_ptr<hoCuNDArray<float> > prior(new hoCuNDArray<float>(*expand( prior3d.get(), is_dims.back() )));
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
  unsigned int subsets;
  float rho;
  float tv_weight,pics_weight;

  po::options_description desc("Allowed options");

  desc.add_options()
    ("help", "produce help message")
    ("acquisition,a", po::value<string>(&acquisition_filename)->default_value("acquisition.hdf5"), "Acquisition data")
    ("samples,n",po::value<unsigned int>(),"Number of samples per ray")
    ("output,f", po::value<string>(&outputFile)->default_value("reconstruction.real"), "Output filename")
    ("size,s",po::value<uintd3>(&imageSize)->default_value(uintd3(512,512,1)),"Image size in pixels")
    ("binning,b",po::value<string>(),"Binning file for 4d reconstruction")
    ("SAG","Use exact SAG correction if present")
    ("voxelSize,v",po::value<floatd3>(&voxelSize)->default_value(floatd3(0.488f,0.488f,1.0f)),"Voxel size in mm")
    ("dimensions,d",po::value<floatd3>(),"Image dimensions in mm. Overwrites voxelSize.")
    ("iterations,i",po::value<unsigned int>(&iterations)->default_value(10),"Number of iterations")
    ("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
    ("downsample,D",po::value<unsigned int>(&downsamples)->default_value(0),"Downsample projections this factor")
    ("subsets,u",po::value<unsigned int>(&subsets)->default_value(10),"Number of subsets to use")
    ("TV",po::value<float>(&tv_weight)->default_value(0),"Total variation weight")
    ("PICS",po::value<float>(&pics_weight)->default_value(0),"PICS weight")
    ("use_prior","Use an FDK prior")
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
    else if (a.type() == typeid(unsigned int)) std::cout << it->second.as<unsigned int>();
    else if (a.type() == typeid(float)) std::cout << it->second.as<float>();
    else if (a.type() == typeid(vector_td<float,3>)) std::cout << it->second.as<vector_td<float,3> >();
    else if (a.type() == typeid(vector_td<int,3>)) std::cout << it->second.as<vector_td<int,3> >();
    else if (a.type() == typeid(vector_td<unsigned int,3>)) std::cout << it->second.as<vector_td<unsigned int,3> >();
    else std::cout << "Unknown type" << std::endl;
    std::cout << std::endl;
  }

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
  osMOMSolver<cuNDArray<float>> solver;

  if (pics_weight > 0){
  	std::cout << "Calculating PICS prior" << std::endl;
  	hoCuConebeamProjectionOperator op;
  	auto bin3D = binning->get_3d_binning();
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

  	//write_nd_array(prior.get(),"fdk_prior.real");
  }

  // Define encoding matrix
  auto E = boost::make_shared<CBSubsetOperator<cuNDArray> >(subsets);


  //E->setup(ps,binning,imageDimensions);
  E->setup(ps,binning,imageDimensions);
  E->set_domain_dimensions(&is_dims);
  E->set_codomain_dimensions(ps->get_projections()->get_dimensions().get());

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
  solver.set_encoding_operator(E);
  //solver.set_domain_dimensions(&is_dims);
  solver.set_max_iterations(iterations);
  solver.set_output_mode(osSPSSolver<cuNDArray<float>>::OUTPUT_VERBOSE);
  //solver.set_non_negativity_constraint(true);
  //solver.set_rho(rho);

  if (tv_weight > 0){




  	auto total_variation = boost::make_shared<cuTvOperator<float,4>>();
  	total_variation->set_weight(tv_weight);
  	//total_variation->set_weight_array(weight_array);
  	solver.add_nonlinear_operator(total_variation);
  	solver.set_kappa(tv_weight);
  	/*
  	auto total_variation_t = boost::make_shared<cuTv1DOperator<float,4>>();
  	total_variation_t->set_weight(tv_weight);
  	solver.add_nonlinear_operator(total_variation_t);
  	*/
/*
  	auto total_variation2 = boost::make_shared<cuWTvOperator<float,4>>();
  	total_variation2->set_step(2);
  	total_variation2->set_weight(tv_weight);
  	total_variation2->set_weight_array(weight_array);
  	solver.add_nonlinear_operator(total_variation2);
  	auto total_variation3 = boost::make_shared<cuWTvOperator<float,4>>();
  	total_variation3->set_step(3);
  	total_variation3->set_weight(tv_weight);
  	total_variation3->set_weight_array(weight_array);
  	solver.add_nonlinear_operator(total_variation3);
*/
  }



  cuNDArray<float> projections(*ps->get_projections());
  std::cout << "Projection norm:" << nrm2(&projections) << std::endl;
  //E->set_use_offset_correction(false);
  E->offset_correct(&projections);
  std::cout << "Projection norm:" << nrm2(&projections) << std::endl;

  {
  	E->set_use_offset_correction(false);
  	linearOperator<cuNDArray<float>>* E_all = E.get();
  	auto precon_image = boost::make_shared<cuNDArray<float>>(is_dims);
  	fill(precon_image.get(),1.0f);
  	cuNDArray<float> tmp_proj(projections.get_dimensions());

  	E_all->mult_M(precon_image.get(),&tmp_proj);
  	E_all->mult_MH(&tmp_proj,precon_image.get());

   	clamp_min(precon_image.get(),1e-6f);
  	reciprocal_inplace(precon_image.get());
  	solver.set_preconditioning_image(precon_image);

  	std::cout << "Precon mean: " << mean(precon_image.get()) << std::endl;
  	E->set_use_offset_correction(true);

  }


/*
    boost::shared_ptr<hoCuNDArray<float> > prior;

  if (vm.count("use_prior")) {
  	prior = calculate_prior(binning,ps,projections,is_dims,imageDimensions);
  	solver.set_x0(prior);
  }
*/

  auto result = solver.solve(&projections);

  write_nd_array<float>( result.get(), outputFile.c_str());
}
