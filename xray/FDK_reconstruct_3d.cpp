#include "parameterparser.h"
#include "CBCT_acquisition.h"
#include "CBCT_binning.h"
#include "hoCuConebeamProjectionOperator.h"
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
#include "dicomWriter.h"
using namespace Gadgetron;
using namespace std;

namespace po = boost::program_options;
int main(int argc, char** argv) 
{ 
  
  std::string acquisition_filename;
  std::string image_filename;
	unsigned int downsamples;
	uintd3 imageSize;
	floatd3 is_dims_in_mm;
	int device;
 // Parse command line
  //
po::options_description desc("Allowed options");

  desc.add_options()
    ("help", "produce help message")
    ("acquisition,a", po::value<string>(&acquisition_filename)->default_value("acquisition.hdf5"), "Acquisition data")
    ("samples,n",po::value<unsigned int>(),"Number of samples per ray")
    ("output,f", po::value<string>(&image_filename)->default_value("fdk.real"), "Output filename")
    ("size,s",po::value<uintd3>(&imageSize)->default_value(uintd3(512,512,1)),"Image size in pixels")
    ("binning,b",po::value<string>(),"Binning file for 3d reconstruction (used to exclude projections)")
    ("SAG","Use exact SAG correction if present")
    ("dimensions,d",po::value<floatd3>(&is_dims_in_mm)->default_value(floatd3(256,256,256)),"Image dimensions in mm. Overwrites voxelSize.")
    ("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
    ("downsample,D",po::value<unsigned int>(&downsamples)->default_value(0),"Downsample projections this factor")
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

  // Load acquisition data
  //

  boost::shared_ptr<CBCT_acquisition> acquisition( new CBCT_acquisition() );

  {
    GPUTimer timer("Loading projections");
    acquisition->load(acquisition_filename);
  }

	// Downsample projections if requested
	//

	{
		GPUTimer timer("Downsampling projections");
		acquisition->downsample(downsamples);
	}
  
  // Load or generate binning data
  //
  
  boost::shared_ptr<CBCT_binning> binning( new CBCT_binning() );

  if (vm.count("binning")){
    std::string binningdata_filename = vm["binning"].as<string>(); 
    std::cout << "Using binning data file: " << binningdata_filename << std::endl;
    binning->load(binningdata_filename);
    binning = binning->get_3d_binning();
  } 
  else 
    binning->set_as_default_3d_bin(acquisition->get_projections()->get_size(2));

  // Configuring...
  //

  uintd2 ps_dims_in_pixels( acquisition->get_projections()->get_size(0),
			    acquisition->get_projections()->get_size(1) );
  
  floatd2 ps_dims_in_mm( acquisition->get_geometry()->get_FOV()[0],
			 acquisition->get_geometry()->get_FOV()[1] );

  float SDD = acquisition->get_geometry()->get_SDD();
  float SAD = acquisition->get_geometry()->get_SAD();

 

  // Allocate array to hold the result
  //
  
  std::vector<size_t> is_dims = {imageSize[0],imageSize[1],imageSize[2]};
  hoCuNDArray<float> fdk_3d(&is_dims);
  hoCuNDArray<float> projections(*acquisition->get_projections());  

  // Define conebeam projection operator
  // - and configure based on input parameters
  //
  
  boost::shared_ptr< hoCuConebeamProjectionOperator > E( new hoCuConebeamProjectionOperator() );

  E->setup( acquisition, binning, is_dims_in_mm );
  E->set_use_filtered_backprojection(true);

 
  // Initialize the device
  // - just to report more accurate timings
  //

  cudaThreadSynchronize();

  //
  // Standard 3D FDK reconstruction
  //

  {
    GPUTimer timer("Running 3D FDK reconstruction");
    E->mult_MH( &projections, &fdk_3d );
    cudaThreadSynchronize();
  }

  write_nd_array<float>( &fdk_3d, image_filename.c_str() );

  std::string s = "";
  write_dicom(&fdk_3d,s,is_dims_in_mm);

/*
  cuSbCgSolver<float> sb;
  auto id = boost::make_shared<identityOperator<cuNDArray<float>>>();
  id->set_domain_dimensions(&is_dims);
  id->set_codomain_dimensions(&is_dims);
  id->set_weight(5);

  auto wave = boost::make_shared<cuATrousOperator<float>>();
  wave->set_levels({4,4,4});
  wave->set_domain_dimensions(&is_dims);
  wave->set_weight(10);

  sb.set_encoding_operator(id);
  sb.set_max_outer_iterations(200);

  sb.add_regularization_operator(wave,1);
  sb.set_output_mode(cuSbCgSolver<float>::OUTPUT_VERBOSE);
  sb.get_inner_solver()->set_output_mode(cuCgSolver<float>::OUTPUT_VERBOSE);
  sb.get_inner_solver()->set_max_iterations(10);
  sb.get_inner_solver()->set_tc_tolerance(1e-8);
  cuNDArray<float> cu_fdk(fdk_3d);
  auto result = sb.solve(&cu_fdk);
*/
/*
  cuNDArray<float> cu_fdk(fdk_3d);

  cuDCTOperator<float> dctOp;
  dctOp.set_domain_dimensions(cu_fdk.get_dimensions().get());
  cuNDArray<float> tmp(dctOp.get_codomain_dimensions());
  dctOp.mult_M(&cu_fdk,&tmp,false);
  dctOp.inverse(&tmp,&cu_fdk,false);
  write_nd_array(&cu_fdk,"fdk_wavelet.real");
*/

  return 0;
}
