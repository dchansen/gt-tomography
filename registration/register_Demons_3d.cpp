/*
  An example of how to register two 2d images using Horn-Schunk optical flow
*/

// Gadgetron includes
#include "cuDemonsSolver.h"
#include "cuNDArray.h"
#include "hoNDArray_fileio.h"
#include "parameterparser.h"
#include "cuCGHSOFSolver.h"
// Std includes
#include <iostream>
#include <cuLinearResampleOperator.h>
#include <boost/program_options.hpp>
#include "cuNDArray_fileio.h"
#include "vector_td_io.h"

using namespace Gadgetron;
using namespace std;

// Define desired precision


namespace po = boost::program_options;
int main(int argc, char** argv)
{

  //
  // Parse command line
  //

  string moving_filename,fixed_filename,vfield_filename;
  float alpha, sigma_diff,sigma_fluid,sigma_int,sigma_vdiff;
  int iterations,levels,device;
  vector_td<float,3> phys_dims;
  bool composite,exponential;
  po::options_description desc("Allowed options");
  desc.add_options()
          ("help", "produce help message")
          ("moving,m", po::value<string>(&moving_filename), "Moving image")
          ("fixed,f", po::value<string>(&fixed_filename), "Fixed image")
          ("output,o", po::value<string>(&vfield_filename)->default_value("deformation_field.real"), "Output filename")
          ("alpha,a",po::value<float>(&alpha)->default_value(0.5),"Maximum step length per iteration")
          ("sigma_diff",po::value<float>(&sigma_diff)->default_value(1),"Diffusion sigma for regularization")
          ("sigma_fluid",po::value<float>(&sigma_fluid)->default_value(0),"Fluid sigma for regularization")
          ("sigma_int",po::value<float>(&sigma_int)->default_value(0),"Intensity sigma for regularization (bilateral)")
          ("iterations,i",po::value<int>(&iterations)->default_value(30),"Number of iterations to use")
          ("levels",po::value<int>(&levels)->default_value(1),"Number of multiresolution levels to use")
          ("sigma_vdiff",po::value<float>(&sigma_vdiff)->default_value(0),"Vector field difference sigma for regularization (bilateral)")
          ("composite",po::value<bool>(&composite)->default_value(true),"Do proper vector composition when adding vector fields")
          ("exponential",po::value<bool>(&exponential)->default_value(true),"Use exponential calculations (ensures invertible field)")
          ("device",po::value<int>(&device)->default_value(0),"Cuda device to use")
          ("physical_dims",po::value<vector_td<float,3>>(&phys_dims)->default_value(vector_td<float,3>(1,1,1)),"Physical dimension in mm")
          ("NGF",po::value<float>(),"Strength of normalized gradient field")
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
  
  // Load sample data from disk
  //
  
  boost::shared_ptr< hoNDArray<float> > host_fixed = 
    read_nd_array<float>(fixed_filename.c_str());

  boost::shared_ptr< hoNDArray<float> > host_moving = 
    read_nd_array<float>(moving_filename.c_str());
  
  if( !host_fixed.get() || !host_moving.get() ){
    cout << endl << "One of the input images is not found. Quitting!\n" << endl;
    return 1;
  }
  
  unsigned int num_fixed_dims = host_fixed->get_number_of_dimensions();
  unsigned int num_moving_dims = host_moving->get_number_of_dimensions();

  if( !(num_fixed_dims == 2 || num_fixed_dims == 3)  ){
    cout << endl << "The fixed image is not two- or three-dimensional. Quitting!\n" << endl;
    return 1;
  }
  
  if( !(num_moving_dims == 2 || num_moving_dims == 3)  ){
    cout << endl << "The moving image is not two- or three-dimensional. Quitting!\n" << endl;
    return 1;
  }
  
  // Upload host data to device
  //

  cuNDArray<float> fixed_image(host_fixed.get());
  cuNDArray<float> moving_image(host_moving.get());
  


  floatd3 sigma_diff_v = floatd3(sigma_diff);
  sigma_diff_v /= phys_dims;



  // Use bilinear interpolation for resampling
  //



  // Setup solver
  //
  
  cuDemonsSolver<float,3> HS;

  HS.set_iterations( iterations);

  HS.set_alpha(alpha);

  HS.set_sigmaDiff(sigma_diff_v);
  HS.set_sigmaFluid(sigma_fluid);
  HS.set_sigmaInt(sigma_int);
  HS.set_sigmaVDiff(sigma_vdiff);
  HS.set_compositive(composite);
  HS.set_exponential(exponential);
  if (vm.count("NGF")){
    HS.use_normalized_gradient_field(vm["NGF"].as<float>());
  }

//  HS.use_normalized_gradient_field(0.01);

  // Run registration
  //

  cuNDArray<float> result = HS.multi_level_reg( &fixed_image, &moving_image,levels );


  auto deformed_moving = deform_image( &moving_image, result );
  
  // All done, write out the result
  //

  write_nd_array<float>(&result,vfield_filename.c_str() );

  auto host_result = deformed_moving.to_host();
  write_nd_array<float>(host_result.get(), "def_moving.real" );

  auto jac = Jacobian(&result);
  jac -= 1.0f;

  write_nd_array<float>(&jac, "jacobian.real" );
  
  return 0;
}
