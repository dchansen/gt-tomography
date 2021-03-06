#include "hoCuNDArray_utils.h"
#include "radial_utilities.h"
#include "hoNDArray_fileio.h"
#include "cuNDArray.h"
#include "imageOperator.h"
#include "identityOperator.h"
#include "hoPartialDerivativeOperator.h"
#include "hoCuConebeamProjectionOperator.h"
#include "cuConvolutionOperator.h"
#include "hoCuIdentityOperator.h"
#include "hoCuNDArray_blas.h"
#include "hoCuNDArray_operators.h"
#include "hoCuNDArray_blas.h"
#include "hoCuCgSolver.h"
#include "CBCT_acquisition.h"
#include "complext.h"
#include "encodingOperatorContainer.h"
#include "vector_td_io.h"
#include "hoCuPartialDerivativeOperator.h"
#include "GPUTimer.h"

#include <iostream>
#include <algorithm>
#include <sstream>
#include <math_constants.h>
#include <boost/program_options.hpp>

using namespace std;
using namespace Gadgetron;

namespace po = boost::program_options;

int 
main(int argc, char** argv)
{
  string acquisition_filename;
  string outputFile;
  uintd3 imageSize;
  floatd3 voxelSize;
  float reg_weight;
  int device;
  unsigned int dump;
  unsigned int downsamples;
  unsigned int iterations;

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
    ("weight,w",po::value<float>(&reg_weight)->default_value(float(0.1f)),"Regularization weight")
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
  } else 
    binning->set_as_default_3d_bin(ps->get_projections()->get_size(2));

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

  hoCuNDArray<float> projections(*ps->get_projections());

  // Define encoding operator
  boost::shared_ptr< hoCuConebeamProjectionOperator >
    E( new hoCuConebeamProjectionOperator() );

  E->setup(ps,binning,imageDimensions);
  E->set_domain_dimensions(&is_dims);
  E->set_codomain_dimensions(ps->get_projections()->get_dimensions().get());

  // Define regularization operator
  boost::shared_ptr< hoCuIdentityOperator<float> >
    I( new hoCuIdentityOperator<float>() );
  
  I->set_weight(reg_weight);

  hoCuCgSolver<float> solver;

  solver.set_encoding_operator(E);

  if( reg_weight>0.0f ) {
    std::cout << "Adding identity operator with weight " << reg_weight << std::endl;
    solver.add_regularization_operator(I);
  }

  solver.set_max_iterations(iterations);
  solver.set_tc_tolerance(1e-8);
  solver.set_output_mode(hoCuCgSolver<float>::OUTPUT_VERBOSE);

  /*  if (vm.count("TV")){
    boost::shared_ptr<hoCuPartialDerivativeOperator<float,4> > dx (new hoCuPartialDerivativeOperator<float,4>(0) );
    boost::shared_ptr<hoCuPartialDerivativeOperator<float,4> > dy (new hoCuPartialDerivativeOperator<float,4>(1) );
    boost::shared_ptr<hoCuPartialDerivativeOperator<float,4> > dz (new hoCuPartialDerivativeOperator<float,4>(2) );
    boost::shared_ptr<hoCuPartialDerivativeOperator<float,4> > dt (new hoCuPartialDerivativeOperator<float,4>(3) );

    dx->set_codomain_dimensions(&is_dims);
    dy->set_codomain_dimensions(&is_dims);
    dz->set_codomain_dimensions(&is_dims);
    dt->set_codomain_dimensions(&is_dims);

    dx->set_domain_dimensions(&is_dims);
    dy->set_domain_dimensions(&is_dims);
    dz->set_domain_dimensions(&is_dims);
    dt->set_domain_dimensions(&is_dims);

    dx->set_weight(vm["TV"].as<float>());
    dy->set_weight(vm["TV"].as<float>());
    dz->set_weight(vm["TV"].as<float>());
    dt->set_weight(vm["TV"].as<float>());

    solver.add_regularization_group_operator(dx);
    solver.add_regularization_group_operator(dy);
    solver.add_regularization_group_operator(dz);
    solver.add_regularization_group_operator(dt);
    solver.add_group(1);
    }*/

  // Run solver
  //

  boost::shared_ptr< hoCuNDArray<float> > result;

  {
    GPUTimer timer("\nRunning conjugate gradient solver");
    result = solver.solve(&projections);
  }

  write_nd_array<float>( result.get(), outputFile.c_str());
}
