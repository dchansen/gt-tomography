/*
  An example of how to register two 2d images using Horn-Schunk optical flow
*/

// Gadgetron includes
#include "cuDemonsSolver.h"
#include "cuTextureResampleOperator.h"
#include "cuNDArray.h"
#include "hoNDArray_fileio.h"
#include "parameterparser.h"
#include "cuCGHSOFSolver.h"
// Std includes
#include <iostream>
#include <cuLinearResampleOperator.h>

using namespace Gadgetron;
using namespace std;

// Define desired precision
typedef float _real; 

int main(int argc, char** argv)
{

  //
  // Parse command line
  //

  ParameterParser parms;
  parms.add_parameter( 'f', COMMAND_LINE_STRING, 1, "Fixed image file name (.real)", true );
  parms.add_parameter( 'm', COMMAND_LINE_STRING, 1, "Moving image file name (.real)", true );
  parms.add_parameter( 'r', COMMAND_LINE_STRING, 1, "Result file name", true, "displacement_field.real" );
  parms.add_parameter( 'a', COMMAND_LINE_FLOAT,  1, "Regularization weight (alpha)", true, "0.1" );
  parms.add_parameter( 's', COMMAND_LINE_FLOAT,  1, "Simga fluid", true, "1.0" );
  parms.add_parameter( 'd', COMMAND_LINE_FLOAT,  1, "Simga diff", true, "1.0" );
  parms.add_parameter( 'l', COMMAND_LINE_INT,    1, "Number of multiresolution levels", true, "3" );
  parms.add_parameter( 'i', COMMAND_LINE_INT,    1, "Number of iterations", true, "30" );
  
  parms.parse_parameter_list(argc, argv);
  if( parms.all_required_parameters_set() ){
    cout << " Running registration with the following parameters: " << endl;
    parms.print_parameter_list();
  }
  else{
    cout << " Some required parameters are missing: " << endl;
    parms.print_parameter_list();
    parms.print_usage();
    return 1;
  }

  cudaSetDevice(0);
  cudaDeviceReset();
  
  // Load sample data from disk
  //
  
  boost::shared_ptr< hoNDArray<_real> > host_fixed = 
    read_nd_array<_real>((char*)parms.get_parameter('f')->get_string_value());

  boost::shared_ptr< hoNDArray<_real> > host_moving = 
    read_nd_array<_real>((char*)parms.get_parameter('m')->get_string_value());
  
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

  cuNDArray<_real> fixed_image(host_fixed.get());
  cuNDArray<_real> moving_image(host_moving.get());
  
  _real alpha = (_real) parms.get_parameter('a')->get_float_value();

  unsigned int multires_levels = parms.get_parameter('l')->get_int_value();

  // Use bilinear interpolation for resampling
  //

  boost::shared_ptr< cuLinearResampleOperator<_real,3> > R( new cuLinearResampleOperator<_real,3>() );

  // Setup solver
  //
  
  cuDemonsSolver<_real,3> HS;
  HS.set_interpolator( R );
  HS.set_output_mode( cuDemonsSolver<_real,3>::OUTPUT_VERBOSE );
  HS.set_max_num_iterations_per_level( parms.get_parameter('i')->get_int_value());
  HS.set_num_multires_levels( multires_levels );
  HS.set_alpha(alpha);

  HS.set_sigmaDiff(parms.get_parameter('d')->get_float_value());
  HS.set_sigmaFluid(parms.get_parameter('s')->get_float_value());
  

  // Run registration
  //

  boost::shared_ptr< cuNDArray<_real> > result = HS.solve( &fixed_image, &moving_image );

  if( !result.get() ){
    cout << endl << "Registration solver failed. Quitting!\n" << endl;
    return 1;
  }
  
  auto deformed_moving = deform_image( &moving_image, result.get() );
  
  // All done, write out the result
  //

  boost::shared_ptr< hoNDArray<_real> > host_result = result->to_host();
  write_nd_array<_real>(host_result.get(), (char*)parms.get_parameter('r')->get_string_value());

  host_result = deformed_moving.to_host();
  write_nd_array<_real>(host_result.get(), "def_moving.real" );
  
  return 0;
}
