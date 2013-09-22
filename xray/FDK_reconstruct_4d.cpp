#include "parameterparser.h"
#include "CBCT_acquisition.h"
#include "CBCT_binning.h"
#include "hoCudaConebeamProjectionOperator.h"
#include "hoNDArray_fileio.h"
#include "hoCuNDArray_math.h"
#include "vector_td_utilities.h"
#include "GPUTimer.h"

#include <iostream>
#include <algorithm>
#include <sstream>

using namespace Gadgetron;
using namespace std;

int main(int argc, char** argv) 
{ 
  // Parse command line
  //

  ParameterParser parms(1024);
  parms.add_parameter( 'd', COMMAND_LINE_STRING, 1, "Input acquisition filename (.hdf5)", true );
  parms.add_parameter( 'b', COMMAND_LINE_STRING, 1, "Binning filename (.hdf5) - 4D FDK only", false );
  parms.add_parameter( 'r', COMMAND_LINE_STRING, 1, "Output image filename (.real)", true, "reconstruction_FDK.real" );
  parms.add_parameter( 'm', COMMAND_LINE_INT, 3, "Matrix size (3d)", true, "256, 256, 144" );
  parms.add_parameter( 'f', COMMAND_LINE_FLOAT, 3, "FOV in mm (3d)", true, "448, 448, 252" );
  parms.add_parameter( 'F', COMMAND_LINE_INT, 1, "Use filtered backprojection (fbp)", true, "1" );
  parms.add_parameter( 'O', COMMAND_LINE_INT, 1, "Use oversampling in fbp", true, "0" );
  parms.add_parameter( 'H', COMMAND_LINE_FLOAT, 1, "Half-scan mode maximum angle", true, "0" );
  parms.add_parameter( 'P', COMMAND_LINE_INT, 1, "Projections per batch", true, "50" );

  parms.parse_parameter_list(argc, argv);
  if( parms.all_required_parameters_set() ) {
    parms.print_parameter_list();
  }
  else{
    parms.print_parameter_list();
    parms.print_usage();
    return 1;
  }
  
  std::string acquisition_filename = (char*)parms.get_parameter('d')->get_string_value();
  std::string binning_filename = (char*)parms.get_parameter('b')->get_string_value();
  std::string image_filename = (char*)parms.get_parameter('r')->get_string_value();

  // Load acquisition data
  //

  boost::shared_ptr<CBCT_acquisition> acquisition( new CBCT_acquisition() );
  acquisition->load(acquisition_filename);

  // Load the binning data
  //
  
  boost::shared_ptr<CBCT_binning> binning( new CBCT_binning() );
  binning->set_as_default_3d_bin(acquisition->get_projections()->get_size(2));
  binning->print();

  // Configuring...
  //

  uintd2 ps_dims_in_pixels( acquisition->get_projections()->get_size(0),
			    acquisition->get_projections()->get_size(1) );
  
  floatd2 ps_dims_in_mm( acquisition->get_geometry()->get_FOV()[0],
			 acquisition->get_geometry()->get_FOV()[1] );

  float SDD = acquisition->get_geometry()->get_SDD();
  float SAD = acquisition->get_geometry()->get_SAD();

  uintd3 is_dims_in_pixels( parms.get_parameter('m')->get_int_value(0),
			    parms.get_parameter('m')->get_int_value(1),
			    parms.get_parameter('m')->get_int_value(2) );
  
  floatd3 is_dims_in_mm( parms.get_parameter('f')->get_float_value(0), 
			 parms.get_parameter('f')->get_float_value(1), 
			 parms.get_parameter('f')->get_float_value(2) );
  
  bool use_fbp = parms.get_parameter('F')->get_int_value();
  bool use_fbp_os = parms.get_parameter('O')->get_int_value();
  float half_scan_max_angle = parms.get_parameter('H')->get_float_value();
  unsigned int projections_per_batch = parms.get_parameter('P')->get_int_value();

  // Allocate array to hold the result
  //
  
  std::vector<unsigned int> is_dims;
  is_dims.push_back(is_dims_in_pixels[0]);
  is_dims.push_back(is_dims_in_pixels[1]);
  is_dims.push_back(is_dims_in_pixels[2]);
  is_dims.push_back(1); // one temporal frame for 3d
  
  hoCuNDArray<float> fdk_3d(&is_dims);

  //
  // Standard 3D FDK reconstruction
  //
  
  boost::shared_ptr< hoCudaConebeamProjectionOperator > E( new hoCudaConebeamProjectionOperator() );
  
  //E->setup( acquisition, binning, projections_per_batch, 0, is_spacing_in_mm, 
  //	    use_fbp, use_fbp_os, (half_scan_max_angle == 0.0f) ? 2.0f*CUDART_PI_F : half_scan_max_angle );


  {
  	hoCuNDArray<float> projections(*acquisition->get_projections());
    GPUTimer timer("Running 3D FDK reconstruction");
    E->mult_MH( &projections, &fdk_3d );
  }

  write_nd_array<float>( &fdk_3d, "fdk.real" );

  /*4D FDK-MB algorithm starts here. McKinnon GC, RHT Bates,
   *
   *"Towards Imaging the Beating Heart Usefully with a Conventional CT Scanner,"
   *" Biomedical Engineering, IEEE Transactions on , vol.BME-28, no.2, pp.123,127, Feb. 1981
   * doi: 10.1109/TBME.1981.324785
   */
  /*
  CBCT_binning *ps_bd4d = new CBCT_binning();
  std::string binningdata_filename = (char*)parms.get_parameter('B')->get_string_value();
  std::cout << "binning data file: " << binningdata_filename << std::endl;
  ps_bd4d->load(binningdata_filename);
  ps_bd4d->print(std::cout);

  size_t numBins = ps_bd4d->get_number_of_bins();
  is_dims.push_back(numBins);
  boost::shared_ptr< hoCudaConebeamProjectionOperator >
  E4D( new hoCudaConebeamProjectionOperator() );*/
  /*E4D->setup( ps_g, ps_bd4d, ps_g->getAnglesArray(),ps_g->getOffsetXArray(),ps_g->getOffsetYArray(), ppb,
	      is_spacing_in_mm, ps_dims_in_pixels,
	      numSamplesPerRay,  true);*/
  /*  E4D->set_codomain_dimensions(projections->get_dimensions().get());
  E4D->set_domain_dimensions(&is_dims);

  hoNDArray<float> diff_proj(projections->get_dimensions());

  E->mult_M(&fdk,&diff_proj);
  *projections -= diff_proj;

  hoNDArray<float> result(&is_dims);
  E4D->mult_MH(projections.get(),&result);

  result += fdk;

  write_nd_array<float>( &result, outFile.c_str() );*/
  return 0;
}