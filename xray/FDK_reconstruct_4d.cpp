#include "parameterparser.h"
#include "CBCT_acquisition.h"
#include "CBCT_binning.h"
#include "hoCuConebeamProjectionOperator.h"
#include "hoNDArray_fileio.h"
#include "hoCuNDArray_math.h"
#include "vector_td_utilities.h"
#include "GPUTimer.h"

#include <boost/program_options.hpp>
#include "dicomWriter.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <proton/hdf5_utils.h>

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
	std::string binning_filename;
 // Parse command line
  //
po::options_description desc("Allowed options");
 desc.add_options()
    ("help", "produce help message")
    ("acquisition,a", po::value<string>(&acquisition_filename)->default_value("acquisition.hdf5"), "Acquisition data")
    ("samples,n",po::value<unsigned int>(),"Number of samples per ray")
    ("output,f", po::value<string>(&image_filename)->default_value("fdk.real"), "Output filename")
    ("size,s",po::value<uintd3>(&imageSize)->default_value(uintd3(512,512,1)),"Image size in pixels")
    ("binning,b",po::value<string>(&binning_filename)->default_value("binning.hdf5"),"Binning file for 3d reconstruction (used to exclude projections)")
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
	acquisition->load(acquisition_filename);

	// Downsample projections if requested
	//

	{
		GPUTimer timer("Downsampling projections");
		acquisition->downsample(downsamples);
	}



	// Configuring...
	//

	uintd2 ps_dims_in_pixels( acquisition->get_projections()->get_size(0),
			acquisition->get_projections()->get_size(1) );

	floatd2 ps_dims_in_mm( acquisition->get_geometry()->get_FOV()[0],
			acquisition->get_geometry()->get_FOV()[1] );

	float SDD = acquisition->get_geometry()->get_SDD();
	float SAD = acquisition->get_geometry()->get_SAD();




	boost::shared_ptr<CBCT_binning> ps_bd4d(  new CBCT_binning());

	ps_bd4d->load(binning_filename);
	ps_bd4d->print(std::cout);

	// Load the binning data
		//

		boost::shared_ptr<CBCT_binning> binning=ps_bd4d->get_3d_binning() ;

	// Allocate array to hold the result
	//

	std::vector<size_t> is_dims{imageSize[0],imageSize[1],imageSize[2]};

	hoCuNDArray<float> fdk_3d(&is_dims);

	//
	// Standard 3D FDK reconstruction
	//

	boost::shared_ptr< hoCuConebeamProjectionOperator > E( new hoCuConebeamProjectionOperator() );

	E->setup( acquisition, binning, is_dims_in_mm );
	E->set_use_filtered_backprojection(true);

	hoCuNDArray<float> projections(*acquisition->get_projections());

	{
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



	size_t numBins = ps_bd4d->get_number_of_bins();
	is_dims.push_back(numBins);
	boost::shared_ptr< hoCuConebeamProjectionOperator >
	E4D( new hoCuConebeamProjectionOperator() );
	E4D->setup(acquisition,ps_bd4d,is_dims_in_mm);
	E4D->set_use_filtered_backprojection(true);
	E4D->set_domain_dimensions(&is_dims);

	hoCuNDArray<float> fdk(*expand(&fdk_3d,numBins));
	hoCuNDArray<float> diff_proj(projections.get_dimensions());

	E4D->mult_M(&fdk,&diff_proj);
	float scaler = dot(&diff_proj,&projections)/dot(&diff_proj,&diff_proj);
	std::cout << "Scale: " << scaler << std::endl;
	diff_proj *= scaler;
	//projections -= diff_proj;
	//projections *= -1.0f;

	hoCuNDArray<float> result(&is_dims);
	E4D->mult_MH(&projections,&result);
	//result *= 10.0f;
	//axpy(scaler,&fdk,&result);
	result += fdk;

	//write_nd_array<float>( &result, image_filename.c_str() );
	write_dicom(&result,"",is_dims_in_mm);
	saveNDArray2HDF5(&result,"fdk.hdf5",is_dims_in_mm,floatd3(0,0,0),"",0);
	return 0;
}
