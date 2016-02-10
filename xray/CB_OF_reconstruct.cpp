#include "hoNDArray_fileio.h"
#include "parameterparser.h"
#include "cuNDArray.h"
#include "hoCuNDArray.h"
#include "hoCuNDArray_utils.h"
#include "cuNDArray_utils.h"
#include "vector_td_utilities.h"
#include "hoCuTvOperator.h"
#include "cuCKOpticalFlowSolver.h"
#include "cuLinearResampleOperator.h"
#include "cuDownsampleOperator.h"
#include "GPUTimer.h"
#include "hoCuCgSolver.h"
#include "hoCuNCGSolver.h"
#include "hoCuEncodingOperatorContainer.h"

#include "CBCT_acquisition.h"
#include "CBCT_binning.h"
#include "hoCuConebeamProjectionOperator.h"
#include "hoCuOFConebeamProjectionOperator.h"
#include "cuCGHSOFSolver.h"

#include "hdf5_utils.h"

#include <string>
#include <sstream>
#include <boost/program_options.hpp>
using namespace Gadgetron;
using namespace std;
namespace po = boost::program_options;

boost::shared_ptr< hoCuNDArray<float> >
perform_registration( boost::shared_ptr< hoCuNDArray<float> > volume, unsigned int phase, float of_alpha, float of_beta, unsigned int num_multires_levels )
{
	std::vector<size_t> volume_dims_3d = *volume->get_dimensions();
	volume_dims_3d.pop_back();
	std::vector<size_t> volume_dims_3d_3 = volume_dims_3d;
	volume_dims_3d_3.push_back(3);

	size_t num_elements_3d = volume_dims_3d[0]* volume_dims_3d[1]* volume_dims_3d[2];
	size_t num_phases = volume->get_size(3);

	boost::shared_ptr< hoCuNDArray<float> > host_result_field( new hoCuNDArray<float> );
	{
		std::vector<size_t> volume_dims_4d = volume_dims_3d;
		volume_dims_4d.push_back(3);
		volume_dims_4d.push_back(volume->get_size(3)-1);
		host_result_field->create( &volume_dims_4d );
	}

	// Upload host data to device
	//

	unsigned int counter = 0;
	hoCuNDArray<float> host_moving( &volume_dims_3d, volume->get_data_ptr()+phase*num_elements_3d );

	for( unsigned int i=0; i<num_phases; i++ ){

		if( i==phase )
			continue;

		hoCuNDArray<float> host_fixed( &volume_dims_3d, volume->get_data_ptr()+i*num_elements_3d );

		cuNDArray<float> fixed_image(&host_fixed);
		cuNDArray<float> moving_image(&host_moving);

		boost::shared_ptr< cuLinearResampleOperator<float,3> > R( new cuLinearResampleOperator<float,3>() );

		// Setup solver
		//
		cuCGHSOFSolver<float,3> OFs;
		//cuCKOpticalFlowSolver<float,3> OFs;
		OFs.set_interpolator( R );
		OFs.set_output_mode( cuCKOpticalFlowSolver<float,3>::OUTPUT_VERBOSE );
		OFs.set_max_num_iterations_per_level( 500 );
		OFs.set_num_multires_levels( num_multires_levels );
		OFs.set_alpha(of_alpha);
		//OFs.set_beta(of_beta);
		//OFs.set_limit(0.01f);

		// Run registration
		//
		std::cout << "Penguin " << std::endl;
		boost::shared_ptr< cuNDArray<float> > result = OFs.solve( &fixed_image, &moving_image );

		std::cout << "Penguin " << std::endl;
		cuNDArray<float> dev_sub;
		dev_sub.create( &volume_dims_3d_3, result->get_data_ptr() );

		hoCuNDArray<float> host_sub( &volume_dims_3d_3, host_result_field->get_data_ptr()+counter*num_elements_3d*3 );
		host_sub = *dev_sub.to_host();

		counter++;
	}

	/*
 {
  std::cout << std::endl << "Writing out registration results for phase " << phase << "." << std::endl;
  char filename[256];
  sprintf(&(filename[0]), "def_moving_%i.real", phase);
  write_nd_array<float>(&host_result_image, (char*)filename);
 }
	 */

	{
		// Permute the displacement field (temporal dimension before 'vector' dimension)
		std::vector<size_t> order;
		order.push_back(0); order.push_back(1); order.push_back(2);
		order.push_back(4); order.push_back(3);
		cuNDArray<float> tmp(host_result_field.get()); // permute is too slow on the host
		*host_result_field = *permute(&tmp, &order);

		/*char filename[256];
    sprintf(&(filename[0]), "displacement_field_%i.real", phase);
    write_nd_array<float>(host_result_field.get(), (char*)filename);*/
	}

	return host_result_field;
}

int main(int argc, char** argv) 
{
	string acquisition_filename;
	string tv_recon_filename;
	string binning_filename;
	string outputFile;
	uintd3 imageSize;
	floatd3 voxelSize;
	int device;
	unsigned int downsamples;
	unsigned int iterations;
	float of_alpha,of_beta;
	unsigned int num_reg_downsamples;
	po::options_description desc("Allowed options");

	desc.add_options()
    								("help", "produce help message")
    								("acquisition,a", po::value<string>(&acquisition_filename)->default_value("acquisition.hdf5"), "Acquisition data")
    								("samples,n",po::value<unsigned int>(),"Number of samples per ray")
    								("output,f", po::value<string>(&outputFile)->default_value("reconstruction_REG.hdf5"), "Output filename")
    								("tv_recon,t", po::value<string>(&tv_recon_filename)->default_value("reconstruction.real"), "Input TV recon")
    								("size,s",po::value<uintd3>(&imageSize)->default_value(uintd3(512,512,1)),"Image size in pixels")
    								("binning,b",po::value<string>(&binning_filename)->default_value("binning.hdf5"),"Binning file for 4d reconstruction")
    								("SAG","Use exact SAG correction if present")
    								("voxelSize,v",po::value<floatd3>(&voxelSize)->default_value(floatd3(0.488f,0.488f,1.0f)),"Voxel size in mm")
    								("dimensions,d",po::value<floatd3>(),"Image dimensions in mm. Overwrites voxelSize.")
    								("iterations,i",po::value<unsigned int>(&iterations)->default_value(10),"Number of iterations")
    								("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
    								("downsample,D",po::value<unsigned int>(&downsamples)->default_value(0),"Downsample projections this factor")
    								("reg_downsample,R",po::value<unsigned int>(&num_reg_downsamples)->default_value(1),"Downsample projections this factor")
    								("alpha",po::value<float>(&of_alpha)->default_value(0.05f),"Alpha for optical Flow");
	("alpha",po::value<float>(&of_beta)->default_value(1.0f),"Alpha for optical Flow");
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

	// Load binning data
	//

	boost::shared_ptr<CBCT_binning> binning_4d( new CBCT_binning() );
	{
		GPUTimer timer("Loading binning data");
		binning_4d->load(binning_filename);
	}

	// Load intermediate reconstruction
	// 

	boost::shared_ptr< hoCuNDArray<float> > tv_recon = boost::make_shared<hoCuNDArray<float>>();
	{
		GPUTimer timer("Loading intermediate reconstruction");
		*tv_recon = *read_nd_array<float>(tv_recon_filename.c_str());
	}

	if( tv_recon->get_number_of_dimensions() != 4 ){
		printf("\nInput volume for registration must be four-dimensional");
		exit(1);
	}

	// Configuring...
	//

	uintd2 ps_dims_in_pixels( acquisition->get_projections()->get_size(0),
			acquisition->get_projections()->get_size(1) );

	floatd2 ps_dims_in_mm( acquisition->get_geometry()->get_FOV()[0],
			acquisition->get_geometry()->get_FOV()[1] );

	float SDD = acquisition->get_geometry()->get_SDD();
	float SAD = acquisition->get_geometry()->get_SAD();
	size_t num_phases = binning_4d->get_number_of_bins();

	//float tv_weight = parms.get_parameter('T')->get_float_value();

	floatd3 imageDimensions;
	if (vm.count("dimensions")){
		imageDimensions = vm["dimensions"].as<floatd3>();
		voxelSize = imageDimensions/imageSize;
	}
	else imageDimensions = voxelSize*imageSize;

	std::vector<size_t> is_dims_3d = to_std_vector((uint64d3)imageSize);

	std::vector<size_t> is_dims_4d = is_dims_3d;
	is_dims_4d.push_back(num_phases);

	// Create mask to zero-out the areas of the images that ar enot fully sampled
	// Do this by a 3D unfiltered backprojection of projections of pure ones.
	// - threshold on a 2% level
	//

	hoCuNDArray<float> image_filter( &is_dims_3d );

	{
		GPUTimer timer("Filtering TV reconstruction");
		hoCuNDArray<float> projections( *acquisition->get_projections() );
		boost::shared_ptr<CBCT_binning> binning_3d( new CBCT_binning );
		hoCuConebeamProjectionOperator E;
		fill( &projections, 1.0f );
		binning_3d->set_as_default_3d_bin(projections.get_size(2));
		E.setup( acquisition, binning_3d, imageDimensions);
		E.set_use_filtered_backprojection(false);
		E.mult_MH( &projections, &image_filter );
		if( !E.get_use_offset_correction() ){
			std::cout << std::endl << "Error: currently offset correction is assumed to guide the TV filtering. Offset correction unavailable" << std::endl;
			exit(1);
		}

		float threshold = 0.98f * 0.5f; // 0.5 is the intensity after offset correction
		clamp( &image_filter, threshold, threshold, 0.0f, 1.0f );

		*tv_recon *= image_filter;
	}

	// Downsample TV recon if specified
	// - for every downsample of OF image space, dowmsample OF projections as well
	//


	boost::shared_ptr<hoCuNDArray<float> > projections( new hoCuNDArray<float>( *acquisition->get_projections() ));

	hoCuNDArray<float> image_3d(&is_dims_3d);
	hoCuNDArray<float> image_4d(&is_dims_4d);

	// Define encoding operator for the reconstruction -- a "plain" CBCT operator
	//

	boost::shared_ptr< hoCuConebeamProjectionOperator > E( new hoCuConebeamProjectionOperator() );
	E->setup( acquisition, imageDimensions);
	E->set_domain_dimensions(&is_dims_3d);
	E->set_codomain_dimensions(acquisition->get_projections()->get_dimensions().get());
	E->offset_correct(projections.get());

	boost::shared_ptr<hoCuNDArray<float> > OF_projections = projections;

	if( num_reg_downsamples > 0 ) {
		GPUTimer timer("Downsampling TV reconstruction (and OF operator projections accordingly)");

		std::vector<size_t> tmp_dims_3d = is_dims_3d;
		for( unsigned int i=0; i<tmp_dims_3d.size(); i++ ){
			for( unsigned int d=0; d<num_reg_downsamples; d++ ){
				if( (tmp_dims_3d[i]%2)==1 )
					throw std::runtime_error("Error: input volume for registration must have even size in all dimensions (at all levels) in order to downsample");
				tmp_dims_3d[i] /= 2;
			}
		}

		cuNDArray<float> tmp_image_in(tv_recon.get());
		cuNDArray<float> tmp_proj_in(OF_projections.get());

		std::vector<size_t> volume_dims_4d = *tv_recon->get_dimensions();
		std::vector<size_t> proj_dims_3d = *projections->get_dimensions();

		for( unsigned int d=0; d<num_reg_downsamples; d++ ){

			for( unsigned int i=0; i<3; i++ ) volume_dims_4d[i] /= 2; // do not downsample temporal dimension
			for( unsigned int i=0; i<2; i++ ) proj_dims_3d[i] /= 2;   // do not downsample #projections dimension

			cuNDArray<float> tmp_image_out(&volume_dims_4d);
			cuNDArray<float> tmp_proj_out(&proj_dims_3d);

			cuDownsampleOperator<float,3> D_image;
			cuDownsampleOperator<float,2> D_proj;

			D_image.mult_M( &tmp_image_in, &tmp_image_out );
			D_proj.mult_M( &tmp_proj_in, &tmp_proj_out );

			tmp_image_in = tmp_image_out;
			tmp_proj_in = tmp_proj_out;
		}

		*tv_recon = tmp_image_in;
		OF_projections = boost::shared_ptr<hoCuNDArray<float> >( new hoCuNDArray<float>( *tmp_proj_in.to_host() ));
	}

	// Allocate 3d array for phase-by-phase reconstruction
	// and 4D array to hold the overall result
	//


	// Define the optical flow regularization operator 
	//

	boost::shared_ptr< hoCuOFConebeamProjectionOperator > OF( new hoCuOFConebeamProjectionOperator() );
	OF->setup( acquisition, binning_4d, imageDimensions);
	OF->set_domain_dimensions(&is_dims_3d);
	OF->set_codomain_dimensions(OF_projections->get_dimensions().get());

	//OF->set_weight(1.0f/float(num_phases));

	// Combine the two operators in an operator container
	//

	boost::shared_ptr< hoCuEncodingOperatorContainer<float> > opContainer( new hoCuEncodingOperatorContainer<float>() );

	opContainer->add_operator(E);
	opContainer->add_operator(OF);

	std::vector<hoCuNDArray<float>*> codoms;
	codoms.push_back( projections.get() );
	codoms.push_back( OF_projections.get() );

	boost::shared_ptr< hoCuNDArray<float> > f = opContainer->create_codomain( codoms );
	codoms.clear(); projections.reset(); OF_projections.reset();

	// Setup solver
	//

	hoCuCgSolver<float> solver;
	//hoCuNCGSolver<float> solver;

	solver.set_encoding_operator( opContainer );

	/*if( tv_weight > 0.0f ){
		boost::shared_ptr<hoCuTvOperator<float,3> > tvOp(new hoCuTvOperator<float,3>());
		tvOp->set_weight(tv_weight);
		tvOp->set_limit(float(1e-6));
		solver.add_nonlinear_operator(tvOp);
    }*/

	//solver.set_non_negativity_constraint(true);
	//solver.set_output_mode(hoCuNCGSolver<float>::OUTPUT_VERBOSE);
	solver.set_output_mode(hoCuCgSolver<float>::OUTPUT_VERBOSE);
	solver.set_max_iterations(iterations);
	solver.set_tc_tolerance(float(1e-9));

	// Solve
	//

	//for( unsigned int phase=0; phase<binning_4d->get_number_of_bins(); phase++ ){

	for( unsigned int phase=0; phase<num_phases; phase++ ){

		// Define the binning for the current phase
		//

		std::cout << "Phase " << phase << std::endl;
		std::vector<unsigned int> bin = binning_4d->get_bin(phase);
		boost::shared_ptr<CBCT_binning> binning_phase( new CBCT_binning() );
		binning_phase->set_bin( bin, 0 );
		E->set_binning(binning_phase);

		OF->set_encoding_phase(phase);
		{
			boost::shared_ptr<hoCuNDArray<float>> displacements;
			{
			GPUTimer timer("Registration");
			displacements =	perform_registration( tv_recon, phase, of_alpha, of_beta, 3 - num_reg_downsamples );
			}
			{
			GPUTimer timer("Setting displacements");
			OF->set_displacement_field(displacements);
			}
		}

		boost::shared_ptr<hoCuNDArray<float> > result_phase = solver.solve( f.get() );

		// Copy result to 4d array
		//

		hoCuNDArray<float> host_result_3d( &is_dims_3d, image_4d.get_data_ptr()+phase*result_phase->get_number_of_elements() );
		host_result_3d = *result_phase;

		// Apply "cropping" filter
		//

		host_result_3d *= image_filter;


		// Write out every 3d volume
		//

		/*{
			char filename[256];
			sprintf(&(filename[0]), "reconstruction_3d_%i.real", phase);
			write_nd_array<float>( result_phase.get(), (char*)filename );
			}*/
	}

	saveNDArray2HDF5(&image_4d,outputFile,imageDimensions,floatd3(0,0,0),command_line_string.str(),iterations);


	return 0;
}
