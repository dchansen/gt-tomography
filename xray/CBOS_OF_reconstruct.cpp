
#include "hoCuNDArray_utils.h"
#include "radial_utilities.h"
#include "hoNDArray_fileio.h"
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
#include "hoCuNDArray_utils.h"
#include "hoCuPartialDerivativeOperator.h"
#include "CBSubsetOperator.h"
#include "osMOMSolverD.h"
#include "osMOMSolverD3.h"
#include "cuSolverUtils.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <math_constants.h>
#include <boost/program_options.hpp>
#include <boost/make_shared.hpp>
#include <solvers/osMOMSolverD3.h>
#include "hoLinearResampleOperator_eigen.h"
#include "multiplicationOperatorContainer.h"
#include "cuLinearResampleOperator.h"
#include "multiresRegistrationSolver.h"
#include "cuCGHSOFSolver.h"

#include "hdf5_utils.h"
#include "hoCuOFPartialDerivativeOperator.h"

using namespace std;
using namespace Gadgetron;

namespace po = boost::program_options;


boost::shared_ptr< hoCuNDArray<float> >
perform_registration( boost::shared_ptr< hoCuNDArray<float> > volume, float of_alpha, float of_beta, unsigned int num_multires_levels )
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
		volume_dims_4d.push_back(volume->get_size(3));
		host_result_field->create( &volume_dims_4d );
	}

	// Upload host data to device
	//

	for( unsigned int i=0; i<num_phases; i++ ){

		hoCuNDArray<float> host_fixed( &volume_dims_3d, volume->get_data_ptr()+((i+1)%num_phases)*num_elements_3d );
		hoCuNDArray<float> host_moving( &volume_dims_3d, volume->get_data_ptr()+i*num_elements_3d );

		cuNDArray<float> fixed_image(&host_fixed);
		cuNDArray<float> moving_image(&host_moving);

		boost::shared_ptr< cuLinearResampleOperator<float,3> > R( new cuLinearResampleOperator<float,3>() );

		// Setup solver
		//
		cuCGHSOFSolver<float,3> OFs;
		//cuCKOpticalFlowSolver<float,3> OFs;
		OFs.set_interpolator( R );
		OFs.set_output_mode( cuCGHSOFSolver<float,3>::OUTPUT_VERBOSE );
		OFs.set_max_num_iterations_per_level( 500 );
		OFs.set_num_multires_levels( num_multires_levels );
		OFs.set_alpha(of_alpha);
		//OFs.set_beta(of_beta);
		//OFs.set_limit(0.01f);

		// Run registration
		//
		boost::shared_ptr< cuNDArray<float> > result = OFs.solve( &fixed_image, &moving_image );
        std::cout << "Dimensions of vector field ";
        auto dims = *result->get_dimensions();
        for (size_t d : dims) std::cout << d << " ";
        std::cout << std::endl;

		cuNDArray<float> dev_sub;
		dev_sub.create( &volume_dims_3d_3, result->get_data_ptr() );

		hoCuNDArray<float> host_sub( &volume_dims_3d_3, host_result_field->get_data_ptr()+i*num_elements_3d*3 );
		host_sub = *dev_sub.to_host();

	}

	/*
 {
  std::cout << std::endl << "Writing out registration results for phase " << phase << "." << std::endl;
  char filename[256];
  sprintf(&(filename[0]), "def_moving_%i.real", phase);
  write_nd_array<float>(&host_result_image, (char*)filename);
 }
	 */


	return host_result_field;
}
class hoCuConvertOperator : public subsetOperator<hoCuNDArray<float>>{

public:
	hoCuConvertOperator(boost::shared_ptr<subsetOperator<hoCuNDArray<float>>> _op) : subsetOperator<hoCuNDArray<float>>(_op->get_number_of_subsets()),op(_op){};


	virtual void mult_M(hoCuNDArray<float> *in, hoCuNDArray<float> *out,int subset, bool accumulate) override { op->mult_M((hoCuNDArray<float>*)in,(hoCuNDArray<float>*) out, subset,accumulate);}
	virtual void mult_MH(hoCuNDArray<float> *in, hoCuNDArray<float> *out,int subset, bool accumulate) override { op->mult_MH((hoCuNDArray<float>*)in,(hoCuNDArray<float>*) out, subset, accumulate);}
	virtual void mult_MH_M(hoCuNDArray<float> *in, hoCuNDArray<float> *out,int subset, bool accumulate) override { op->mult_MH_M((hoCuNDArray<float>*)in,(hoCuNDArray<float>*) out, subset, accumulate);}

	virtual boost::shared_ptr< std::vector<size_t> > get_codomain_dimensions(int i) { return op->get_codomain_dimensions(i);}
	virtual boost::shared_ptr< std::vector<size_t> > get_domain_dimensions() { return op->get_domain_dimensions();}
	virtual void set_domain_dimensions( std::vector<size_t> * dims) { return op->set_domain_dimensions(dims);}
	virtual void set_codomain_dimensions( std::vector<size_t> * dims) { return op->set_codomain_dimensions(dims);}



protected:
	boost::shared_ptr<subsetOperator<hoCuNDArray<float>>> op;
};

boost::shared_ptr<hoCuNDArray<float> > calculate_prior(boost::shared_ptr<CBCT_binning>  binning,boost::shared_ptr<CBCT_acquisition> ps, hoCuNDArray<float>& projections, std::vector<size_t> is_dims, floatd3 imageDimensions){
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
	int reg_iter;
	float rho;
	float tv_weight;
    float tv_weight4d;
    float tau;
	po::options_description desc("Allowed options");
	string tv_prior_filename;
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
    		("TV",po::value<float>(&tv_weight)->default_value(0),"Total variation weight")
            ("TV4D",po::value<float>(&tv_weight4d)->default_value(0),"Total variation in temporal direction")
                    ("tau",po::value<float>(&tau)->default_value(1e-5),"Solver tau")
					("reg_iter",po::value<int>(&reg_iter)->default_value(2),"Regularization iterations")
    		("use_prior","Use an FDK prior")
    		("TV-prior",po::value<string>(&tv_prior_filename)->default_value("reconstructionTV.real"),"TV prior for registration")
                    ("vector-field",po::value<string>(),"Stored vector field")
    		("3D","Only use binning data to determine wrong projections")
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

	// Define encoding matrix
	auto E = boost::make_shared<CBSubsetOperator<hoCuNDArray> >(subsets);

	//E->setup(ps,binning,imageDimensions);
	E->setup(ps,binning,imageDimensions);
	E->set_domain_dimensions(&is_dims);
	E->set_codomain_dimensions(ps->get_projections()->get_dimensions().get());

	auto E2 = boost::make_shared<hoCuConvertOperator>(E);

	//hoCuGPBBSolver<float> solver;
	//hoCuCgDescentSolver<float> solver;
//	osSPSSolver<hoCuNDArray<float>> solver;
	osMOMSolverD<hoCuNDArray<float>> solver;
	//osSPSSolver<hoCuNDArray<float>> solver;
	//hoCuNCGSolver<float> solver;
	solver.set_encoding_operator(E);
	//solver.set_domain_dimensions(&is_dims);
	solver.set_max_iterations(iterations);
	solver.set_output_mode(hoCuGPBBSolver<float>::OUTPUT_VERBOSE);
	solver.set_non_negativity_constraint(true);
    solver.set_tau(tau);
	solver.set_reg_steps(reg_iter);
	//solver.set_rho(rho);

	hoCuNDArray<float> projections = *ps->get_projections();
	E->offset_correct(&projections);

	boost::shared_ptr<hoCuNDArray<float> > prior;

	if (vm.count("use_prior")) {
		prior = calculate_prior(binning,ps,projections,is_dims,imageDimensions);
		solver.set_x0(prior);
	}
	/*
	if (tv_weight > 0){
		auto total_variation = boost::make_shared<hoCuTvOperator<float,4>>();
		total_variation->set_weight(tv_weight);
		//total_variation->set_weight_array(weight_array);
		solver.add_nonlinear_operator(total_variation);
		solver.set_kappa(tv_weight);
}
*/


    if (tv_weight4d > 0){
        boost::shared_ptr<hoNDArray<float>> displacements;
        if (vm.count("vector-field")){
            displacements = read_nd_array<float>(vm["vector-field"].as<string>().c_str());
            auto vdims = *displacements->get_dimensions();
            for (auto v : vdims) std::cout << v << " ";
            std::cout<<std::endl;
        } else {
            auto tv_recon = boost::make_shared<hoCuNDArray<float>>();
            *tv_recon = *read_nd_array<float>(tv_prior_filename.c_str());
            displacements = perform_registration(tv_recon, 0.01, 1, 3);
        }
        //clear(displacements.get());
        auto DtOF = boost::make_shared<hoCuOFPartialDerivativeOperator<float>>();
		//auto DtOF = boost::make_shared<hoCuPartialDerivativeOperator<float,4>>(3);
        DtOF->set_weight(tv_weight4d);
        DtOF->set_displacement_field(displacements);
        DtOF->set_domain_dimensions(&is_dims);
        DtOF->set_codomain_dimensions(&is_dims);
        solver.add_regularization_operator(DtOF);
    }

  if (tv_weight > 0){

  	auto Dx = boost::make_shared<hoCuPartialDerivativeOperator<float,4>>(0);
  	Dx->set_weight(tv_weight);
  	Dx->set_domain_dimensions(&is_dims);
  	Dx->set_codomain_dimensions(&is_dims);

  	auto Dy = boost::make_shared<hoCuPartialDerivativeOperator<float,4>>(1);
  	Dy->set_weight(tv_weight);
  	Dy->set_domain_dimensions(&is_dims);
  	Dy->set_codomain_dimensions(&is_dims);


  	auto Dz = boost::make_shared<hoCuPartialDerivativeOperator<float,4>>(2);
  	Dz->set_weight(tv_weight);
  	Dz->set_domain_dimensions(&is_dims);
  	Dz->set_codomain_dimensions(&is_dims);
	solver.add_regularization_group({Dx,Dy,Dz});
  }



	auto result = solver.solve(&projections);

	//write_nd_array<float>( result.get(), outputFile.c_str());
	saveNDArray2HDF5(result.get(),outputFile,imageDimensions,floatd3(0,0,0),"",iterations);
}

