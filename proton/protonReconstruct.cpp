/*
 * protonReconstruct.cpp
 *
 *  Created on: Dec 22, 2011
 *      Author: u051747
 */
#include <iostream>
#include <boost/program_options.hpp>
#include "cuNDArray.h"
#include "cuCgSolver.h"
#include "cuImageOperator.h"
#include "splineBackprojectionOperator.h"
#include "cuPartialDerivativeOperator.h"
#include "cuLaplaceOperator.h"
#include "hoNDArray_fileio.h"
#include "check_CUDA.h"
#include "cuGpBbSolver.h"
#include "identityOperator.h"

#include "hoNDArray_math.h"

#include <sstream>
#include "hdf5_utils.h"

#include "encodingOperatorContainer.h"
#include "cuSARTSolver.h"
#include "cuMLSolver.h"
#include "cuNCGSolver.h"
#include "cuNlcgSolver.h"
#include "cuCgSolver.h"
#include "vector_td_io.h"
#include "protonPreconditioner.h"
#include "cuLbfgsSolver.h"
#include "solver_utils.h"

#include "cuTvOperator.h"

#include "circulantPreconditioner.h"

#include "ADMM.h"

using namespace std;
using namespace Gadgetron;

template <class T> class EXPORTGPUSOLVERS cuLbfgsSolver2 : public lbfgsSolver<cuNDArray<T> >
  {
  public:

    cuLbfgsSolver2() : lbfgsSolver<cuNDArray<T> >() {}
    virtual ~cuLbfgsSolver2() {}

    virtual void solver_non_negativity_filter(cuNDArray<T> *x,cuNDArray<T> *g){
    	::solver_non_negativity_filter(x,g);
    }
    virtual void iteration_callback(cuNDArray<T>* x ,int iteration,typename realType<T>::Type value){
  	  if (iteration == 0){
  		  std::ofstream textFile("residual.txt",std::ios::trunc);
  	  	  textFile << value << std::endl;
  	  } else{
  		  std::ofstream textFile("residual.txt",std::ios::app);
  		  textFile << value << std::endl;
  	  }


   	  std::stringstream ss;
   	  ss << "LBFGS-" << iteration << ".real";
   	  write_nd_array(x->to_host().get(),ss.str().c_str());

    };
  };



template <class T> class EXPORTGPUSOLVERS cuNlcgSolver2 : public cuNlcgSolver<T >
  {
  public:

    cuNlcgSolver2() : cuNlcgSolver<T >() {}
    virtual ~cuNlcgSolver2() {}

    virtual void iteration_callback(cuNDArray<T>* x ,int iteration,typename realType<T>::Type value, typename realType<T>::Type val2){
  	  if (iteration == 0){
  		  std::ofstream textFile("residual.txt",std::ios::trunc);
  	  	  textFile << value << std::endl;
  	  } else{
  		  std::ofstream textFile("residual.txt",std::ios::app);
  		  textFile << value << std::endl;
  	  }


   	  std::stringstream ss;
   	  ss << "NLCG-" << iteration << ".real";
   	  write_nd_array(x->to_host().get(),ss.str().c_str());

    };
  };

/*
boost::shared_ptr< cuNDArray<float> >  recursiveSolver(cuNDArray<float> * rhs,cuCGSolver<float, float> * cg,int depth){
	std::cout << "Recursion depth " << depth << " reached" << std::endl;
	if (depth == 0){
		std::cout << "Running solver for depth " << depth  << std::endl;
		return cg->solve(rhs);
	} else {
		boost::shared_ptr< cuNDArray<float> > rhs_temp = cuNDA_downsample<float,2>(rhs);

		boost::shared_ptr< cuNDArray<float> > guess = recursiveSolver(rhs_temp.get(),cg,depth-1);
		guess = cuNDA_upsample<float,2>(guess.get());
		std::cout << "Running solver for depth " << depth  << std::endl;
		return cg->solve(rhs,guess.get());

	}

}
 */

/*
template<class T> void notify(T val){
	std::cout << val <<std::endl;
}
 */

template<class T> void notify(std::string val){
	std::cout << val << std::endl;
}

namespace po = boost::program_options;
int main( int argc, char** argv)
{

	//
	// Parse command line
	//
	float background =  0.00106;


	std::string projectionsName;
	std::string splinesName;
	std::string outputFile;
	vector_td<int,3> dimensions;
	vector_td<float,3> physical_dims;
	vector_td<float,3> origin;
	int iterations;
	int device;
	bool precon,use_hull,use_weights;
	po::options_description desc("Allowed options");
	desc.add_options()
      								("help", "produce help message")
      								("projections,p", po::value<std::string>(&projectionsName)->default_value("projections.real"), "File containing the projection data")
      								("splines,s", po::value<std::string>(&splinesName)->default_value("splines.real"), "File containing the spline trajectories")
      								("data,D", po::value<std::string>(), "HDF5 file containing projections and splines. Used instead of the *.real files")
      								("dimensions,d", po::value<vector_td<int,3> >(&dimensions)->default_value(vector_td<int,3>(512,512,1)), "Pixel dimensions of the image")
      								("size,S", po::value<vector_td<float,3> >(&physical_dims)->default_value(vector_td<float,3>(20,20,5)), "Dimensions of the image in cm")
      								("center,c", po::value<vector_td<float,3> >(&origin)->default_value(vector_td<float,3>(0,0,0)), "Center of the reconstruction")
      								("iterations,i", po::value<int>(&iterations)->default_value(10), "Dimensions of the image")
      								("output,f", po::value<std::string>(&outputFile)->default_value("image.hdf5"), "Output filename")
      								("prior,P", po::value<std::string>(),"Prior image filename")
      								("prior-weight,k",po::value<float>(),"Weight of the prior image")
      								("TV",po::value<float>(),"Total variation weight")
      								("weights,w",po::value<std::string>(),"File containing the variance of data")
      								("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
      								("preconditioner",po::value<bool>(&precon)->default_value(false),"Use preconditioner")
      								("use_hull",po::value<bool>(&use_hull)->default_value(true),"Estimate convex hull of object")
      								("use_weights",po::value<bool>(&use_weights)->default_value(true),"Use weights if available. Always true if variance is set")
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
		else if (a.type() == typeid(float)) std::cout << it->second.as<float>();
		else if (a.type() == typeid(bool)) std::cout << it->second.as<bool>();
		else if (a.type() == typeid(vector_td<float,3>)) std::cout << it->second.as<vector_td<float,3> >();
		else if (a.type() == typeid(vector_td<int,3>)) std::cout << it->second.as<vector_td<int,3> >();
		else std::cout << "Unknown type" << std::endl;
		std::cout << std::endl;
	}

	cudaSetDevice(device);
	cudaDeviceReset();
	std::cout <<  std::endl;

	boost::shared_ptr<protonDataset<cuNDArray> > data;

	if (vm.count("data")){
		data = boost::shared_ptr<protonDataset<cuNDArray> >(new protonDataset<cuNDArray>(vm["data"].as<std::string>(),use_weights));

	} else {
		boost::shared_ptr<hoNDArray<vector_td<float,3> > > host_splines = read_nd_array< vector_td<float,3> >(splinesName.c_str());
		cout << "Number of spline elements: " << host_splines->get_number_of_elements() << endl;
		boost::shared_ptr< hoNDArray<float> > host_projections = read_nd_array<float >(projectionsName.c_str());
		//boost::shared_ptr<cuNDArray<float > > projections_old = projections;
		cout << "Number of elements " << host_projections->get_number_of_elements() << endl;
		cout << "Number of projection elements: " << host_projections->get_number_of_elements() << endl;
		if (vm.count("weights")){
			boost::shared_ptr< hoNDArray<float> > host_weights = read_nd_array<float >(vm["weights"].as<std::string>().c_str());
			if (host_weights->get_number_of_elements() != host_projections->get_number_of_elements())
				throw std::runtime_error("Number of elements in the variance vector does not match the number of projections ");
			reciprocal_inplace(host_weights.get());
			data = boost::shared_ptr<protonDataset<cuNDArray> >(new protonDataset<cuNDArray>(host_projections,host_splines,host_weights));

		} else data = boost::shared_ptr<protonDataset<cuNDArray> >(new protonDataset<cuNDArray>(host_projections,host_splines));

	}

	vector<size_t> ndims;
	ndims.push_back(3);




	//cuGpBbSolver<float> solver;
	cuNCGSolver<float> solver;
	//cuNlcgSolver2<float> solver;
	//cuCgSolver<float> solver;
	//cuLbfgsSolver2<float> solver;

	solver.set_max_iterations( iterations);
	solver.set_tc_tolerance((float)std::sqrt(1e-10));
	//solver.set_tc_tolerance((float)(1e-10));
	//solver.set_m(12);
	solver.set_output_mode( cuNCGSolver<float>::OUTPUT_VERBOSE );
	//if (!use_hull)
	//solver.set_non_negativity_constraint(true);





	vector<size_t> rhs_dims(&dimensions[0],&dimensions[3]); //Line to turn vector_td into std::vector.

	data->preprocess(rhs_dims,physical_dims,use_hull);

	if (data->get_weights())
		*data->get_projections() *= *data->get_weights(); //Have to scale projection data by weights before handing it to the solver. Write up the cost function and see why.

	if (use_hull) write_nd_array<float>(data->get_hull()->to_host().get(),"hull.real");



	boost::shared_ptr<cuNDArray<float> > rhs = data->get_projections();


	//E->set_codomain_dimensions(projections->get_dimensions().get());


	boost::shared_ptr< splineBackprojectionOperator<cuNDArray> > E(new splineBackprojectionOperator<cuNDArray>(data,physical_dims));
	E->set_domain_dimensions(&rhs_dims);


	boost::shared_ptr<encodingOperatorContainer<cuNDArray<float> > > enc (new encodingOperatorContainer<cuNDArray<float> >());


	if (precon){
		/*boost::shared_ptr<protonPreconditioner> P (new protonPreconditioner(rhs_dims));
		if (use_hull) P->set_hull(data->get_hull());*/
		std::cout << "Using preconditioner" << std::endl;
		boost::shared_ptr<circulantPreconditioner<cuNDArray,float> > P(new circulantPreconditioner<cuNDArray,float>(E));

		solver.set_preconditioner(P);
	}
	enc->add_operator(E);

	boost::shared_ptr<cuNDArray<float > > prior;
	if (vm.count("prior")){
		std::cout << "Prior image regularization in use" << std::endl;
		boost::shared_ptr<hoNDArray<float> > host_prior = read_nd_array<float >(vm["prior"].as<std::string>().c_str());

		host_prior->reshape(&rhs_dims);
		prior = boost::shared_ptr<cuNDArray<float> >(new cuNDArray<float>(host_prior.get()));
		float offset = float(0.01);
		//cuNDA_add(offset,prior.get());


		if (vm.count("prior-weight")){

			//boost::shared_ptr<cuImageOperator<float > > I (new cuImageOperator<float >());
			boost::shared_ptr<identityOperator<cuNDArray<float > > > I (new identityOperator<cuNDArray<float > >());
			//I->compute(prior.get());

			I->set_weight(vm["prior-weight"].as<float>());

			I->set_codomain_dimensions(&rhs_dims);
			I->set_domain_dimensions(&rhs_dims);
			cuNDArray<float> tmp = *prior;


			I->mult_M(prior.get(),&tmp);

			//cuNDA_scal(I->get_weight(),&tmp);
			std::vector<cuNDArray<float>* > proj;
			proj.push_back(rhs.get());
			proj.push_back(&tmp);
			enc->add_operator(I);
			rhs = enc->create_codomain(proj);

		} else {
			std::cout << "WARNING: Prior image set, but weight not specified" << std::endl;
		}
		solver.set_x0(prior);
	}

	if (vm.count("TV")){
			std::cout << "Total variation regularization in use" << std::endl;
			boost::shared_ptr<cuTvOperator<float,3> > tv(new cuTvOperator<float,3>);
			tv->set_weight(vm["TV"].as<float>());
			solver.add_nonlinear_operator(tv);

		}

	solver.set_encoding_operator(enc);
/*
	boost::shared_ptr< cuNDArray<float> > cgresult(new cuNDArray<float>(&rhs_dims));
	clear(cgresult.get());
	std::cout << "Groups: " << data->get_number_of_groups() << std::endl;
*/

	//E->mult_MH(data->get_projection_group(0).get(),cgresult.get());
	//E->mult_MH(rhs.get(),cgresult.get());
	//P->apply(cgresult.get(),cgresult.get());
	/*
	ADMM<cuNDArray<float>, cuCgSolver<float> > admm(&solver);
	admm.set_iterations(10);
	admm.set_weights(data->get_weights());
	E->set_use_weights(false);
	*/
	boost::shared_ptr< cuNDArray<float> > cgresult = solver.solve(rhs.get());
	//boost::shared_ptr< cuNDArray<float> > cgresult = admm.solve(rhs.get());

	//cuNDArray<float> tp = *projections;

	//E->mult_M(cgresult.get(),&tp);
	//axpy(-1.0f,projections.get(),&tp);
	//std::cout << "Total residual " << nrm2(&tp) << std::endl;

	std::cout << "GRadient norm: " << nrm2(cgresult.get()) << std::endl;
	boost::shared_ptr< hoNDArray<float> > host_result = cgresult->to_host();
	//write_nd_array<float>(host_result.get(), (char*)parms.get_parameter('f')->get_string_value());
	std::stringstream ss;
	for (int i = 0; i < argc; i++){
		ss << argv[i] << " ";
	}

	saveNDArray2HDF5<3>(host_result.get(),outputFile,physical_dims,origin,ss.str(), solver.get_max_iterations());

}



