#include "cuNDArray_math.h"
#include "cuPartialDerivativeOperator.h"
#include "osLALMSolver.h"
#include "cuSolverUtils.h"
#include <boost/program_options.hpp>
#include "hoNDArray_fileio.h"
#include "cuNlcgSolver.h"
#include "identityOperator.h"
using namespace Gadgetron;
typedef cuNDArray<float> ARRAY_TYPE;
typedef float REAL;
typedef std::vector<std::vector<boost::shared_ptr<linearOperator<cuNDArray<float >>>>> RG;

	void denoise(ARRAY_TYPE& x, ARRAY_TYPE& s, REAL scaling,REAL avg_lambda, int inner_iterations, RG& regularization_groups){
		std::cout << "scaling " << scaling << std::endl;
		REAL tau=1.0;
		REAL gam=0.35/(scaling*avg_lambda);
		REAL sigma = 1;
		REAL alpha = 0;
		ARRAY_TYPE g(x.get_dimensions());

		for (auto it = 0u; it < inner_iterations; it++){
			clear(&g);


			for (auto & reg_group : regularization_groups){
				std::vector<ARRAY_TYPE> datas(reg_group.size());
				REAL val = 0;
				for (auto i = 0u; i < reg_group.size(); i++){
					datas[i] = ARRAY_TYPE(reg_group[i]->get_codomain_dimensions());
					reg_group[i]->mult_M(&x,&datas[i]);
					datas[i] *= sigma*reg_group[i]->get_weight()/avg_lambda;
				}
				//updateFgroup is the resolvent operators on the group
				updateFgroup(datas,alpha,sigma);

				for (auto i = 0u; i < reg_group.size(); i++){
					datas[i] *= reg_group[i]->get_weight()/avg_lambda;
					reg_group[i]->mult_MH(&datas[i],&g,true);

				}

			}
			//updateG is the resolvent operator on the |x-s| part of the optimization
	axpy(-tau,&g,&x);
			g = s;

			axpy(tau/(scaling*avg_lambda),&g,&x);

			//g = precon;
			fill(&g,1.0f);

			reciprocal_inplace(&g);
			g *= tau/(scaling*avg_lambda);
			g += REAL(1);
			x /= g;
			//x *= 1/(1+tau/(scaling*avg_lambda));
			REAL theta = 1/std::sqrt(1+2*gam*tau);
			tau *= theta;
			sigma /= theta;
		}



	};

namespace po = boost::program_options;

int main(int argc, char** argv){
	std::string filename;
	std::string outputFile;
	int iterations;
	float tv_weight;
	po::options_description desc("Allowed options");

	desc.add_options()
    				("help", "produce help message")
    				("filename,f",po::value<std::string>(&filename)->default_value("lena.real"),"input filename")
    				("output,o", po::value<std::string>(&outputFile)->default_value("reconstruction.real"), "Output filename")
    				("TV",po::value<float>(&tv_weight)->default_value(0),"Total variation weight")
    				("iterations,i",po::value<int>(&iterations)->default_value(10),"Denoising iterations")
    				;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	auto img = cuNDArray<float>(*read_nd_array<float>(filename.c_str()));
	img *= 1e-5f;
	auto dims = *img.get_dimensions();
	auto Dx = boost::make_shared<cuPartialDerivativeOperator<float,2>>(0);
	Dx->set_weight(tv_weight);
	Dx->set_domain_dimensions(&dims);
	Dx->set_codomain_dimensions(&dims);
	auto Dy = boost::make_shared<cuPartialDerivativeOperator<float,2>>(1);
	Dy->set_weight(tv_weight);
	Dy->set_domain_dimensions(&dims);
	Dy->set_codomain_dimensions(&dims);

	std::vector<boost::shared_ptr<linearOperator<cuNDArray<float>>>> reg_group({Dx,Dy});

	cuNDArray<float> out(img);

	RG group({reg_group});

	denoise(out,img,1.0f,tv_weight,iterations,group);

	/*
	cuNlcgSolver<float> solver;
	auto id = boost::make_shared<identityOperator<cuNDArray<float>>>();
	id->set_codomain_dimensions(&dims);
	id->set_domain_dimensions(&dims);
	solver.set_encoding_operator(id);

	solver.add_regularization_group_operator(Dx);
	solver.add_regularization_group_operator(Dy);
	solver.add_group(1);

	solver.set_max_iterations(iterations);
	auto out = solver.solve(&img);
*/
	write_nd_array(out.to_host().get(),outputFile.c_str());

}
