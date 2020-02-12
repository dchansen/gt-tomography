#pragma once

#include "hoNDArray_math.h"
#include "cuNDArray_math.h"
namespace Gadgetron{
template<class ARRAY, class SOLVER> class ADMM{

public:
	ADMM(SOLVER* solver): solver_(solver){

	}

	boost::shared_ptr<ARRAY> solve(ARRAY* input){
		boost::shared_ptr<ARRAY> x;

		float mu = mean(weights_.get());

		std::vector<size_t> dims = *solver_->get_encoding_operator()->get_domain_dimensions();
		ARRAY u(*input);

		ARRAY Wy(*input);
		ARRAY nabla(input->get_dimensions());
		clear(&nabla);
		Wy *= *weights_;

		ARRAY D(*weights_);
		D += mu;


		for (int i = 0; i < iterations; i++ ){
			ARRAY tmp(u);
			tmp -= nabla;
			x = solver_->solve(&tmp);
			solver_->get_encoding_operator()->mult_M(x.get(),&tmp);
			u = tmp;
			u += nabla;
			u *= mu;
			u += Wy;
			u /= D;

			tmp -= u;

			std::cout << "Iteration " << i << " Res: " << nrm2(&tmp) << std::endl;
			nabla += tmp;




		}
		return x;



	}


	SOLVER* get_solver(){
		return solver_;
	}

	void set_weights(boost::shared_ptr<ARRAY> weights){weights_ = weights;}
	void set_iterations(int it ){ iterations = it;}
	SOLVER* solver_;
	boost::shared_ptr<ARRAY> weights_;
	int iterations;
};

}
