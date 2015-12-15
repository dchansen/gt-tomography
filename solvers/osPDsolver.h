/*
 /**
 * Implements an OS version of "A first orer primal-dual algorithm for Convex Problems with Applications to Imaging" by Antonin Chambolle and Thomas Pock -  doi 0.1007/s10851-010-0251-1
 */
#pragma once
#include "subsetOperator.h"
#include "solver.h"
#include <numeric>
#include <vector>
#include <functional>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/make_shared.hpp>
#include <boost/math/constants/constants.hpp>
#include <initializer_list>
namespace Gadgetron{
template <class ARRAY_TYPE> class osPDSolver : public solver< ARRAY_TYPE,ARRAY_TYPE> {
	typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
	typedef typename realType<ELEMENT_TYPE>::Type REAL;
public:
	osPDSolver() :solver< ARRAY_TYPE,ARRAY_TYPE>() {
		_iterations=10;
		non_negativity_=false;
		alpha = 0;
		t = 1;
	}
	virtual ~osPDSolver(){};

	void set_max_iterations(int i){_iterations=i;}
	int get_max_iterations(){return _iterations;}
	void set_non_negativity_constraint(bool neg=true){non_negativity_=neg;}

	void set_damping(REAL damp){
		t = damp;
	}
	void set_alpha(REAL a){
		alpha = a;
	}
	/**
	 * Sets the preconditioning image. In most cases this is not needed, and the preconditioning is calculated based on the system transform
	 * @param precon_image
	 */
	void set_preconditioning_image(boost::shared_ptr<ARRAY_TYPE> precon_image){
		this->preconditioning_image_ = precon_image;
	}



	boost::shared_ptr<ARRAY_TYPE> solve(ARRAY_TYPE* in){
		//boost::shared_ptr<ARRAY_TYPE> rhs = compute_rhs(in);
		if( this->encoding_operator_.get() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : no encoding operator is set" );
			return boost::shared_ptr<ARRAY_TYPE>();
		}

		// Get image space dimensions from the encoding operator
		//

		boost::shared_ptr< std::vector<size_t> > image_dims = this->encoding_operator_->get_domain_dimensions();
		if( image_dims->size() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : encoding operator has not set domain dimension" );
			return boost::shared_ptr<ARRAY_TYPE>();
		}

		ARRAY_TYPE * x = new ARRAY_TYPE(*image_dims);
		if (this->x0_.get()){
			*x = *(this->x0_.get());
		} else  {
			clear(x);
		}
		std::vector<boost::shared_ptr<ARRAY_TYPE> > subsets = this->encoding_operator_->projection_subsets(in);

		ARRAY_TYPE tmp_projection(in->get_dimensions());
		clear(&tmp_projection);
		std::vector<boost::shared_ptr<ARRAY_TYPE> > tmp_projections = this->encoding_operator_->projection_subsets(&tmp_projection);

		boost::shared_ptr<ARRAY_TYPE> precon_image;
		if (preconditioning_image_)
			precon_image = preconditioning_image_;
		else {
			precon_image = boost::make_shared<ARRAY_TYPE>(image_dims.get());
			fill(precon_image.get(),ELEMENT_TYPE(1));
			this->encoding_operator_->mult_M(precon_image.get(),&tmp_projection,false);
			this->encoding_operator_->mult_MH(&tmp_projection,precon_image.get(),false);
			clamp_min(precon_image.get(),REAL(1e-6));

			reciprocal_inplace(precon_image.get());
			//ones_image *= (ELEMENT_TYPE) this->encoding_operator_->get_number_of_subsets();
		}


		ARRAY_TYPE g(image_dims);
		clear(&g);


		if( this->output_mode_ >= solver<ARRAY_TYPE,ARRAY_TYPE>::OUTPUT_VERBOSE ){
			std::cout << "osLALM setup done, starting iterations:" << std::endl;
		}

		REAL avg_lambda = calc_avg_lambda();
		std::vector<int> isubsets(boost::counting_iterator<int>(0), boost::counting_iterator<int>(this->encoding_operator_->get_number_of_subsets()));

		if (!this->x0_)
		{
			this->encoding_operator_->mult_M(x,tmp_projections[0].get(),0,false);
			*tmp_projections[0] -= *subsets[0];
			this->encoding_operator_->mult_MH(tmp_projections[0].get(),x,0,false);
			*x *= -REAL(this->encoding_operator_->get_number_of_subsets());
			*x *= *precon_image;
		}


		ARRAY_TYPE* xold = new ARRAY_TYPE(*x);
		REAL L = 8;
		REAL gam=0.35/(avg_lambda);
		REAL tau = L;
		REAL sigma = 1/(tau*L*L);

		for (int i =0; i < _iterations; i++){
			for (int isubset = 0; isubset < this->encoding_operator_->get_number_of_subsets(); isubset++){
				int subset = isubsets[isubset];


				if (regularization_groups.size() != 0 || regularization_operators.size() != 0)
					add_dual(*x,avg_lambda, tau,sigma);

				this->encoding_operator_->mult_M(x,tmp_projections[subset].get(),subset,false);
				*tmp_projections[subset] -= *subsets[subset];
				if( this->output_mode_ >= solver<ARRAY_TYPE,ARRAY_TYPE>::OUTPUT_VERBOSE ){
					std::cout << "Iteration " <<i << " Subset " << subset << " Update norm: " << nrm2(tmp_projections[subset].get()) << std::endl;
				}
				this->encoding_operator_->mult_MH(tmp_projections[subset].get(),&g,subset,false);

				g /= avg_lambda*2;
				if (regularization_groups.size() != 0 || regularization_operators.size() != 0){
					axpy(-1/(2*tau),x,&g);
				}

				g *= *precon_image;

				*x -= g;



				REAL theta = 1/std::sqrt(1+2*gam*tau);
				tau *= theta;
				sigma /= theta;
				std::cout << "Theta " << theta << " " << tau << " " << sigma << " " << gam << std::endl;
				*x *= (1+theta);
				axpy(-theta,xold,x);
				std::cout << mean(x) << std::endl;
				if (non_negativity_){
					clamp_min(x,REAL(0));
				}

				*xold = *x;

			}
		}
		delete xold;


		return boost::shared_ptr<ARRAY_TYPE>(x);
	}

	void set_encoding_operator(boost::shared_ptr<subsetOperator<ARRAY_TYPE> > encoding_operator){ encoding_operator_ = encoding_operator; }

	virtual void add_regularization_operator(boost::shared_ptr<linearOperator<ARRAY_TYPE>> op){
		regularization_operators.push_back(op);
	}

	virtual void add_regularization_group(std::initializer_list<boost::shared_ptr<linearOperator<ARRAY_TYPE>>> ops){
		regularization_groups.push_back(std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE>>>(ops));
	}

	virtual void add_regularization_group(std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE>>> ops){
		regularization_groups.push_back(ops);
	}


	virtual void set_tau(REAL tau){
		tau0 = tau;
	}


protected:

	/**
	 * Solves an image denoising problem, using the regularization operators for denoising.
	 * This is done via the AHMOD algorithm (see "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging" - Antonin Chambolle & Thomas Pock, 2010)
	 * @param x
	 * @param s
	 * @param scaling
	 */
	void add_dual(ARRAY_TYPE& x, REAL avg_lambda,REAL tau, REAL sigma ){

		ARRAY_TYPE g(x.get_dimensions());
		clear(&g);
		for (auto reg_op : regularization_operators){
			ARRAY_TYPE data(reg_op->get_codomain_dimensions());
			reg_op->mult_M(&x,&data);
			data *= sigma*reg_op->get_weight()/avg_lambda;
			//updateF is the resolvent operator on the regularization
			updateF(data, alpha, sigma);
			data *= reg_op->get_weight()/avg_lambda;
			reg_op->mult_MH(&data,&g,true);
		}

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

		axpy(-tau,&g,&x);

	};

	REAL calc_avg_lambda(){
		REAL result = 0;
		auto num = 0u;
		for (auto op : regularization_operators){
			auto w = op->get_weight();
			result += w;
			num++;
		}

		for (auto & group : regularization_groups)
			for (auto op : group){
				auto w = op->get_weight();
				std::cout << "Weight " << w << std::endl;
				result += w;
				num++;
			}
		if (result == 0)
			return 0.5;

		result /= num;

		return result;

	}
	int _iterations;
	bool non_negativity_;
	unsigned int reg_steps_;
	REAL alpha,t;
	REAL tau0;
	boost::shared_ptr<subsetOperator<ARRAY_TYPE> > encoding_operator_;
	boost::shared_ptr<ARRAY_TYPE> preconditioning_image_;
	std::vector<std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE>>>> regularization_groups;
	std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE> >> regularization_operators;

};
}