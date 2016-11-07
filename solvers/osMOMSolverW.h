/*
 * osMOMSolver.h
 *
 *  Created on: Mar 23, 2015
 *      Author: u051747
 */
//Based on Donghwan Kim; Ramani, S.; Fessler, J.A., "Combining Ordered Subsets and Momentum for Accelerated X-Ray CT Image Reconstruction,"
#pragma once
#include "subsetOperator.h"
#include "solver.h"
#include <numeric>
#include <vector>
#include <functional>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/make_shared.hpp>
#include "primalDualOperator.h"

namespace Gadgetron{
template <class ARRAY_TYPE> class osMOMSolverW : public solver< ARRAY_TYPE,ARRAY_TYPE> {
	typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
	typedef typename realType<ELEMENT_TYPE>::Type REAL;
public:
	osMOMSolverW() :solver< ARRAY_TYPE,ARRAY_TYPE>() {
		_iterations=10;
		_beta = REAL(1);
		_alpha = 0.2;
		_gamma = 0;
		non_negativity_=false;
		reg_steps_=1;
		_kappa = REAL(1);
		tau0=1e-3;
		denoise_alpha=0;
		dump=true;
	}
	virtual ~osMOMSolverW(){};

	void set_max_iterations(int i){_iterations=i;}
	int get_max_iterations(){return _iterations;}
	void set_non_negativity_constraint(bool neg=true){non_negativity_=neg;}
	/**
	 * @brief Sets the weight of each step in the SART iteration
	 * @param beta
	 */
	void set_beta(REAL beta){_beta = beta;}
	void set_gamma(REAL gamma){_gamma = gamma;}
	void set_kappa(REAL kappa){_kappa = kappa;}

	void set_dump(bool d){dump = d;}
	virtual void add_regularization_operator(boost::shared_ptr<linearOperator<ARRAY_TYPE>> op, REAL alpha = REAL(0)){
		regularization_operators.push_back(boost::make_shared<linearPrimalDualOperator>(op,alpha));
	}

	virtual void add_regularization_operator(boost::shared_ptr<primalDualOperator<ARRAY_TYPE>> op){
		regularization_operators.push_back(op);
	}

	virtual void add_regularization_group(std::initializer_list<boost::shared_ptr<linearOperator<ARRAY_TYPE>>> ops,REAL alpha = REAL(0)){
		regularization_operators.push_back(boost::make_shared<linearGroupPrimalDualOperator>(ops,alpha));
	}



	virtual void set_tau(REAL tau){
		tau0 = tau;
	}

	virtual void set_huber(REAL hub){
		denoise_alpha=hub;
	}

	/**
	 * Sets the preconditioning image. In most cases this is not needed, and the preconditioning is calculated based on the system transform
	 * @param precon_image
	 */
	void set_preconditioning_image(boost::shared_ptr<ARRAY_TYPE> precon_image){
		this->preconditioning_image_ = precon_image;
	}


	void set_reg_steps(unsigned int reg_steps){ reg_steps_ = reg_steps;}

	boost::shared_ptr<ARRAY_TYPE> solve(ARRAY_TYPE* in){
		//boost::shared_ptr<ARRAY_TYPE> rhs = compute_rhs(in);
		if( this->encoding_operator_.get() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : no encoding operator is set" );
		}

		// Get image space dimensions from the encoding operator
		//

		boost::shared_ptr< std::vector<size_t> > image_dims = this->encoding_operator_->get_domain_dimensions();
		if( image_dims->size() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : encoding operator has not set domain dimension" );
		}

		ARRAY_TYPE * z = new ARRAY_TYPE(*image_dims);
		if (this->x0_.get()){
			*z = *(this->x0_.get());
		} else  {
			clear(z);
		}

		ARRAY_TYPE * x = new ARRAY_TYPE(*z);
		ARRAY_TYPE * xold = new ARRAY_TYPE(*z);

		std::vector<boost::shared_ptr<ARRAY_TYPE> > subsets = this->encoding_operator_->projection_subsets(in);

		ARRAY_TYPE tmp_projection(in->get_dimensions());
		std::vector<boost::shared_ptr<ARRAY_TYPE> > tmp_projections = this->encoding_operator_->projection_subsets(&tmp_projection);

		boost::shared_ptr<ARRAY_TYPE> precon_image;
		if (preconditioning_image_)
			precon_image = preconditioning_image_;
		else {
			precon_image = boost::make_shared<ARRAY_TYPE>(image_dims.get());
			fill(precon_image.get(),ELEMENT_TYPE(1));
			this->encoding_operator_->mult_M(precon_image.get(),&tmp_projection,false);
			this->encoding_operator_->mult_MH(&tmp_projection,precon_image.get(),false);
			clamp_min(precon_image.get(),ELEMENT_TYPE(1e-6));
			reciprocal_inplace(precon_image.get());
			//ones_image *= (ELEMENT_TYPE) this->encoding_operator_->get_number_of_subsets();
		}


		REAL avg_lambda = 0;
		for (auto op : regularization_operators) avg_lambda += op->get_weight()/regularization_operators.size();
		for (auto op : regularization_operators) op->set_weight(op->get_weight()/avg_lambda);



		if( this->output_mode_ >= solver<ARRAY_TYPE,ARRAY_TYPE>::OUTPUT_VERBOSE ){
			std::cout << "osMOM setup done, starting iterations:" << std::endl;
		}


		REAL t = 1;
		REAL told = 1;
		std::vector<int> isubsets(boost::counting_iterator<int>(0), boost::counting_iterator<int>(this->encoding_operator_->get_number_of_subsets()));
		REAL kappa_int = _kappa;
		REAL step_size;
		for (int i =0; i < _iterations; i++){

			//clear(xold);
			update_weights(z);

			for (int isubset = 0; isubset < this->encoding_operator_->get_number_of_subsets(); isubset++){

				t = 0.5*(1+std::sqrt(1+4*t*t));
				int subset = isubsets[isubset];
				this->encoding_operator_->mult_M(x,tmp_projections[subset].get(),subset,false);
				*tmp_projections[subset] -= *subsets[subset];

				if( this->output_mode_ >= solver<ARRAY_TYPE,ARRAY_TYPE>::OUTPUT_VERBOSE ){
					std::cout << "Iteration " <<i << " Subset " << subset << " Update norm: " << nrm2(tmp_projections[subset].get()) << std::endl;
				}

				{

					ARRAY_TYPE tmp_image(image_dims.get());

					this->encoding_operator_->mult_MH(tmp_projections[subset].get(),&tmp_image,subset,false);
					tmp_image *= *precon_image;
					axpy(-REAL(_beta/(1+_gamma*i))*this->encoding_operator_->get_number_of_subsets(),&tmp_image,z);

				}


				denoise(*x,*z,*precon_image,avg_lambda);
				//*x = *z;

				//axpy(REAL(_beta),&tmp_image,x);
				if (non_negativity_){
					clamp_min(x,ELEMENT_TYPE(0));
				}


				*z = *x;
				*z *= 1+(told-1)/t;
				axpy(-(told-1)/t,xold,z);
				std::swap(x,xold);
				*x = *z;

				told = t;

				//step_size *= 0.99;

			}

			if (dump){
				std::stringstream ss;
				ss << "osMOM-" << i << ".real";

				write_nd_array<ELEMENT_TYPE>(x,ss.str().c_str());
			}

		}
		delete x,xold;
		for (auto op : regularization_operators) op->set_weight(op->get_weight()*avg_lambda);

		return boost::shared_ptr<ARRAY_TYPE>(z);
	}

	void set_encoding_operator(boost::shared_ptr<subsetOperator<ARRAY_TYPE> > encoding_operator){ encoding_operator_ = encoding_operator; }


protected:



	void denoise(ARRAY_TYPE& x, ARRAY_TYPE& s, ARRAY_TYPE& precon,REAL avg_lambda ){
		REAL gam=0.35/(avg_lambda)/(precon.get_number_of_elements()/asum(&precon));
		REAL L = 4; //Hmm.. this seems a little..well... guessy?
		REAL tau = tau0;
		REAL sigma = 1/(tau*L*L);
		ARRAY_TYPE g(x.get_dimensions());
		if (regularization_operators.empty()){
			x = s;
			return;
		}


		for (auto it = 0u; it < reg_steps_; it++){
			clear(&g);
			for (auto reg_op : regularization_operators){
					reg_op->primalDual(&x,&g,sigma,true);
			}
			//updateG is the resolvent operator on the |x-s| part of the optimization
			this->updateG(g,x,s,precon,tau,avg_lambda);
			//x *= 1/(1+tau/(scaling*avg_lambda));
			REAL theta = 1/std::sqrt(1+2*gam*tau);
			tau *= theta;
			sigma /= theta;

		}
	}


	virtual void updateG(ARRAY_TYPE& g, ARRAY_TYPE& x,ARRAY_TYPE& s, ARRAY_TYPE& precon, REAL tau, REAL avg_lambda )
	{
		axpy(-tau,&g,&x);
		g = s;
		g /= precon;

		axpy(tau/(avg_lambda),&g,&x);

		g = precon;

		reciprocal_inplace(&g);
		g *= tau/(avg_lambda);
		g += REAL(1);
		x /= g;
	}
	std::vector<boost::shared_ptr<primalDualOperator<ARRAY_TYPE>>> regularization_operators;


	int _iterations;
	REAL _beta, _gamma, _alpha, _kappa,tau0, denoise_alpha;
	bool non_negativity_, dump;
	unsigned int reg_steps_;
	boost::shared_ptr<subsetOperator<ARRAY_TYPE> > encoding_operator_;
	boost::shared_ptr<ARRAY_TYPE> preconditioning_image_;


private:
	void update_weights(ARRAY_TYPE* x){
		for (auto op : regularization_operators)
			op->update_weights(x);
	}

	class linearPrimalDualOperator : public primalDualOperator<ARRAY_TYPE>{

	public:

		linearPrimalDualOperator(boost::shared_ptr<linearOperator<ARRAY_TYPE>> in_op, REAL alpha = 0) :
		op(in_op), denoise_alpha(alpha){}

		virtual void primalDual(ARRAY_TYPE* in, ARRAY_TYPE* out,REAL sigma, bool accumulate){
			auto data = ARRAY_TYPE(op->get_codomain_dimensions());
			op->mult_M(in,&data,false);
			data *= *this->weight_arr;
			data *= sigma*op->get_weight();
			updateF(data,denoise_alpha,sigma);
			data *= *this->weight_arr;
			op->mult_MH(&data,out,accumulate);

		}

		virtual REAL get_weight() override {return op->get_weight();}

		virtual void set_weight(REAL weight_) override { op->set_weight(weight_);}

		virtual void update_weights(ARRAY_TYPE* x) override {
			this->weight_arr = boost::make_shared<ARRAY_TYPE>(op->get_codomain_dimensions());

			op->mult_M(x,this->weight_arr.get());
			abs_inplace(this->weight_arr.get());
			*this->weight_arr += REAL(1);
			reciprocal_inplace(this->weight_arr.get());

		} ;

	private:
		boost::shared_ptr<linearOperator<ARRAY_TYPE>> op;
		REAL denoise_alpha;

	};

	class linearGroupPrimalDualOperator : public primalDualOperator<ARRAY_TYPE>{

	public:

		linearGroupPrimalDualOperator(std::initializer_list<boost::shared_ptr<linearOperator<ARRAY_TYPE>>> in_ops, REAL alpha = 0) :
				ops(in_ops), denoise_alpha(alpha){}

		virtual void primalDual(ARRAY_TYPE* in, ARRAY_TYPE* out,REAL sigma, bool accumulate){
			if (!accumulate) clear(out);
			std::vector<ARRAY_TYPE> datas(ops.size());
			REAL reg_val = 0;
			for (auto i = 0u; i < ops.size(); i++){
				datas[i] = ARRAY_TYPE(ops[i]->get_codomain_dimensions());
				ops[i]->mult_M(in,&datas[i]);
				reg_val += asum(&datas[i])*ops[i]->get_weight();
				datas[i] *= sigma*ops[i]->get_weight();

			}

			std::cout << "Reg val: " << reg_val << std::endl;
			updateFgroup(datas,denoise_alpha,sigma);
			for (auto i = 0u; i < ops.size(); i++){
				datas[i] *= ops[i]->get_weight();
				ops[i]->mult_MH(&datas[i],out,true);

			}


		}

		virtual REAL get_weight() override {
			REAL res = 0;
			for (auto op : ops) res += op->get_weight();
			return res;
		}

		virtual void set_weight(REAL weight_) override { for (auto op : ops ) op->set_weight(weight_);}


	private:
		std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE>>> ops;
		REAL denoise_alpha;

	};
};
}
