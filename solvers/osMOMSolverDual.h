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
#include <tuple>

namespace Gadgetron{
template <class ARRAY_TYPE> class osMOMSolverDual : public solver< ARRAY_TYPE,ARRAY_TYPE> {
	typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
	typedef typename realType<ELEMENT_TYPE>::Type REAL;
public:
	osMOMSolverDual() :solver< ARRAY_TYPE,ARRAY_TYPE>() {
		_iterations=10;
		_beta = REAL(1);
		_alpha = 0.2;
		_gamma = 0;
		non_negativity_=false;
		reg_steps_=1;
		_kappa = REAL(1);
		tau0=1e-3;
		denoise_alpha=0;
		dump=false;
		splits=2;
	}
	virtual ~osMOMSolverDual(){};

	void set_max_iterations(int i){_iterations=i;}
	int get_max_iterations(){return _iterations;}
	void set_non_negativity_constraint(bool neg=true){non_negativity_=neg;}
	void set_splits(unsigned int splits_ ){splits = splits_;}
	/**
	 * @brief Sets the weight of each step in the SART iteration
	 * @param beta
	 */
	void set_beta(REAL beta){_beta = beta;}
	void set_gamma(REAL gamma){_gamma = gamma;}
	void set_kappa(REAL kappa){_kappa = kappa;}

	void set_dump(bool d){dump = d;}
	virtual void add_regularization_operator(boost::shared_ptr<linearOperator<ARRAY_TYPE>> op, boost::shared_ptr<ARRAY_TYPE> prior = boost::shared_ptr<ARRAY_TYPE>()){
		regularization_operators.push_back(std::make_tuple(op,prior));
	}

	virtual void add_regularization_group(std::initializer_list<boost::shared_ptr<linearOperator<ARRAY_TYPE>>> ops, boost::shared_ptr<ARRAY_TYPE> prior = boost::shared_ptr<ARRAY_TYPE>()){
		regularization_groups.push_back(std::make_tuple(std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE>>>(ops),prior));
	}

	virtual void add_regularization_group(std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE>>> ops, boost::shared_ptr<ARRAY_TYPE> prior = boost::shared_ptr<ARRAY_TYPE>()){
		regularization_groups.push_back(std::make_tuple(ops,prior));
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
			return boost::shared_ptr<ARRAY_TYPE>();
		}

		// Get image space dimensions from the encoding operator
		//

		boost::shared_ptr< std::vector<size_t> > image_dims = this->encoding_operator_->get_domain_dimensions();
		if( image_dims->size() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : encoding operator has not set domain dimension" );
			return boost::shared_ptr<ARRAY_TYPE>();
		}

		auto split_dims = *image_dims;
		split_dims.push_back(splits);

		ARRAY_TYPE * z = new ARRAY_TYPE(split_dims);
		if (this->x0_.get()){
			*z = *(this->x0_.get());
		} else  {
			clear(z);
		}

		ARRAY_TYPE * x = new ARRAY_TYPE(*z);
		ARRAY_TYPE * xold = new ARRAY_TYPE(*z);

		auto B = boost::make_shared<subsetAccumulateOperator<ARRAY_TYPE>>(this->encoding_operator_);
		B->set_domain_dimensions(&split_dims);
		B->set_codomain_dimensions(in->get_dimensions().get());

		std::vector<boost::shared_ptr<ARRAY_TYPE> > subsets = this->encoding_operator_->projection_subsets(in);

		ARRAY_TYPE tmp_projection(in->get_dimensions());
		std::vector<boost::shared_ptr<ARRAY_TYPE> > tmp_projections = this->encoding_operator_->projection_subsets(&tmp_projection);

		boost::shared_ptr<ARRAY_TYPE> precon_image;
		if (preconditioning_image_)
			precon_image = preconditioning_image_;
		else {
			precon_image = boost::make_shared<ARRAY_TYPE>(split_dims);
			fill(precon_image.get(),ELEMENT_TYPE(1));
			B->mult_M(precon_image.get(),&tmp_projection,false);
			B->mult_MH(&tmp_projection,precon_image.get(),false);
			*precon_image += _beta;
			clamp_min(precon_image.get(),ELEMENT_TYPE(1e-6));
			reciprocal_inplace(precon_image.get());
			//ones_image *= (ELEMENT_TYPE) B->get_number_of_subsets();
		}
		ARRAY_TYPE tmp_image(split_dims);

		ARRAY_TYPE f(split_dims);
		ARRAY_TYPE d(split_dims);
		clear(&f);
		clear(&d);
		REAL avg_lambda = calc_avg_lambda();
		REAL t = 1;
		REAL told = 1;
		if( this->output_mode_ >= solver<ARRAY_TYPE,ARRAY_TYPE>::OUTPUT_VERBOSE ){
			std::cout << "osMOM setup done, starting iterations:" << std::endl;
		}

		std::vector<int> isubsets(boost::counting_iterator<int>(0), boost::counting_iterator<int>(B->get_number_of_subsets()));
		REAL kappa_int = _kappa;
		REAL step_size;
		for (int i =0; i < _iterations; i++){
			for (int isubset = 0; isubset < B->get_number_of_subsets(); isubset++){

				t = 0.5*(1+std::sqrt(1+4*t*t));
				int subset = isubsets[isubset];
				B->mult_M(x,tmp_projections[subset].get(),subset,false);
				*tmp_projections[subset] -= *subsets[subset];
				if( this->output_mode_ >= solver<ARRAY_TYPE,ARRAY_TYPE>::OUTPUT_VERBOSE ){
					std::cout << "Iteration " <<i << " Subset " << subset << " Update norm: " << nrm2(tmp_projections[subset].get()) << std::endl;
				}

				B->mult_MH(tmp_projections[subset].get(),&tmp_image,subset,false);
				tmp_image *= -REAL(B->get_number_of_subsets());
				axpy(+_beta,&d,&tmp_image);


				tmp_image *= *precon_image;

				*z += tmp_image;


				{

					ARRAY_TYPE s(z);
					s -= d;
					*x = *z;
					denoise(*x,s,*precon_image,1.0,avg_lambda);
				}
				//axpy(REAL(_beta),&tmp_image,x);
				if (non_negativity_){

					auto images = split_image_array(x);
					clamp_min(images[0].get(),ELEMENT_TYPE(0));
				}
				d += *x;
				d -= *z;

				/*
				for (auto op : regularization_operators){

					op->gradient(x,&tmp_image);
					tmp_image /= nrm2(&tmp_image);
					auto reg_val = op->magnitude(x);
					std::cout << "Reg val: " << reg_val << std::endl;
					ARRAY_TYPE y = *x;
					axpy(-kappa_int,&tmp_image,&y);


					while(op->magnitude(&y) > reg_val){

						kappa_int /= 2;
						axpy(kappa_int,&tmp_image,&y);
						std::cout << "Kappa: " << kappa_int << std::endl;
					}
					reg_val = op->magnitude(&y);
					*x = y;

				}
			*/

				*z = *x;

				*z *= 1+(told-1)/t;
				axpy(-(told-1)/t,xold,z);
				std::swap(x,xold);
				*x = *z;
				told = t;

				//step_size *= 0.99;

			}
			//std::reverse(isubsets.begin(),isubsets.end());
			//std::random_shuffle(isubsets.begin(),isubsets.end());
			/*
			ARRAY_TYPE tmp_proj(*in);
			clear(&tmp_proj);
			B->mult_M(x,&tmp_proj,false);
			tmp_proj -= *in;
			 */

			if (dump){
				std::stringstream ss;
				ss << "osMOM-" << i << ".real";

				write_nd_array<ELEMENT_TYPE>(x,ss.str().c_str());
			}
			/*
			//calc_regMultM(x,regEnc);
			//REAL f = functionValue(&tmp_proj,regEnc,x);
			std::cout << "Function value: " << dot(&tmp_proj,&tmp_proj) << std::endl;
			 */
		}
		delete x,xold;


		return boost::shared_ptr<ARRAY_TYPE>(z);
	}

	void set_encoding_operator(boost::shared_ptr<subsetOperator<ARRAY_TYPE> > encoding_operator){ encoding_operator_ = encoding_operator; }


protected:

	std::vector<boost::shared_ptr<ARRAY_TYPE>> split_image_array(ARRAY_TYPE* image){
		std::vector<boost::shared_ptr<ARRAY_TYPE>> result;
		auto dims = *image->get_dimensions();
		dims.pop_back();
		auto ptr = image->get_data_ptr();
		for (int  i = 0; i < splits; i++){
			auto tmp = boost::make_shared<ARRAY_TYPE>(dims,ptr);
			ptr += tmp->get_number_of_elements();
			result.push_back(tmp);
		}
		return result;
	}

	REAL calc_avg_lambda(){
		REAL result = 0;
		auto num = 0u;
		for (auto op_pair : regularization_operators){
			auto op = std::get<0>(op_pair);
			auto w = op->get_weight();
			result += w;
			num++;
		}

		for (auto & group_pair : regularization_groups){
			auto group = std::get<0>(group_pair);
			for (auto op : group){
				auto w = op->get_weight();
				result += w;
				num++;
			}
		}

		result /= num;

		return result;

	}
	void denoise(ARRAY_TYPE& x, ARRAY_TYPE& s, ARRAY_TYPE& precon,REAL scaling,REAL avg_lambda ){
		REAL gam=0.35/(scaling*avg_lambda)/(precon.get_number_of_elements()/asum(&precon));
		REAL L = 4; //Hmm.. this seems a little..well... guessy?
		REAL tau = tau0;
		REAL sigma = 1/(tau*L*L);
		ARRAY_TYPE g(x.get_dimensions());
		if (regularization_groups.empty() && regularization_operators.empty()){
			x = s;
			return;
		}

		for (auto it = 0u; it < reg_steps_; it++){
			clear(&g);
			for (auto reg_op_pair : regularization_operators){
				auto reg_op = std::get<0>(reg_op_pair);
				auto prior = std::get<1>(reg_op_pair);
				ARRAY_TYPE data(reg_op->get_codomain_dimensions());
				if (prior) x -= *prior;
				reg_op->mult_M(&x,&data);
				if (prior) x += *prior;
				data *= sigma*reg_op->get_weight()/avg_lambda;
				//updateF is the resolvent operator on the regularization
				updateF(data, denoise_alpha, sigma);
				data *= reg_op->get_weight()/avg_lambda;
				std::cout << "Reg val: " << asum(&data) << std::endl;
				reg_op->mult_MH(&data,&g,true);
			}

			for (auto & reg_group_pair : regularization_groups){
				auto reg_group = std::get<0>(reg_group_pair);
				auto prior = std::get<1>(reg_group_pair);
				std::vector<ARRAY_TYPE> datas(reg_group.size());
				REAL val = 0;
				REAL reg_val = 0;
				if (prior) x -= *prior;
				for (auto i = 0u; i < reg_group.size(); i++){
					datas[i] = ARRAY_TYPE(reg_group[i]->get_codomain_dimensions());

					reg_group[i]->mult_M(&x,&datas[i]);

					auto tmp_dims = *datas[i].get_dimensions();
					reg_val += asum(&datas[i])*reg_group[i]->get_weight();
					datas[i] *= sigma*reg_group[i]->get_weight()/avg_lambda;
				}
				if (prior) x += *prior;

				std::cout << "Reg val: " << reg_val << " Scaling " << scaling*avg_lambda  << std::endl;
				//updateFgroup is the resolvent operators on the group
				updateFgroup(datas,denoise_alpha,sigma);

				for (auto i = 0u; i < reg_group.size(); i++){
					datas[i] *= reg_group[i]->get_weight()/avg_lambda;
					reg_group[i]->mult_MH(&datas[i],&g,true);

				}

			}
			//updateG is the resolvent operator on the |x-s| part of the optimization
			axpy(-tau*_beta,&g,&x);
			g = s;
			g /= precon;

			axpy(tau*_beta/(scaling*avg_lambda),&g,&x);

			g = precon;

			reciprocal_inplace(&g);
			g *= tau*_beta/(scaling*avg_lambda);
			g += REAL(1);
			x /= g;
			//x *= 1/(1+tau/(scaling*avg_lambda));
			REAL theta = 1/std::sqrt(1+2*gam*tau);
			tau *= theta;
			sigma /= theta;

		}
	}

	std::vector<std::tuple< std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE>>>,boost::shared_ptr<ARRAY_TYPE> >> regularization_groups;

	std::vector<std::tuple<boost::shared_ptr<linearOperator<ARRAY_TYPE> >, boost::shared_ptr<ARRAY_TYPE>>> regularization_operators;

	int _iterations;
	REAL _beta, _gamma, _alpha, _kappa,tau0, denoise_alpha;
	bool non_negativity_, dump;
	unsigned int reg_steps_, splits;
	boost::shared_ptr<subsetOperator<ARRAY_TYPE> > encoding_operator_;
	boost::shared_ptr<ARRAY_TYPE> preconditioning_image_;

};
}
