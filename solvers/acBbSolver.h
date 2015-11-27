#pragma once

#include "gpSolver.h"
#include "linearOperatorSolver.h"
#include "real_utilities.h"
#include "complext.h"
#include "cgPreconditioner.h"

#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace Gadgetron{
/** Nonlinear conjugate gradient solver.
 * Adapted from Y.H. Dai & Y. Yuan 2001 "An Efficient Hybrid Conjugate Gradient Method for Unconstrained Optimization"
 * Annals of Operations Research, March 2001, Volume 103, Issue 1-4, pp 33-47
 *
 */

template <class ARRAY_TYPE> class acBbSolver : public gpSolver<ARRAY_TYPE>
{


protected:
	typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
	typedef typename realType<ELEMENT_TYPE>::Type REAL;
	typedef ARRAY_TYPE ARRAY_CLASS;
	typedef gpSolver<ARRAY_TYPE> GP;
	typedef typename gpSolver<ARRAY_TYPE>::l1GPRegularizationOperator l1GPRegularizationOperator;

public:

	acBbSolver(): gpSolver<ARRAY_TYPE>() {
		iterations_ = 10;
		tc_tolerance_ = (REAL)1e-7;
		non_negativity_constraint_=false;
		dump_residual = false;
		threshold= REAL(1e-7);
		barrier_threshold=1e4;
		rho = 0.5f;
	}

	virtual ~acBbSolver(){}


	virtual void set_rho(REAL _rho){
		rho = _rho;
	}

	virtual boost::shared_ptr<ARRAY_TYPE> solve(ARRAY_TYPE* in)
																															{
		if( this->encoding_operator_.get() == 0 ){
			throw std::runtime_error("Error: acBbSolver::compute_rhs : no encoding operator is set" );
		}

		// Get image space dimensions from the encoding operator
		//

		boost::shared_ptr< std::vector<size_t> > image_dims = this->encoding_operator_->get_domain_dimensions();
		if( image_dims->size() == 0 ){
			throw std::runtime_error("Error: acBbSolver::compute_rhs : encoding operator has not set domain dimension" );
		}

		ARRAY_TYPE * x = new ARRAY_TYPE;
		x->create(image_dims.get());



		ARRAY_TYPE * g = new ARRAY_TYPE;
		g->create(image_dims.get());
		ARRAY_TYPE y (image_dims.get());

		if (this->x0_.get()){
			*x = *(this->x0_.get());
		} else  {
			clear(x);
		}

		//REAL delta = REAL(0.01);
		//REAL sigma = REAL(0.4);

		std::vector<ARRAY_TYPE> regEnc;

		for (int i = 0; i < this->regularization_operators_.size(); i++){
			regEnc.push_back(ARRAY_TYPE(this->regularization_operators_[i]->get_codomain_dimensions()));
			if (reg_priors[i].get()){
				regEnc.back() = *reg_priors[i];
				regEnc.back() *= -std::sqrt(this->regularization_operators_[i]->get_weight());
			}

		}
		std::vector<ARRAY_TYPE> regEnc2 = regEnc;

		ARRAY_TYPE d(image_dims.get());
		clear(&d);
		ARRAY_TYPE encoding_space(in->get_dimensions().get());

		//ARRAY_TYPE gtmp(image_dims.get());
		ARRAY_TYPE encoding_space2(in->get_dimensions().get());
		REAL reg_res,data_res;
		//ARRAY_TYPE x2(x);
		ARRAY_TYPE s(x);

		if( this->output_mode_ >= solver<ARRAY_TYPE,ARRAY_TYPE>::OUTPUT_VERBOSE ){
			std::cout << "Iterating..." << std::endl;
		}
		REAL grad_norm0;
		REAL alpha_cur=0;

		std::vector<REAL> old_norms;

		ELEMENT_TYPE ys;
		REAL ss, yy;

		for (int i = 0; i < iterations_; i++){
			if (i==0){
				if (this->x0_.get()){
					this->encoding_operator_->mult_M(x,&encoding_space);

				} else clear(&encoding_space);
				encoding_space -= *in;
				this->encoding_operator_->mult_MH(&encoding_space,g);

				*g *=  this->encoding_operator_->get_weight();
				data_res = std::sqrt(this->encoding_operator_->get_weight())*real(dot(&encoding_space,&encoding_space));

				calc_regMultM(x,regEnc);
				for (int n = 0; n < regEnc.size(); n++)
					if (reg_priors[n].get())
						axpy(-std::sqrt(this->regularization_operators_[n]->get_weight()),reg_priors[n].get(),&regEnc[n]);
				this->add_gradient(x,g);
				add_linear_gradient(regEnc,g);
				reg_res=REAL(0);
				if (this->precond_.get()){
					this->precond_->apply(g,g);
					this->precond_->apply(g,g);
				}

			}else {
				data_res = real(dot(&encoding_space,&encoding_space));
			}



			if (non_negativity_constraint_) solver_non_negativity_filter(x,g);
			if (i==0) grad_norm0=nrm2(g);
			REAL grad_norm = nrm2(g);
			if( this->output_mode_ >= solver<ARRAY_TYPE,ARRAY_TYPE>::OUTPUT_VERBOSE ){

				std::cout << "Iteration " <<i << ". Realtive gradient norm: " <<  grad_norm/grad_norm0 << std::endl;
			}



			clear(d);
			d -= *g;
			this->encoding_operator_->mult_M(&d,&encoding_space2);
			calc_regMultM(&d,regEnc2);




			if (alpha_cur == 0){
				if (i == 0)
					alpha_cur = -real(dot(&encoding_space,&encoding_space2)+calc_dot(regEnc,regEnc2))/real(dot(&encoding_space2,&encoding_space2)+calc_dot(regEnc2,regEnc2));
				else {

					alpha_cur = real(ss/ys);
				}
				old_norms.clear();
			}

			REAL alpha0=alpha_cur;
			REAL alpha;
			old_norms.push_back(functionValue(&encoding_space,regEnc,x));

			REAL gd = real(dot(g,&d));

			y = *g;
			s = *x;


			{
				FunctionEstimator f(&encoding_space,&encoding_space2,&regEnc,&regEnc2,x,&d,this);
				alpha = backtracking(f,alpha0,gd,rho,*std::max_element(old_norms.begin(),old_norms.end()));
				if (alpha <= 0){
					std::cout << "Backtracking linesearch failed, returning current iteration" << std::endl;
					return boost::shared_ptr<ARRAY_TYPE>(x);
				}				std::cout << "Alpha: " << alpha << std::endl;
			}


			if (non_negativity_constraint_){

				axpy(-alpha,&encoding_space2,&encoding_space);
				reg_axpy(-alpha,regEnc2,regEnc);

				ARRAY_TYPE x2 = *x;
				axpy(alpha,&d,&x2);

				clamp_min(&x2,REAL(0));

				d = x2;
				d -= *x;
				gd = real(dot(g,&d));
				x2 = *x;
				alpha0 = 1;
				this->encoding_operator_->mult_M(&d,&encoding_space2);
				calc_regMultM(&d,regEnc2);
				FunctionEstimator f(&encoding_space,&encoding_space2,&regEnc,&regEnc2,x,&d,this);
				//alpha=gold(f,0,alpha0*(1.0+std::sqrt(5.0))/2.0);
				alpha = backtracking(f,alpha0,gd,rho,*std::max_element(old_norms.begin(),old_norms.end()));
				if (alpha <= 0){
					std::cout << "Backtracking linesearch failed, returning current iteration" << std::endl;
					return boost::shared_ptr<ARRAY_TYPE>(x);
				}
				axpy(alpha,&d,x);
			} else {
				axpy(alpha,&d,x);

			}


			REAL f = functionValue(&encoding_space,regEnc,x)
			std::cout << "Function value: " <<  f << std::endl;


			this->encoding_operator_->mult_MH(&encoding_space,g);
			this->add_gradient(x,g);
			add_linear_gradient(regEnc,g);
			if (this->precond_.get()){
				this->precond_->apply(g,g);
				this->precond_->apply(g,g);
			}

			s -= *x;
			y -= *g;
			ss = dot(&x_old,&x_old);
			ys = dot(g_old,&x_old);
			yy = dot(g_old,g_old);

			if (old_norms.size()%max_cycles == 0)
				alpha_cur = 0;

			REAL criteria =
			if ( sqrt(ss) < std::min())




			if (grad_norm/grad_norm0 < tc_tolerance_)  break;
		}
		delete g,g_old;

		return boost::shared_ptr<ARRAY_TYPE>(x);
																															}



	// Set preconditioner
	//
	/*virtual void set_preconditioner( boost::shared_ptr< cgPreconditioner<ARRAY_TYPE> > precond ) {
      precond_ = precond;
      }*/

	// Set/get maximally allowed number of iterations
	//
	virtual void set_max_iterations( unsigned int iterations ) { iterations_ = iterations; }
	virtual unsigned int get_max_iterations() { return iterations_; }

	// Set/get tolerance threshold for termination criterium
	//
	virtual void set_tc_tolerance( REAL tolerance ) { tc_tolerance_ = tolerance; }
	virtual REAL get_tc_tolerance() { return tc_tolerance_; }

	virtual void set_non_negativity_constraint(bool non_negativity_constraint){
		non_negativity_constraint_=non_negativity_constraint;
	}

	virtual void set_dump_residual(bool dump_res){
		dump_residual = dump_res;
	}
	// Set preconditioner
	//

	virtual void set_preconditioner( boost::shared_ptr< cgPreconditioner<ARRAY_TYPE> > precond ) {
		precond_ = precond;
	}

	virtual void add_regularization_operator( boost::shared_ptr< linearOperator< ARRAY_TYPE> > op)
	{
		if( !op.get() ){
			throw std::runtime_error( "Error: linearOperatorSolver::add_regularization_operator : NULL operator provided" );
		}
		this->regularization_operators_.push_back(op);
		reg_priors.push_back(boost::shared_ptr<ARRAY_TYPE>((ARRAY_TYPE*)0));
	}

	virtual void add_regularization_operator( boost::shared_ptr< linearOperator< ARRAY_TYPE> > op,boost::shared_ptr<ARRAY_TYPE> prior)
	{
		if( !op.get() ){
			throw std::runtime_error( "Error: linearOperatorSolver::add_regularization_operator : NULL operator provided" );
		}

		this->regularization_operators_.push_back(op);
		reg_priors.push_back(prior);
	}

	virtual void add_regularization_operator(boost::shared_ptr< linearOperator<ARRAY_TYPE> > op, int L_norm ){
		if (L_norm==1){

			this->operators.push_back(boost::shared_ptr< l1GPRegularizationOperator>(new l1GPRegularizationOperator(op)));
		}else{
			add_regularization_operator(op);
		}
	}


	virtual void add_regularization_operator(boost::shared_ptr< linearOperator<ARRAY_TYPE> > op, boost::shared_ptr<ARRAY_TYPE> prior, int L_norm ){
		if (L_norm==1){
			this->operators.push_back(boost::shared_ptr<l1GPRegularizationOperator>(new l1GPRegularizationOperator(op,prior)));
		}else{
			add_regularization_operator(op,prior);
		}
	}


protected:
	typedef typename std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE> > >::iterator  csIterator;
	typedef typename std::vector< std::vector<boost::shared_ptr<linearOperator<ARRAY_TYPE> > > >::iterator csGroupIterator;

	virtual void solver_non_negativity_filter(ARRAY_TYPE*,ARRAY_TYPE*)=0;
	virtual void iteration_callback(ARRAY_TYPE*,int i,REAL,REAL){};




	ELEMENT_TYPE calc_dot(std::vector<ARRAY_TYPE>& x,std::vector<ARRAY_TYPE>& y){
		ELEMENT_TYPE res(0);
		for (int  i = 0; i < x.size(); i++)
			res += dot(&x[i],&y[i]);
		return res;
	}

	void add_linear_gradient(std::vector<ARRAY_TYPE>& elems, ARRAY_TYPE* g){
		ARRAY_TYPE tmp(g->get_dimensions());
		for (int i = 0; i <elems.size(); i++){
			this->regularization_operators_[i]->mult_MH(&elems[i],&tmp);
			axpy(std::sqrt(this->regularization_operators_[i]->get_weight()),&tmp,g);
		}
	}

	void calc_regMultM(ARRAY_TYPE* x,std::vector<ARRAY_TYPE>& elems){
		for (int i = 0; i <elems.size(); i++){
			this->regularization_operators_[i]->mult_M(x,&elems[i]);
			elems[i] *= std::sqrt(this->regularization_operators_[i]->get_weight());
		}
	}

	void reg_axpy(REAL alpha, std::vector<ARRAY_TYPE>& x, std::vector<ARRAY_TYPE>& y){
		for (int i = 0; i <x.size(); i++){
			axpy(alpha,&x[i],&y[i]);

		}
	}


	class FunctionEstimator{
	public:

		FunctionEstimator(ARRAY_TYPE* _encoding_space,ARRAY_TYPE* _encoding_step,std::vector<ARRAY_TYPE>* _regEnc,std::vector<ARRAY_TYPE>* _regEnc_step, ARRAY_TYPE * _x, ARRAY_TYPE * _d, acBbSolver<ARRAY_TYPE> * _parent)
	{
			encoding_step = _encoding_step;
			encoding_space = _encoding_space;
			regEnc = _regEnc;
			regEnc_step = _regEnc_step;
			x = _x;
			xtmp = *x;
			d = _d;
			parent = _parent;
			alpha_old = 0;
	}


		REAL operator () (REAL alpha){
			axpy(alpha-alpha_old,encoding_step,encoding_space);
			parent->reg_axpy(alpha-alpha_old,*regEnc_step,*regEnc);
			axpy(alpha-alpha_old,d,&xtmp);

			alpha_old = alpha;
			REAL res = parent->functionValue(encoding_space,*regEnc,&xtmp);
			return res;

		}




	private:

		REAL alpha_old;
		ARRAY_TYPE* encoding_step;
		ARRAY_TYPE * encoding_space;
		std::vector<ARRAY_TYPE>* regEnc;
		std::vector<ARRAY_TYPE>* regEnc_step;
		ARRAY_TYPE* x, *d;
		acBbSolver<ARRAY_TYPE>* parent;
		ARRAY_TYPE xtmp;


	};
	friend class FunctionEstimator;



	REAL backtracking(FunctionEstimator& f, const REAL alpha0, const REAL gd, const REAL rho, const REAL old_norm){
		REAL alpha;
		REAL delta=0.01;
		bool wolfe = false;
		int  k=0;

		while (not wolfe){
			alpha=alpha0*std::pow(rho,k);
			if (f(alpha) <= old_norm-abs(alpha*delta*gd)) wolfe = true;//Strong Wolfe condition..
			k++;
			//if (alpha == 0) throw std::runtime_error("Wolfe line search failed");
			if (alpha ==0) return 0;
		}

		return alpha;

	}



	REAL functionValue(ARRAY_TYPE* encoding_space,std::vector<ARRAY_TYPE>& regEnc, ARRAY_TYPE * x){
		REAL res= std::sqrt(this->encoding_operator_->get_weight())*real(dot(encoding_space,encoding_space));

		for (int i = 0; i  < this->operators.size(); i++){
			res += this->operators[i]->magnitude(x);
		}

		res += real(calc_dot(regEnc,regEnc));
		return res;

	}






protected:

	// Preconditioner
	//boost::shared_ptr< cgPreconditioner<ARRAY_TYPE> > precond_;
	// Maximum number of iterations
	unsigned int iterations_;
	bool non_negativity_constraint_;
	REAL tc_tolerance_;
	REAL threshold;
	bool dump_residual;
	REAL rho;

	REAL barrier_threshold;
	// Preconditioner

	std::vector<boost::shared_ptr<ARRAY_TYPE> > reg_priors;
	boost::shared_ptr< cgPreconditioner<ARRAY_TYPE> > precond_;

};
}
