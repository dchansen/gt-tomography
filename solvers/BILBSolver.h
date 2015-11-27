#pragma once

#include "gpSolver.h"
#include "linearOperatorSolver.h"
#include "real_utilities.h"
#include "complext.h"
#include "cgPreconditioner.h"

#include <vector>
#include <iostream>
#include <numeric>
//#include <functional>
#include <list>

#include <boost/iterator/counting_iterator.hpp>
#include "solver_utils.h"
namespace Gadgetron{
/** Memory Limited BFGS Solver Adapted from Numerical Optimization (Wright and Nocedal 1999).
 *
 */

template <class ARRAY_TYPE> class BILBSolver : public gpSolver<ARRAY_TYPE>
{


protected:
	typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
	typedef typename realType<ELEMENT_TYPE>::Type REAL;
	typedef ARRAY_TYPE ARRAY_CLASS;
	typedef gpSolver<ARRAY_TYPE> GP;
	typedef typename gpSolver<ARRAY_TYPE>::l1GPRegularizationOperator l1GPRegularizationOperator;

public:

	BILBSolver(): gpSolver<ARRAY_TYPE>() {
		iterations_ = 10;
		tc_tolerance_ = (REAL)1e-7;
		non_negativity_constraint_=false;
		dump_residual = false;
		threshold= REAL(1e-7);
		m_ = 3;
		rho = 0.5f;
	}

	virtual ~BILBSolver(){}


	virtual void set_rho(REAL _rho){
		rho = _rho;
	}

	/***
	 * @brief Sets the number of iterations to use for estimating the Hessian. Memory usage increases linearly with m_;
	 * @param m
	 */
	virtual void set_m(unsigned int m){
		m_ = m;
	}

	void set_encoding_operator(boost::shared_ptr<subsetOperator<ARRAY_TYPE> > encoding_operator){ encoding_operator_ = encoding_operator; }

	virtual boost::shared_ptr<ARRAY_TYPE> solve(ARRAY_TYPE* in)
																																													{
		if( this->encoding_operator_.get() == 0 ){
			throw std::runtime_error("Error: BILBSolver::compute_rhs : no encoding operator is set" );
		}

		// Get image space dimensions from the encoding operator
		//

		boost::shared_ptr< std::vector<size_t> > image_dims = this->encoding_operator_->get_domain_dimensions();
		if( image_dims->size() == 0 ){
			throw std::runtime_error("Error: BILBSolver::compute_rhs : encoding operator has not set domain dimension" );
		}

		ARRAY_TYPE * x = new ARRAY_TYPE(image_dims.get()); //The image. Will be returned inside a shared_ptr

		ARRAY_TYPE g(image_dims.get()); //Contains the gradient of the current step
		//ARRAY_TYPE g_old(image_dims.get()); //Contains the gradient of the previous step


		//ARRAY_TYPE g_linear(image_dims.get()); //Contains the linear part of the gradient;

		//If a prior image was given, use it for the initial guess.
		if (this->x0_.get()){
			*x = *(this->x0_.get());
		} else  {
			clear(x);
		}

		// Contains the encoding space of the linear regularization operators
		std::vector<ARRAY_TYPE> regEnc;

		//Initialize encoding space
		for (int i = 0; i < this->regularization_operators_.size(); i++){
			regEnc.push_back(ARRAY_TYPE(this->regularization_operators_[i]->get_codomain_dimensions()));
			if (reg_priors[i].get()){
				regEnc.back() = *reg_priors[i];
				regEnc.back() *= -std::sqrt(this->regularization_operators_[i]->get_weight());
			}

		}


		ARRAY_TYPE d(image_dims.get()); //Search direction.
		clear(&d);

		ARRAY_TYPE encoding_space(in->get_dimensions().get()); //Contains the encoding space, or, equivalently, the residual vector

		std::vector<boost::shared_ptr<ARRAY_TYPE> > subsets = this->encoding_operator_->projection_subsets(in);

		//ARRAY_TYPE encoding_space(in->get_dimensions());
		//std::vector<boost::shared_ptr<ARRAY_TYPE> > encoding_spaces = this->encoding_operator_->projection_subsets(&encoding_space);

		ARRAY_TYPE x_old(*x);
		ARRAY_TYPE g_old(g);


		ARRAY_TYPE ones_projection(in->get_dimensions().get());
		ARRAY_TYPE tmp_image(image_dims.get());
		fill(&tmp_image,ELEMENT_TYPE(1));
		this->encoding_operator_->mult_M(&tmp_image,&ones_projection,false);
		clamp_min(&ones_projection,ELEMENT_TYPE(1e-6));
		reciprocal_inplace(&ones_projection);

		std::vector<boost::shared_ptr<ARRAY_TYPE> > ones_projections = this->encoding_operator_->projection_subsets(&ones_projection);

		std::vector<ARRAY_TYPE> ones_images;
		{
			ARRAY_TYPE tmp_projection(in->get_dimensions());
			fill(&tmp_projection,ELEMENT_TYPE(1));
			std::vector<boost::shared_ptr<ARRAY_TYPE> > tmp_projections = this->encoding_operator_->projection_subsets(&tmp_projection);



			for (int i = 0; i < this->encoding_operator_->get_number_of_subsets(); i++){
				ones_images.push_back(ARRAY_TYPE(image_dims.get()));
				this->encoding_operator_->mult_MH(tmp_projections[i].get(),&ones_images.back(),i,false);
				clamp_min(&ones_images.back(),ELEMENT_TYPE(1e-6));
				reciprocal_inplace(&ones_images.back());


			}
		}

		REAL reg_res,data_res;


		std::list<bfgsPair> subspace;

		if( this->output_mode_ >= solver<ARRAY_TYPE,ARRAY_TYPE>::OUTPUT_VERBOSE ){
			std::cout << "Iterating..." << std::endl;
		}
		REAL grad_norm0;
		REAL alpha0;
		REAL alpha = 1;

		std::vector<int> isubsets(boost::counting_iterator<int>(0), boost::counting_iterator<int>(this->encoding_operator_->get_number_of_subsets()));
		for (int i = 0; i < iterations_; i++){
			for (int isubset = 0; isubset < this->encoding_operator_->get_number_of_subsets(); isubset++){
				//int subset = isubsets[isubset];
				int subset = isubset;

				ARRAY_TYPE encoding_space(subsets[subset]->get_dimensions());
				this->encoding_operator_->mult_M(x,&encoding_space,subset,false);
				encoding_space -= *subsets[subset];
				encoding_space *= *ones_projections[subset];
				this->encoding_operator_->mult_MH(&encoding_space,&g,subset,false);

				alpha = 1;

				g *=  this->encoding_operator_->get_weight();
				calc_regMultM(x,regEnc);
				for (int n = 0; n < regEnc.size(); n++)
					if (reg_priors[n].get())
						axpy(-std::sqrt(this->regularization_operators_[n]->get_weight()),reg_priors[n].get(),&regEnc[n]);
				add_linear_gradient(regEnc,&g);
				this->add_gradient(x,&g);
				reg_res=REAL(0);


				if (non_negativity_constraint_) solver_non_negativity_filter(x,&g);
				REAL grad_norm = nrm2(&g);
				if( this->output_mode_ >= solver<ARRAY_TYPE,ARRAY_TYPE>::OUTPUT_VERBOSE ){
					std::cout << "Iteration " <<i << " Subset " << subset << " Update norm: " << nrm2(&encoding_space) << std::endl;
					//std::cout << "Iteration " <<i << ". Gradient norm: " <<  grad_norm << std::endl;
				}

				if ((i != 0) || (isubset != 0)){
					//Expand current BFGS subspace with new pair
					bfgsPair pair;
					if (subspace.size() == m_){
						pair=subspace.back();
						subspace.pop_back();
						*(pair.s) = *x;
						*(pair.y) = g;
					} else {
						pair.s = boost::shared_ptr<ARRAY_TYPE>(new ARRAY_TYPE(*x));
						pair.y = boost::shared_ptr<ARRAY_TYPE>(new ARRAY_TYPE(g));
					}
					*(pair.s) -= x_old;
					*(pair.y) -= g_old;

					pair.rho = dot(pair.s.get(),pair.y.get());

					subspace.push_front(pair);
				}
				//g *= ones_images[i];
				lbfgs_update(&g,&d,subspace);
				//				d = g;
				//				d *= REAL(-1);
				if (this->precond_.get()){
					this->precond_->apply(&d,&d);
					this->precond_->apply(&d,&d);
				}



				x_old = x;
				axpy(alpha,&d,x);
				clamp_min(x,REAL(0));
				g_old = g;

			}
			ARRAY_TYPE tmp_proj(*in);
			clear(&tmp_proj);
			this->encoding_operator_->mult_M(x,&tmp_proj,false);
			tmp_proj -= *in;
			calc_regMultM(x,regEnc);
			REAL f = functionValue(&tmp_proj,regEnc,x);
			std::cout << "Function value: " << f << std::endl;

			//iteration_callback(x,i,f);
			std::reverse(isubsets.begin(),isubsets.end());
			//if (grad_norm/grad_norm0 < tc_tolerance_)  break;

			std::stringstream ss;
			ss << "BILB-" << i << ".real";
			write_nd_array(x->to_host().get(),ss.str().c_str());
		}

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


	virtual void iteration_callback(ARRAY_TYPE* x ,int iteration,REAL value){};




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
	struct bfgsPair{
		boost::shared_ptr<ARRAY_TYPE> s;
		boost::shared_ptr<ARRAY_TYPE> y;
		ELEMENT_TYPE rho;
	};

	/***
	 * @brief L-BFGS update, following algorithm 9.2 in Numerical Optimization
	 * @param[in] g gradient
	 * @param[out] d search direction
	 * @param[in] pairs
	 */
	void lbfgs_update(ARRAY_TYPE* g, ARRAY_TYPE* d, std::list<bfgsPair>& pairs){
		*d = *g;

		if (pairs.size() > 0){
			std::list<ELEMENT_TYPE> alpha_list;
			for (typename std::list<bfgsPair>::iterator it = pairs.begin(); it != pairs.end(); ++it){
				ELEMENT_TYPE alpha = dot(it->s.get(),d)/it->rho;
				axpy(-alpha,it->y.get(),d);
				alpha_list.push_back(alpha);
			}

			bfgsPair front = pairs.front();
			ELEMENT_TYPE gamma = front.rho/dot(front.y.get(),front.y.get());
			*d *= gamma;

			typename std::list<ELEMENT_TYPE>::reverse_iterator alpha_it = alpha_list.rbegin();
			//Reverse iteration
			for (typename std::list<bfgsPair>::reverse_iterator it = pairs.rbegin(); it != pairs.rend(); ++it, ++alpha_it){
				ELEMENT_TYPE beta = dot(it->y.get(),d)/it->rho;
				ELEMENT_TYPE alpha = *alpha_it;
				axpy(alpha-beta,it->s.get(),d);
			}
		}
		*d *= REAL(-1);

	}





	class FunctionEstimator{
	public:

		FunctionEstimator(ARRAY_TYPE* _encoding_space,ARRAY_TYPE* _encoding_step,std::vector<ARRAY_TYPE>* _regEnc,std::vector<ARRAY_TYPE>* _regEnc_step, ARRAY_TYPE * _x, ARRAY_TYPE * _d, ARRAY_TYPE * _g, ARRAY_TYPE * _g_step, BILBSolver<ARRAY_TYPE> * _parent)
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
			g = _g;
			g_step = _g_step;

	}



		REAL operator () (REAL alpha){
			axpy(alpha-alpha_old,encoding_step,encoding_space);

			axpy(alpha-alpha_old,g_step,g);
			parent->reg_axpy(alpha-alpha_old,*regEnc_step,*regEnc);
			axpy(alpha-alpha_old,d,&xtmp);

			alpha_old = alpha;
			REAL res = parent->functionValue(encoding_space,*regEnc,&xtmp);
			return res;

		}

		ELEMENT_TYPE dir_deriv(){
			ARRAY_TYPE g_tmp = *g;
			parent->add_gradient(&xtmp,&g_tmp);
			return dot(d,&g_tmp);
		}






	private:

		REAL alpha_old;
		ARRAY_TYPE* encoding_step;
		ARRAY_TYPE * encoding_space;
		std::vector<ARRAY_TYPE>* regEnc;
		std::vector<ARRAY_TYPE>* regEnc_step;
		ARRAY_TYPE* x, *d;
		ARRAY_TYPE* g, *g_step;

		BILBSolver<ARRAY_TYPE>* parent;
		ARRAY_TYPE xtmp;


	};
	friend class FunctionEstimator;

	/***
	 * @brief Gold section search algorithm. Only works with unimodal functions, which we assume we're dealing with, at least locally
	 * @param f Functor to calculate the function to minimize
	 * @param a Start of the bracketing
	 * @param d End of bracketing
	 * @return Value minimizing the function f.
	 */
	REAL gold(FunctionEstimator& f, REAL a, REAL d){
		const REAL gold = 1.0/(1.0+std::sqrt(5.0))/2;

		REAL b = d-(d-a)*gold;
		REAL c = (d-a)*gold-a;

		REAL fa = f(a);
		REAL fb = f(b);
		REAL fc = f(c);
		REAL fd = f(d);
		REAL tol = 1e-6;

		while (abs(a-d) > tol*(abs(b)+abs(c))){
			if (fb > fc){
				a = b;
				fa = fb;
				b = c;
				fb = fc;
				c= b*gold+(1.0-gold)*d;
				fc = f(c);
			} else {
				d = c;
				fd = fc;
				c = b;
				fc = fb;
				b = c*gold+(1-gold)*a;
				fb = f(b);
			}
		}
		if (fb < fc){
			f(b);
			return b;
		}else {
			f(c);
			return c;
		}
	}

	/***
	 * Armijo type linesearch
	 * @param f
	 * @param alpha0
	 * @param gd
	 * @param rho
	 * @param old_norm
	 * @return
	 */
	REAL backtracking(FunctionEstimator& f, const REAL alpha0, const REAL gd, const REAL rho, const REAL old_norm){
		REAL alpha;
		REAL delta=1e-4;
		REAL sigma=0.9;
		//REAL precision = 0.0003; //Estimated precision of function evaluation
		REAL precision = 1e-4f; //Estimated precision of function evaluation
		bool wolfe = false;
		int  k=0;

		while (not wolfe){
			alpha=alpha0*std::pow(rho,k);
			//if (f(alpha) <= old_norm+alpha*delta*gd) wolfe = true;//Strong Wolfe condition..
			REAL fa = f(alpha);
			ELEMENT_TYPE dir_deriv = f.dir_deriv();
			if (((2*delta-1.0)*real(gd) >= real(dir_deriv)) && (fa < (old_norm+precision))) wolfe=true; //Approx Wolfe condition from Hager, W. and Zhang, H.SIAM Journal on Optimization 2005 16:1, 170-192
			if (abs(dir_deriv) > sigma*abs(gd)) wolfe = false;//Strong Wolfe condition..
			k++;
			if (alpha == 0){
				//std::cout << "Backtracking search failed, switching to slow wolfe-search" << std::endl;
				//return wolfesearch(f,alpha0,gd,rho,old_norm);
				return 0;
			}
		}

		return alpha;

	}

	/***
	 * Line search taken from Numerical Optimization (Wright and Nocedal 1999).
	 * Adapted from the scipy optimize algorithm.
	 * Like the gold-section method it works quite poorly in practice.
	 * @param f
	 * @param alpha0
	 * @param gd
	 * @param rho
	 * @param old_norm
	 * @return
	 */
	REAL wolfesearch(FunctionEstimator& f, const REAL alpha_init, const REAL gd, const REAL rho, const REAL old_norm){
		using std::sqrt;
		using std::abs;
		REAL delta=0.01;
		unsigned int k=0;
		REAL alpha0 = alpha_init;
		REAL f0 = f(alpha0);

		if (f0 <= old_norm+alpha0*delta*gd){//Strong Wolfe condition..
			return alpha0;
		}


		REAL alpha1 = -gd*alpha0*alpha0/2.0/(f0-old_norm-gd*alpha0);
		//std::cout << "F0 " <<f0 << " old " << old_norm << " gd " << gd <<std::endl;
		std::cout << "Alpha0: "  << alpha0 << std::endl;
		//std::cout << "Alpha1: "  << alpha1 << std::endl;
		REAL f1 = f(alpha1);


		if (f1 <= old_norm+alpha1*delta*gd){//Strong Wolfe condition..
			return alpha1;
		}


		while (alpha1 > 0){
			double factor = alpha0*alpha0*alpha1*alpha1*(alpha1-alpha0);
			double a = alpha0*alpha0*(f1-old_norm-gd*alpha1) - alpha1*alpha1*(f0-old_norm-gd*alpha0);
			a /= factor;

			double b = -alpha0*alpha0*alpha0*(f1-old_norm-gd*alpha1) + alpha1*alpha1*alpha1*(f0-old_norm-gd*alpha0);
			b /= factor;

			double alpha2 = (-b+std::sqrt(std::abs(b*b-3*a*gd)))/(3*a);
			REAL f2 = f(alpha2);
			//std::cout << "a " << a << "b " << b << std::endl;
			std::cout << "Alpha1: "  << alpha1 << std::endl;
			std::cout << "Alpha2: "  << alpha2 << std::endl;
			if (f2 < old_norm+alpha2*delta*gd){//Strong Wolfe condition..
				return alpha2;
			}

			if (((alpha1-alpha2) > (alpha1/2.0)) || ((1.0-alpha2/alpha1) < 0.96)){
				alpha2 = alpha1 / 2.0;
			}

			alpha0 = alpha1;
			alpha1 = alpha2;
			f0 = f1;
			f1 = f2;
			k++;


		}

		throw std::runtime_error("Wolfe line search failed");


	}



	/***
	 * CG linesearch adapted from  Hager, W. and Zhang, H.SIAM Journal on Optimization 2005 16:1, 170-192
	 * @param f
	 * @param alpha0
	 * @param gd
	 * @param rho
	 * @param old_norm
	 * @return
	 */
	REAL cg_linesearch(FunctionEstimator& f, const REAL alpha0, const REAL gd, const REAL old_norm){
		REAL delta=0.1;
		REAL sigma=0.9;
		REAL nabla=0.66;
		//REAL precision = 0.0003; //Estimated precision of function evaluation
		REAL precision = 1e-4f; //Estimated precision of function evaluation




		REAL a=0;
		REAL b = alpha0;

		REAL ak = a;
		REAL bk = b;
		REAL fa = old_norm;
		ELEMENT_TYPE a_deriv = gd;
		REAL fb = f(alpha0);
		ELEMENT_TYPE b_deriv = f.dir_deriv();

		while (abs(a-b) > 0){
			if ((((2*delta-1.0)*real(gd) >= real(b_deriv)) && (fb < old_norm+precision)) && //Check Approximate Wolfe conditions
					(abs(b_deriv) <= sigma*abs(gd))){
				f(b);
				return b;
			}

			if ((((2*delta-1.0)*real(gd) >= real(a_deriv)) && (fa < old_norm+precision)) && //Check Approximate Wolfe conditions
					(abs(a_deriv) <= sigma*abs(gd))){
				f(a);
				return a;
			}

			secant2(a,b,f,old_norm+precision);
			if ((b-a) > nabla*(bk-ak)) {
				REAL c = (a+b)/2;
				interval_update(a,b,c,f,old_norm);
			}
			if (a != ak){
				fa = f(a);
				a_deriv = f.dir_deriv();
			}

			if (b != bk){
				fb = f(b);
				b_deriv = f.dir_deriv();
			}

			ak = a;
			bk = b;

			std::cout << "a: " << a << " b: " << b << std::endl;
		}
		return 0;
		//throw std::runtime_error("CG_linesearch failed");

	}


	void secant2(REAL& a, REAL& b,FunctionEstimator& f,REAL old_norm){
		REAL fa = f(a);
		ELEMENT_TYPE dfa = f.dir_deriv();
		REAL fb = f(b);
		ELEMENT_TYPE dfb = f.dir_deriv();

		REAL c= real((a*dfb-b*dfa)/(dfb-dfa));

		REAL fc = f(c);
		ELEMENT_TYPE dfc = f.dir_deriv();

		REAL A=a;
		REAL B = b;

		interval_update(A,B,c,f,old_norm);

		if (c == B){
			c= real((b*dfc-c*dfb)/(dfc-dfb));
			interval_update(A,B,c,f,old_norm);
		} if (c == A){
			c= real((a*dfc-c*dfa)/(dfc-dfa));
			interval_update(A,B,c,f,old_norm);
		}

		a= A;
		b = B;
	}

	void interval_update(REAL & a, REAL & b, REAL c,FunctionEstimator& f,REAL old_norm){
		REAL theta = 0.5;
		if (c < a || c > b) return; // C not in interval
		REAL fc = f(c);
		ELEMENT_TYPE dfc = f.dir_deriv();

		if (real(dfc) >= 0){
			b =c;
			return;
		}
		if (fc < old_norm){
			a = c;
			return;
		}
		b =c;
		while(true){
			REAL d = (1-theta)*a+theta*b;
			REAL fd = f(d);
			ELEMENT_TYPE dfd = f.dir_deriv();

			if (real(dfd) >= 0){
				b = d;
				return;
			}
			if (fd < old_norm){
				a = d;
			} else 	b = d;

			std::cout << "Interval a: " << a << " b: " << b << std::endl;

		}




	}

	REAL functionValue(ARRAY_TYPE* encoding_space,std::vector<ARRAY_TYPE>& regEnc, ARRAY_TYPE * x){
		REAL res= std::sqrt(this->encoding_operator_->get_weight())*abs(dot(encoding_space,encoding_space));

		for (int i = 0; i  < this->operators.size(); i++){
			res += this->operators[i]->magnitude(x);
		}

		res += abs(calc_dot(regEnc,regEnc));
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

	unsigned int m_; // Number of copies to use.

	// Preconditioner

	std::vector<boost::shared_ptr<ARRAY_TYPE> > reg_priors;
	boost::shared_ptr< cgPreconditioner<ARRAY_TYPE> > precond_;
	boost::shared_ptr<subsetOperator<ARRAY_TYPE> > encoding_operator_;

};
}
