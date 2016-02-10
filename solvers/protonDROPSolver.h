#pragma once
#include "subsetOperator.h"
#include "solver.h"
#include <numeric>
#include <vector>
#include <functional>
#include <boost/iterator/counting_iterator.hpp>



namespace Gadgetron{
template <template<class> class ARRAY> class protonDROPSolver : public solver< ARRAY<float> ,ARRAY<float> > {
	typedef float ELEMENT_TYPE;
	typedef float REAL;
public:
	protonDROPSolver() :solver< ARRAY<float> ,ARRAY<float> >() {
		_iterations=10;
		_beta = REAL(1);
		_kappa = REAL(1);
		_gamma = 0;
		non_negativity_=false;
		_dump = false;
	}
	virtual ~protonDROPSolver(){};

	void set_max_iterations(int i){_iterations=i;}
	int get_max_iterations(){return _iterations;}
	void set_non_negativity_constraint(bool neg=true){non_negativity_=neg;}
	/**
	 * @brief Sets the weight of each step in the SART iteration
	 * @param beta
	 */
	void set_beta(REAL beta){_beta = beta;}
	void set_gamma(REAL gamma){_gamma = gamma;}
	void set_dump(bool dump){_dump = dump;}

	boost::shared_ptr<ARRAY<float> > solve(ARRAY<float>* in){
		//boost::shared_ptr<ARRAY> rhs = compute_rhs(in);
		if( this->encoding_operator_.get() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : no encoding operator is set" );
			return boost::shared_ptr<ARRAY<float> >();
		}

		// Get image space dimensions from the encoding operator
		//

		boost::shared_ptr< std::vector<size_t> > image_dims = this->encoding_operator_->get_domain_dimensions();
		if( image_dims->size() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : encoding operator has not set domain dimension" );
			return boost::shared_ptr<ARRAY<float> >();
		}

		ARRAY<float> * x = new ARRAY<float>;
		x->create(image_dims.get());
		if (this->x0_.get()){
			*x = *(this->x0_.get());
		} else  {
			clear(x);
		}
		ARRAY<float> tmp_projection(in->get_dimensions());
		std::vector<boost::shared_ptr<ARRAY<float> > > tmp_projections = this->encoding_operator_->projection_subsets(&tmp_projection);

		ARRAY<float> tmp_image(image_dims);
		std::vector<boost::shared_ptr<ARRAY<float> > > subsets = this->encoding_operator_->projection_subsets(in);

		ARRAY<float> projection_norm(in->get_dimensions().get());
		//ARRAY tmp_image(image_dims.get());

		//fill(&tmp_image,REAL(1));
		//this->encoding_operator_->mult_M(&tmp_image,&projection_norm,false);


		float  kappa_int = _kappa;

		encoding_operator_->pathNorm(&projection_norm);
		clamp_min(&projection_norm,ELEMENT_TYPE(1e-6));

		std::cout << "Path norm: " << asum(&projection_norm) << std::endl;
		reciprocal_inplace(&projection_norm);
		std::cout << "Path norm2: " << asum(&projection_norm) << std::endl;

		std::vector<boost::shared_ptr<ARRAY<float> > > projection_norms = this->encoding_operator_->projection_subsets(&projection_norm);

		std::vector<ARRAY<float> > count_images;

		for (int i = 0; i < this->encoding_operator_->get_number_of_subsets(); i++){
			count_images.push_back(ARRAY<float>(image_dims.get()));
			clear(&count_images.back());
			this->encoding_operator_->protonCount(&count_images.back(),i);
			clamp_min(&count_images.back(),REAL(1));
			reciprocal_inplace(&count_images.back());
		}
		if( this->output_mode_ >= solver<ARRAY<float>,ARRAY<float> >::OUTPUT_VERBOSE ){
			std::cout << "protonDROP setup done, starting iterations:" << std::endl;
		}

		std::vector<int> isubsets(boost::counting_iterator<int>(0), boost::counting_iterator<int>(this->encoding_operator_->get_number_of_subsets()));

		float reg_val;

		for (int i =0; i < _iterations; i++){
			for (int isubset = 0; isubset < this->encoding_operator_->get_number_of_subsets(); isubset++){
				int subset = isubsets[isubset];
				this->encoding_operator_->mult_M(x,tmp_projections[subset].get(),subset,false);
				*tmp_projections[subset] -= *subsets[subset];
				*tmp_projections[subset] *= ELEMENT_TYPE(-1);
				if( this->output_mode_ >= solver<ARRAY<float> ,ARRAY<float> >::OUTPUT_VERBOSE ){
					std::cout << "Iteration " <<i << " Subset " << subset << " Update norm: " << nrm2(tmp_projections[subset].get()) << std::endl;
				}

				*tmp_projections[subset] *= *projection_norms[subset];
				this->encoding_operator_->mult_MH(tmp_projections[subset].get(),&tmp_image,subset,false);
				tmp_image *= count_images[subset];
				axpy(REAL(_beta/(1+_gamma*i)),&tmp_image,x);
				if (non_negativity_){
					clamp_min(x,ELEMENT_TYPE(0));
				}

				if (reg_op){
					ARRAY<float> grad(x->get_dimensions());
					reg_op->gradient(x,&grad);
					grad /= nrm2(&grad);
					reg_val = reg_op->magnitude(x);
					std::cout << "Reg val: " << reg_val << std::endl;
					ARRAY<float> y = *x;
					axpy(-kappa_int,&grad,&y);


					while(reg_op->magnitude(&y) > reg_val){

						kappa_int /= 2;
						axpy(kappa_int,&grad,&y);
						std::cout << "Kappa: " << kappa_int << std::endl;
					}
					reg_val = reg_op->magnitude(&y);
					std::cout << "Reg val new: " << reg_val << std::endl;

					*x = y;

				}

			}
			if (_dump){
				std::stringstream ss;
				ss << "protonDROP-" << i << ".real";
				write_nd_array(x,ss.str().c_str());
			}
		}
		return boost::shared_ptr<ARRAY<float> >(x);
	}

	void set_encoding_operator(boost::shared_ptr<protonSubsetOperator<ARRAY> > encoding_operator){ encoding_operator_ = encoding_operator; }
	void set_reg_op(boost::shared_ptr< generalOperator<ARRAY<float>  > > regularization_operator){reg_op = regularization_operator;}

protected:
	int _iterations;
	REAL _beta, _gamma, _kappa;
	bool non_negativity_;
	bool _dump;
	boost::shared_ptr< generalOperator<ARRAY<float>  > > reg_op;
	boost::shared_ptr<protonSubsetOperator<ARRAY> > encoding_operator_;



};
}
