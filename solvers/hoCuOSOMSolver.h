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
template<class T> class hoCuOSMOMSolver : public solver< hoNDArray<T>,hoNDArray<T>> {

	typedef typename realType<T>::Type REAL;
public:
	hoCuOSMOMSolver() :solver< hoNDArray<T>,hoNDArray<T>>() {
		_iterations=10;
		_beta = REAL(1);
		_step_size = 1.0;
		_gamma = 0.0;
		non_negativity_=false;
		reg_steps_=1;
		_kappa = REAL(1);
		tau0=1e-3;
		denoise_alpha=0;
		dump=false;
	}
	virtual ~hoCuOSMOMSolver(){};

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
	void set_stepsize(REAL s){_step_size = s;}

	void set_dump(bool d){dump = d;}
	virtual void add_regularization_operator(boost::shared_ptr<linearOperator<cuNDArray<T>>> op, boost::shared_ptr<cuNDArray<T>> prior = boost::shared_ptr<cuNDArray<T>>()){
		regularization_operators.push_back(std::make_tuple(op,prior));
	}

	virtual void add_regularization_group(std::initializer_list<boost::shared_ptr<linearOperator<cuNDArray<T>>>> ops, boost::shared_ptr<cuNDArray<T>> prior = boost::shared_ptr<cuNDArray<T>>()){
		regularization_groups.push_back(std::make_tuple(std::vector<boost::shared_ptr<linearOperator<cuNDArray<T>>>>(ops),prior));
	}

	virtual void add_regularization_group(std::vector<boost::shared_ptr<linearOperator<cuNDArray<T>>>> ops, boost::shared_ptr<cuNDArray<T>> prior = boost::shared_ptr<cuNDArray<T>>()){
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
	void set_preconditioning_image(boost::shared_ptr<cuNDArray<T>> precon_image){
		this->preconditioning_image_ = precon_image;
	}


	void set_reg_steps(unsigned int reg_steps){ reg_steps_ = reg_steps;}

    boost::shared_ptr<hoNDArray<T>> solve(hoNDArray<T>* in){
        return this->solve(in, nullptr);
    }
	boost::shared_ptr<hoNDArray<T>> solve(hoNDArray<T>* in, hoNDArray<T>* weights){
		//boost::shared_ptr<hoNDArray<T>> rhs = compute_rhs(in);
		if( this->encoding_operator_.get() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : no encoding operator is set" );
			return boost::shared_ptr<hoNDArray<T>>();
		}

		// Get image space dimensions from the encoding operator
		//

		boost::shared_ptr< std::vector<size_t> > image_dims = this->encoding_operator_->get_domain_dimensions();
		if( image_dims->size() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : encoding operator has not set domain dimension" );
			return boost::shared_ptr<hoNDArray<T>>();
		}

		cuNDArray<T> * z = new cuNDArray<T>(*image_dims);
		if (this->x0_.get()){
			*z = *(this->x0_.get());
		} else  {
			clear(z);
		}


		cuNDArray<T> * uold = new cuNDArray<T>(*z);
		//cuNDArray<T>* u =new cuNDArray<T>(*z);

		std::vector<boost::shared_ptr<hoNDArray<T>> > subsets = projection_subsets(in);
        std::vector<boost::shared_ptr<hoNDArray<T>> > subset_weights;
        if (weights)
            subset_weights= projection_subsets(weights);


		boost::shared_ptr<cuNDArray<T>> precon_image;
		if (preconditioning_image_)
			precon_image = preconditioning_image_;
		else {
			precon_image = boost::make_shared<cuNDArray<T>>(image_dims.get());
			clear(precon_image.get());
			fill(uold,T(1));

			for (int isubset = 0; isubset < this->encoding_operator_->get_number_of_subsets(); isubset++){
				auto tmp_proj = cuNDArray<float>(subsets[isubset]->get_dimensions());
                cuNDArray<float> tmp_weights;
                if (weights) tmp_weights = cuNDArray<float>(*subset_weights[isubset]);
				this->encoding_operator_->mult_M(uold,&tmp_proj,isubset,false);
				std::cout << "Proj " << isubset << " " << nrm2(&tmp_proj) << std::endl;
                if (weights) tmp_proj *= tmp_weights;
                if (weights) tmp_proj *= tmp_weights;
				this->encoding_operator_->mult_MH(&tmp_proj,precon_image.get(),isubset,true);
			}

			std::cout << "Precon image " << nrm2(precon_image.get()) << std::endl;
			//*precon_image += _beta;
			clamp_min(precon_image.get(),T(1e-6));
			reciprocal_inplace(precon_image.get());
			*uold = *z;
			//ones_image *= (T) this->encoding_operator_->get_number_of_subsets();
		}

		cuNDArray<T> * x = new cuNDArray<T>(*z);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

		//hoNDArray<T> d(image_dims.get());
		//clear(&d);
		REAL avg_lambda = calc_avg_lambda();
		REAL t = 1;
		REAL told = 1;
		if( this->output_mode_ >= solver<hoNDArray<T>,hoNDArray<T>>::OUTPUT_VERBOSE ){
			std::cout << "osMOM setup done, starting iterations:" << std::endl;
		}

		std::vector<int> isubsets(boost::counting_iterator<int>(0), boost::counting_iterator<int>(this->encoding_operator_->get_number_of_subsets()));
		REAL kappa_int = _kappa;

		for (int i =0; i < _iterations; i++){
			for (int isubset = 0; isubset < this->encoding_operator_->get_number_of_subsets(); isubset++){

				t = 0.5*(1+std::sqrt(1+4*t*t));
				int subset = isubsets[isubset];
                {
                    auto tmp_proj = cuNDArray<T>(subsets[isubset]->get_dimensions());
                    cuNDArray <T> tmp_image(image_dims.get());
                    cuNDArray <T> cu_proj(subsets[subset]->get_dimensions());
                    cudaMemcpyAsync(cu_proj.get_data_ptr(),subsets[subset]->get_data_ptr(),cu_proj.get_number_of_bytes(),cudaMemcpyHostToDevice,stream);
                    //cudaMemcpy(cu_proj.get_data_ptr(),subsets[subset]->get_data_ptr(),cu_proj.get_number_of_bytes(),cudaMemcpyHostToDevice);
                    cuNDArray <T> cu_weights;
                    if (weights) {
                        cu_weights = cuNDArray<float>(subsets[subset]->get_dimensions());
                        cudaMemcpyAsync(cu_weights.get_data_ptr(), subset_weights[subset]->get_data_ptr(),
                                        cu_weights.get_number_of_bytes(), cudaMemcpyHostToDevice, stream);

                    }

                    this->encoding_operator_->mult_M(z, &tmp_proj, subset, false);
                    cudaStreamSynchronize(stream);
                    if (weights) tmp_proj *= cu_weights;
                    tmp_proj -= cu_proj;
                    if (weights) tmp_proj *= cu_weights;
                    if (this->output_mode_ >= solver < hoNDArray < T > , hoNDArray < T >> ::OUTPUT_VERBOSE) {
                        std::cout << "Iteration " << i << " Subset " << subset << " Update norm: " << nrm2(&tmp_proj) << " " << nrm2(&cu_proj) <<
                        std::endl;
                    }

                    this->encoding_operator_->mult_MH(&tmp_proj, &tmp_image, subset, false);

                    tmp_image *= -REAL(this->encoding_operator_->get_number_of_subsets()) /(1+_gamma*i);
                    //axpy(_beta,&d,&tmp_image);


                    tmp_image *= *precon_image;


                    *z += tmp_image;
                }

				{


					//s -= d;
					*x = *z;
					denoise(*x,*z,*precon_image,1.0,avg_lambda);

				}
				if (non_negativity_){
					clamp_min(x,T(0));
				}
				//axpy(REAL(_beta),&tmp_image,x);

				//d += *x;
				//d -= *z;
				//*z = *x;

				*z = *x;
				//*z = *u;
				*z *= 1+(told-1)/t;
				axpy(-(told-1)/t,uold,z);
				*uold = *x;
				//*u = z;
				//*x = *z;
				told = t;

				/*
				for (auto op : regularization_operators){

					op->gradient(x,&tmp_image);
					tmp_image /= nrm2(&tmp_image);
					auto reg_val = op->magnitude(x);
					std::cout << "Reg val: " << reg_val << std::endl;
					hoNDArray<T> y = *x;
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

				/**z = *x;

				*z *= 1+(told-1)/t;
				axpy(-(told-1)/t,xold,z);
				std::swap(x,xold);
				*x = *z;
				told = t;
				*/
				//step_size *= 0.99;

			}
			//std::reverse(isubsets.begin(),isubsets.end());
			//std::random_shuffle(isubsets.begin(),isubsets.end());
			/*
			hoNDArray<T> tmp_proj(*in);
			clear(&tmp_proj);
			this->encoding_operator_->mult_M(x,&tmp_proj,false);
			tmp_proj -= *in;
			 */

			if (dump){
				std::stringstream ss;
				ss << "osMOM-" << i << ".real";

				write_nd_array<T>(x,ss.str().c_str());
			}
			/*
			//calc_regMultM(x,regEnc);
			//REAL f = functionValue(&tmp_proj,regEnc,x);
			std::cout << "Function value: " << dot(&tmp_proj,&tmp_proj) << std::endl;
			 */
		}
		delete x,uold;


		auto result = z->to_host();

		cudaDeviceSynchronize();
        delete z;
        return result;
	}

	void set_encoding_operator(boost::shared_ptr<subsetOperator<cuNDArray<T>> > encoding_operator){ encoding_operator_ = encoding_operator; }


protected:

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
	void denoise(cuNDArray<T>& x, cuNDArray<T>& s, cuNDArray<T>& precon,REAL scaling,REAL avg_lambda ){
		REAL gam=0.35/(scaling*avg_lambda)/(precon.get_number_of_elements()/asum(&precon));
		REAL L = 4; //Hmm.. this seems a little..well... guessy?
		REAL tau = tau0;
		REAL sigma = 1/(tau*L*L);
		cuNDArray<T> g(x.get_dimensions());
		cuNDArray<T> xold(x);
		if (regularization_groups.empty() && regularization_operators.empty()){
			x = s;
			return;
		}

		for (auto it = 0u; it < reg_steps_; it++){
			clear(&g);
			REAL reg_val = 0;
			for (auto reg_op_pair : regularization_operators){
				auto reg_op = std::get<0>(reg_op_pair);
				auto prior = std::get<1>(reg_op_pair);
				cuNDArray<T> data(reg_op->get_codomain_dimensions());
				if (prior) x -= *prior;
				reg_op->mult_M(&x,&data);
				if (prior) x += *prior;

				reg_val += asum(&data)*reg_op->get_weight();
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
				std::vector<cuNDArray<T>> datas(reg_group.size());
				REAL val = 0;

				if (prior) x -= *prior;
				for (auto i = 0u; i < reg_group.size(); i++){
					datas[i] = cuNDArray<T>(reg_group[i]->get_codomain_dimensions());

					reg_group[i]->mult_M(&x,&datas[i]);

					auto tmp_dims = *datas[i].get_dimensions();
					reg_val += asum(&datas[i])*reg_group[i]->get_weight();
					datas[i] *= sigma*reg_group[i]->get_weight()/avg_lambda;
				}
				if (prior) x += *prior;

				//updateFgroup is the resolvent operators on the group
				updateFgroup(datas,denoise_alpha,sigma);

				for (auto i = 0u; i < reg_group.size(); i++){
					datas[i] *= reg_group[i]->get_weight()/avg_lambda;
					reg_group[i]->mult_MH(&datas[i],&g,true);

				}

			}
			//updateG is the resolvent operator on the |x-s| part of the optimization
			axpy(-tau,&g,&x);
			g = s;
			g /= precon;

			axpy(tau/(scaling*avg_lambda),&g,&x);

			g = precon;

			reciprocal_inplace(&g);
			g *= tau/(scaling*avg_lambda);
			g += REAL(1);
			x /= g;
			//x *= 1/(1+tau/(scaling*avg_lambda));
			//REAL theta = 1/std::sqrt(1+2*gam*tau);
			REAL theta = 0.5;
			//tau *= theta;
			//sigma /= theta;
			x *= REAL(1.0)+theta;
			axpy(REAL(-theta),&xold,&x);
			xold = x;
			g = s;
			g -= x;
			REAL ival = nrm2(&g);
			ival = ival*ival;

			std::cout << "Cost " << reg_val+ival << std::endl;

		}
	}


    std::vector<boost::shared_ptr<hoNDArray<T>> > projection_subsets(hoNDArray<T>* projections){

        std::vector<boost::shared_ptr<hoNDArray<T>> > res;
        T* curPtr = projections->get_data_ptr();
        for (int subset = 0; subset < encoding_operator_->get_number_of_subsets(); subset++){
            std::vector<size_t> subset_dim = *encoding_operator_->get_codomain_dimensions(subset);
            res.push_back(boost::shared_ptr<hoNDArray<T>>(new hoNDArray<T>(&subset_dim,curPtr)));
            curPtr += std::accumulate(subset_dim.begin(),subset_dim.end(),1,std::multiplies<unsigned int>());
        }
        return res;
    }

	std::vector<std::tuple< std::vector<boost::shared_ptr<linearOperator<cuNDArray<T>>>>,boost::shared_ptr<cuNDArray<T>> >> regularization_groups;

	std::vector<std::tuple<boost::shared_ptr<linearOperator<cuNDArray<T>> >, boost::shared_ptr<cuNDArray<T>>>> regularization_operators;

	int _iterations;
	REAL _beta, _gamma, _kappa,tau0, denoise_alpha, _step_size;
	bool non_negativity_, dump;
	unsigned int reg_steps_;
	boost::shared_ptr<subsetOperator<cuNDArray<T>> > encoding_operator_;
	boost::shared_ptr<cuNDArray<T>> preconditioning_image_;

};
}
