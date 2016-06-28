/*
 * osMOMSolver.h
 *
 *  Created on: Mar 23, 2015
 *      Author: u051747
 */
//Based on Wang et al, Accelerated statistical reconstruction using Nesterovs method
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
template<class T> class hoCuOSNESTSolver : public solver< hoCuNDArray<T>,hoCuNDArray<T>> {

	typedef typename realType<T>::Type REAL;
public:
	hoCuOSNESTSolver() :solver< hoCuNDArray<T>,hoCuNDArray<T>>() {
		_iterations=10;
		_beta = REAL(1);
		_step_size = 1.0;
		_gamma = 0;
		non_negativity_=false;
		reg_steps_=1;
		_kappa = REAL(1);
		tau0=1e-3;
		denoise_alpha=0;
		dump=false;
	}
	virtual ~hoCuOSNESTSolver(){};

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

    boost::shared_ptr<hoCuNDArray<T>> solve(hoCuNDArray<T>* in){
        this->solve(in, nullptr);
    }
	boost::shared_ptr<hoCuNDArray<T>> solve(hoCuNDArray<T>* in, hoCuNDArray<T>* I0){
		//boost::shared_ptr<hoCuNDArray<T>> rhs = compute_rhs(in);
		if( this->encoding_operator_.get() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : no encoding operator is set" );
			return boost::shared_ptr<hoCuNDArray<T>>();
		}

		// Get image space dimensions from the encoding operator
		//

		boost::shared_ptr< std::vector<size_t> > image_dims = this->encoding_operator_->get_domain_dimensions();
		if( image_dims->size() == 0 ){
			throw std::runtime_error( "Error: cgSolver::compute_rhs : encoding operator has not set domain dimension" );
			return boost::shared_ptr<hoCuNDArray<T>>();
		}

		cuNDArray<T> * z = new cuNDArray<T>(*image_dims);
		if (this->x0_.get()){
			*z = *(this->x0_.get());
		} else  {
			clear(z);
		}

		cuNDArray<T> * x = new cuNDArray<T>(*z);
		cuNDArray<T> * uold = new cuNDArray<T>(*z);
		//cuNDArray<T>* u =new cuNDArray<T>(*z);

		std::vector<boost::shared_ptr<hoCuNDArray<T>> > subsets = projection_subsets(in);
        std::vector<boost::shared_ptr<hoCuNDArray<T>> > subset_I0 = projection_subsets(I0);


		boost::shared_ptr<hoCuNDArray<T>> precon_proj = boost::make_shared<hoCuNDArray<T>>(in->get_dimensions());
		std::vector<boost::shared_ptr<hoCuNDArray<T>> > subset_precon = projection_subsets(precon_proj.get());
		{

			fill(uold,T(1));

			for (int isubset = 0; isubset < this->encoding_operator_->get_number_of_subsets(); isubset++){
				auto tmp_proj = cuNDArray<float>(subsets[isubset]->get_dimensions());
				this->encoding_operator_->mult_M(uold,&tmp_proj,isubset,false);
				*subset_precon[isubset] = tmp_proj;

			}
			*uold = *z;
			//ones_image *= (T) this->encoding_operator_->get_number_of_subsets();
		}


        cudaStream_t stream;
        cudaStreamCreate(&stream);

		cudaStream_t stream2;
		cudaStreamCreate(&stream2);

		//hoCuNDArray<T> d(image_dims.get());
		//clear(&d);
		REAL avg_lambda = calc_avg_lambda();
		REAL t = 1;
		REAL told = 1;
		if( this->output_mode_ >= solver<hoCuNDArray<T>,hoCuNDArray<T>>::OUTPUT_VERBOSE ){
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
					cuNDArray <T> cu_I0(subsets[subset]->get_dimensions());
					cudaMemcpyAsync(cu_I0.get_data_ptr(),subset_I0[subset]->get_data_ptr(),cu_I0.get_number_of_bytes(),cudaMemcpyHostToDevice,stream);
					cuNDArray <T> cu_precon(subsets[subset]->get_dimensions());
					cudaMemcpyAsync(cu_precon.get_data_ptr(),subset_precon[subset]->get_data_ptr(),cu_precon.get_number_of_bytes(),cudaMemcpyHostToDevice,stream2);

                    this->encoding_operator_->mult_M(z, &tmp_proj, subset, false);

                    cudaStreamSynchronize(stream);

					{
						std::cout << "Tmp proj " << min(&tmp_proj) << " " << max(&tmp_proj) << std::endl;
						cuNDArray<float> tmp_proj2(tmp_proj);
						//clamp_min(&tmp_proj2,T(0));
						exp_inv_inplace(&tmp_proj2);
						std::cout << "Tmp proj2 " << asum(&tmp_proj2) << std::endl;
						tmp_proj2 *= cu_I0;
						std::cout << "Tmp proj2 " << nrm2(&tmp_proj2) << std::endl;
						cu_proj -= tmp_proj2;
					}

					if (this->output_mode_ >= solver < hoCuNDArray < T > , hoCuNDArray < T >> ::OUTPUT_VERBOSE) {
						std::cout << "Iteration " << i << " Subset " << subset << " Update norm: " << nrm2(&cu_proj) <<
						std::endl;
					}

					cuNDArray <T> L(image_dims.get());
					this->encoding_operator_->mult_MH(&cu_proj, &L, subset, false);
					std::cout<< "L " << nrm2(&L) << std::endl;

					calcC(&tmp_proj,&cu_I0);
					std::cout << "Tmp_proj " << nrm2(&tmp_proj) << std::endl;
					cudaStreamSynchronize(stream2);
					tmp_proj *= cu_precon;
					std::cout<< "Tmp_proj " << nrm2(&tmp_proj) << std::endl;


					this->encoding_operator_->mult_MH(&tmp_proj, &tmp_image, subset, false);
					std::cout<< "Tmp_image " << nrm2(&tmp_image) << std::endl;

					//clamp_min(&tmp_image,1e-6);
					/*
					cuNDArray<float> numerator(image_dims.get());
					cuNDArray<float> denominator(image_dims.get());
					huber_norm(z,&numerator,&denominator,denoise_alpha);


					std::cout << mean(&tmp_image) << std::endl;

					axpy(_beta,&denominator,&tmp_image);
					//tmp_image += 1.0f;
					axpy(_beta,&numerator,&L);
*/

					tmp_image += _beta;
					L -= _beta;
					L /= tmp_image;
					std::cout<< "L " << nrm2(&L) << std::endl;
					std::cout << "Min " << min(&L) << " max " << max(&L) << std::endl;
                    //axpy(_beta,&d,&tmp_image);



                    *z -= L;
					std::cout<< "Z " << nrm2(z) << std::endl;

					//denoise(*x,*z,1.0,avg_lambda);
                }

				/*
				{


					//s -= d;
					*x = *z;
					denoise(*x,*z,1.0,avg_lambda);

				}
				 */



				//axpy(REAL(_beta),&tmp_image,x);

				clamp_min(z,T(0));

				*x = *z;

				*z *= 1+(told-1)/t;
				axpy(-(told-1)/t,uold,z);
				*uold = *x;

				told = t;


				clamp_min(z,T(0));




			}
			//std::reverse(isubsets.begin(),isubsets.end());
			//std::random_shuffle(isubsets.begin(),isubsets.end());
			/*
			hoCuNDArray<T> tmp_proj(*in);
			clear(&tmp_proj);
			this->encoding_operator_->mult_M(x,&tmp_proj,false);
			tmp_proj -= *in;
			 */

			if (dump){
				std::stringstream ss;
				ss << "osNEST-" << i << ".real";

				write_nd_array<T>(x,ss.str().c_str());
			}
			/*
			//calc_regMultM(x,regEnc);
			//REAL f = functionValue(&tmp_proj,regEnc,x);
			std::cout << "Function value: " << dot(&tmp_proj,&tmp_proj) << std::endl;
			 */
		}
		delete x,uold;

        cudaStreamDestroy(stream);
		cudaStreamDestroy(stream2);
        auto result = boost::make_shared<hoCuNDArray<T>>();
        *result = *z;
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
	void denoise(cuNDArray<T>& x, cuNDArray<T>& s, REAL scaling,REAL avg_lambda ){
		REAL gam=0.35/(scaling*avg_lambda);
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


			axpy(tau/(scaling*avg_lambda),&g,&x);


			x /= tau/(scaling*avg_lambda)+1;
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


    std::vector<boost::shared_ptr<hoCuNDArray<T>> > projection_subsets(hoCuNDArray<T>* projections){

        std::vector<boost::shared_ptr<hoCuNDArray<T>> > res;
        T* curPtr = projections->get_data_ptr();
        for (int subset = 0; subset < encoding_operator_->get_number_of_subsets(); subset++){
            std::vector<size_t> subset_dim = *encoding_operator_->get_codomain_dimensions(subset);
            res.push_back(boost::shared_ptr<hoCuNDArray<T>>(new hoCuNDArray<T>(&subset_dim,curPtr)));
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
