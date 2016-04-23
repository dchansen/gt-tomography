#pragma once

#include "cuEdgeWavelet.h"
#include "linearOperator.h"
#include "invertibleOperator.h"
#include "cuNDArray_math.h"
#include "cuNDArray_fileio.h"
#include <initializer_list>

namespace Gadgetron {
template<class T> class cuEdgeATrousOperator : public invertibleOperator<cuNDArray<T> > {

public:
	typedef typename realType<T>::Type REAL;

	cuEdgeATrousOperator() : invertibleOperator<cuNDArray<T>>() {

		std::vector<REAL> host_kernel= {1.0/16,1.0/4, 3.0/8, 1.0/4,1.0/16};
		kernel = thrust::device_vector<REAL>(host_kernel);
	}

	void set_sigma(REAL sigma_) { sigma=sigma_;}

	virtual void inverse(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate = false) override {
		auto img_dims = *out->get_dimensions();
		size_t depth = in->get_number_of_elements()/out->get_number_of_elements();
		if (!accumulate) clear(out);
		for (size_t i = 0; i < depth; i++){
			cuNDArray<T> in_view(img_dims,in->get_data_ptr()+i*out->get_number_of_elements());
			*out += in_view;
		}

	}

	virtual boost::shared_ptr<std::vector<size_t> > get_codomain_dimensions() override {
		auto return_dims = this->get_domain_dimensions();

		unsigned int max_levels = *std::max_element(levels.begin(),levels.end());
		return_dims->push_back(max_levels+1);
		return return_dims;
	}

	virtual void mult_MH_M(cuNDArray<T>* in, cuNDArray<T> * out, bool accumulate=false) override{
		cuNDArray<float> tmp(this->get_codomain_dimensions());
		mult_M(in,&tmp);
		mult_MH(&tmp,out,accumulate);

	}

	virtual void mult_M(cuNDArray<T>* in , cuNDArray<T>* out, bool accumulate=false){

		std::vector<unsigned int> current_levels = levels;

		if (!accumulate)
			clear(out);

		unsigned int max_levels = *std::max_element(levels.begin(),levels.end());

		if (out->get_number_of_elements()/in->get_number_of_elements() != (max_levels+1)){
			std::cout << "Out elements: " << out->get_number_of_elements() << " In elements " << in->get_number_of_elements() << std::endl;
			throw std::runtime_error("Output dimensions do not match number of levels requested");
		}

		auto prev = boost::make_shared<cuNDArray<T>>(*in);
		for (int level = 0; level < max_levels; level++){
			cuNDArray<T> out_view(in->get_dimensions(),out->get_data_ptr()+level*in->get_number_of_elements());
			out_view += *prev;
			apply_wavelet(prev.get(),level);

			/*std::stringstream ss;
			ss << "level" << level << ".real";
			write_nd_array(wav.get(),ss.str());*/
			out_view -= *prev; 

		}


		cuNDArray<T> out_view (in->get_dimensions(),out->get_data_ptr()+max_levels*in->get_number_of_elements());
		out_view = *prev;

	}

	virtual void mult_MH(cuNDArray<T>* in , cuNDArray<T>* out, bool accumulate=false){

/*
		std::vector<unsigned int> current_levels = levels;

		if (!accumulate)
			clear(out);

		unsigned int max_levels = *std::max_element(levels.begin(),levels.end());


		if (in->get_number_of_elements()/out->get_number_of_elements() != (max_levels+1)){

			throw std::runtime_error("Output dimensions do not match number of levels requested");
		}

		std::cout << "In norm " << nrm2(in) << std::endl;
		auto prev = boost::make_shared<cuNDArray<T>>(*in);
		for (int level = 0; level < max_levels; level++){
			cuNDArray<T> in_view(out->get_dimensions(), in->get_data_ptr()+level*out->get_number_of_elements());

			auto wave = boost::make_shared<cuNDArray<T>>(in_view);
			for (int i = 0; i < level; i++)wave = apply_wavelet(wave.get(),i);
			*out += *wave;
			*out -= *apply_wavelet(wave.get(),level);
		}

		cuNDArray<T> in_view(out->get_dimensions(), in->get_data_ptr()+max_levels*out->get_number_of_elements());
		*out += *apply_wavelet(&in_view,max_levels);
*/
		throw std::runtime_error("Not implemented!");

	}




	void set_kernel(std::vector<REAL> kernel_in){

		kernel = thrust::device_vector<REAL>(kernel_in);
	}
	void set_levels(std::initializer_list<unsigned int> levels_list){
		levels = std::vector<unsigned int>(levels_list);
	}

protected:

	void apply_wavelet(cuNDArray<T>* in_out, int level){
		cuNDArray<T>* tmp1 = in_out;
		if (level == -1)
			return;
		cuNDArray<T>* tmp = new cuNDArray<T>(in_out->get_dimensions());
		auto tmp2 = tmp;

		for (auto dim = 0; dim < levels.size(); dim++){
			if (levels[dim] > level){
				EdgeWavelet(tmp1,tmp2,&kernel,std::pow(2,level),dim,sigma,false);
				std::swap(tmp1,tmp2);
			}
		}

		if (tmp1 != in_out)
			*in_out = *tmp1;
		delete tmp;
	}
	std::vector<unsigned int> levels;
	thrust::device_vector<REAL> kernel;
	REAL sigma;
};

}
