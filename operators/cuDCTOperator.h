#pragma once 
#include "cuDCT.h"

#include "invertibleOperator.h"

namespace Gadgetron {

	template<class T> class cuDCTOperator : public Gadgetron::invertibleOperator<cuNDArray<T>> {

		static constexpr int stencil_size = 4;
	public:

		cuDCTOperator(): invertibleOperator<cuNDArray<T>>(){
			repetitions = 2;
		};

		virtual ~cuDCTOperator(){};
		virtual void mult_M(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate=false) override {

			cuNDArray<T>* tmp = out;

			if (accumulate) tmp = new cuNDArray<T>(out->get_dimensions());

			auto in_dims = in->get_dimensions();
			size_t elements = in->get_number_of_elements();
			for (auto i = 0; i < repetitions; i++){
				cuNDArray<T> tmp_view(in_dims,tmp->get_data_ptr()+elements*i);
				tmp_view = *in;
				dct2<T,stencil_size>(&tmp_view,i*stencil_size/repetitions);
				if (in->get_size(2) > 1)
					dct<T,stencil_size>(&tmp_view,2,i*stencil_size/repetitions);
				if (in->get_size(3) > 1)
					dct<T,10>(&tmp_view,3,i*10/repetitions);
				//dct2<T,2>(&tmp_view,2);
			}

			*tmp /= std::sqrt(T(repetitions));
			if (accumulate){
				*out += *tmp;
				delete tmp;
			}

		}


		virtual void mult_MH(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate=false) override {


			if (!accumulate) clear(out);

			auto out_dims = out->get_dimensions();
			size_t elements = out->get_number_of_elements();
			for (auto i = 0; i < repetitions; i++){
				cuNDArray<T> in_view(out_dims,in->get_data_ptr()+i*elements);
				cuNDArray<T> tmp(in_view);
				idct2<T,stencil_size>(&tmp,i*stencil_size/repetitions);
				if (out->get_size(2) > 1)
					idct<T,stencil_size>(&tmp,2,i*stencil_size/repetitions);
				if (out->get_size(3) > 1)
					idct<T,10>(&tmp,3,i*10/repetitions);
				//idct2<T,2>(&tmp,2);
				axpy(std::sqrt(T(1)/repetitions),&tmp,out);
			}

		}

		virtual void inverse(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate=false) override {
			mult_MH(in,out,accumulate);
		}



		virtual void set_domain_dimensions(std::vector<size_t>* dims) override {
			std::vector<size_t> codims = *dims;
			codims.push_back(repetitions);
			linearOperator<cuNDArray<T>>::set_codomain_dimensions(&codims);
			linearOperator<cuNDArray<T>>::set_domain_dimensions(dims);
		}

		virtual void set_codomain_dimensions(std::vector<size_t>* dims) override {
			throw std::runtime_error("Do not set codomain dimensions manually");
		}

	protected:
		unsigned int repetitions;


	};
};
