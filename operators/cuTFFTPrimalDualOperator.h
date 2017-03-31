#pragma once

#include "primalDualOperator.h"
#include "cuTFFT.h"
#include <cuNDArray.h>

namespace Gadgetron {


	class cuTFFTPrimalDualOperator : public primalDualOperator<cuNDArray < float>> {

	public:
		virtual void primalDual(cuNDArray<float> *in, cuNDArray<float> *out, float sigma = 0,
								bool accumulate = false) override {

			auto data = cuNDArray<float>(op.get_codomain_dimensions());
			op.mult_M(in, &data, false);
			data *= sigma * this->get_weight();

			auto dims = op.get_codomain_dimensions();

			auto cdims = std::vector<size_t>(dims->begin() + 1, dims->end()); //Copy cdims, excluding first dimension

			auto c_data = cuNDArray<complext<float>>(cdims, (complext<float> *) data.get_data_ptr());

			updateF(c_data, 0, sigma);
			op.mult_MH(&data, out, accumulate);


		}

		virtual void set_domain_dimensions(std::vector<size_t> *dims) {
			op.set_domain_dimensions(dims);
		}

	private:
		cuTFFT op;

	};

}
