#pragma once

#include "hoNDArray.h"
#include "cuNDArray.h"
#include "cgPreconditioner.h"

namespace Gadgetron{

class protonPreconditioner: public cgPreconditioner<cuNDArray<float> > {

public:

	protonPreconditioner(std::vector<size_t> dims) : cgPreconditioner<cuNDArray<float> >() {

		uint64d2 vdims;
		vdims[0] = dims[0];
		vdims[1] = dims[1];
		angles = 720;
		kernel_ = calcKernel(vdims,angles);

	}

	virtual void apply(cuNDArray<float> * in, cuNDArray<float> * out);

	virtual void set_hull(boost::shared_ptr<cuNDArray<float> > hull){hull_ = hull;}
protected:

	unsigned int angles;
	boost::shared_ptr<cuNDArray<float_complext> > kernel_;

	boost::shared_ptr<cuNDArray<float> > hull_;
	static boost::shared_ptr<cuNDArray<float_complext> > calcKernel(uint64d2 dims, int angles);
};
}
