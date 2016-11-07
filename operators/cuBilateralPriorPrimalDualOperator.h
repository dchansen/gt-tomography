#pragma once

#include "primalDualOperator.h"
#include "cuTVPrimalDualOperator.h"
#include "cuWTVPrimalDualOperator.h"
#include <cuNDArray.h>
#include "BilateralPriorOperator.h"

namespace Gadgetron {

    class cuBilateralPriorPrimalDualOperator : public cuTVPrimalDualOperator<float> {
    typedef cuTVPrimalDualOperator<float> parent;


        BilateralPriorOperator bilOp;
    public:
        cuBilateralPriorPrimalDualOperator() : parent() {
        }
        cuBilateralPriorPrimalDualOperator(float sigma_int, float sigma_spatial, boost::shared_ptr<cuNDArray<float>> prior, float alpha_= 0 ) :  parent(alpha_){
            bilOp = BilateralPriorOperator();
            bilOp.set_sigma_spatial(sigma_spatial);
            bilOp.set_sigma_int(sigma_int);
            bilOp.set_prior(prior);
        }
        virtual void primalDual(cuNDArray<float>* in, cuNDArray<float>* out,float sigma, bool accumulate) override {


            cuNDArray<float> tmp_in = *in;
            cuNDArray<float> tmp_out = *out;
            bilOp.mult_M(in,&tmp_in);
            parent::primalDual(&tmp_in,&tmp_out,sigma,false);
            bilOp.mult_MH(&tmp_out,out,accumulate);



        };






    };
}