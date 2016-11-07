#pragma once

#include "primalDualOperator.h"
#include "cuTVPrimalDualOperator.h"
#include <cuNDArray.h>
#include "cuGaussianFilterOperator.h"
namespace Gadgetron {
    template<class T>
    class cuSSTVPrimalDualOperator : public cuTVPrimalDualOperator<T> {
    typedef cuTVPrimalDualOperator<T> parent;


        cuGaussianFilterOperator<T,3> gauss;
    public:
        cuSSTVPrimalDualOperator(T scale_ ) : parent() {
            gauss = cuGaussianFilterOperator<T,3>();
            gauss.set_sigma(scale_);
        }
        cuSSTVPrimalDualOperator(T scale_ , T alpha_ ) :  parent(alpha_){
            gauss = cuGaussianFilterOperator<T,3>();
            gauss.set_sigma(scale_);
        }
        virtual void primalDual(cuNDArray<T>* in, cuNDArray<T>* out,T sigma, bool accumulate) override {

            cuNDArray<T> tmp_in = *in;
            cuNDArray<T> tmp_out = *out;
            gauss.mult_M(in,&tmp_in);
            parent::primalDual(&tmp_in,&tmp_out,sigma,false);
            gauss.mult_MH(&tmp_out,out,accumulate);



        };






    };
}