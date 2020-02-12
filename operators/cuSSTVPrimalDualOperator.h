#pragma once

#include "primalDualOperator.h"
#include "cuTVPrimalDualOperator.h"
#include <cuNDArray.h>
#include "cuGaussianFilterOperator.h"
namespace Gadgetron {
    template<class T>
    class cuSSTVPrimalDualOperator : public cuTVPrimalDualOperator<T> {
    typedef cuTVPrimalDualOperator<T> parent;

        cuScaleOperator<T,3> scaleOp;
        cuGaussianFilterOperator<T,3> gauss;
    public:
        cuSSTVPrimalDualOperator() : parent() {
            gauss = cuGaussianFilterOperator<T,3>();
            gauss.set_sigma(1);
        }
        cuSSTVPrimalDualOperator(T alpha_ ) :  parent(alpha_){
            gauss = cuGaussianFilterOperator<T,3>();
            gauss.set_sigma(1);
        }
        virtual void primalDual(cuNDArray<T>* in, cuNDArray<T>* out,T sigma, bool accumulate) override {
             auto dims_small = *in->get_dimensions();
            for (int i = 0; i < 3; i++)
                dims_small[i] /= 2;

            cuNDArray<T> tmp_in(in->get_dimensions());
            cuNDArray<T> tmp_out(in->get_dimensions());
            gauss.mult_M(in,&tmp_in);

            cuNDArray<T> tmp_in_small(dims_small);
            cuNDArray<T> tmp_out_small(dims_small);

            scaleOp.mult_M(&tmp_in,&tmp_in_small);
            parent::primalDual(&tmp_in_small,&tmp_out_small,sigma,false);

            scaleOp.mult_MH(&tmp_out_small,&tmp_out);
            gauss.mult_MH(&tmp_out,out,accumulate);



        };






    };
}