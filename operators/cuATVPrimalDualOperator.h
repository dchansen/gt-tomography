#pragma once

#include "primalDualOperator.h"
#include "cuTVPrimalDualOperator.h"
#include <cuNDArray.h>
#include "cuBoxFilterOperator.h"
namespace Gadgetron {
    template<class T>
    class cuATVPrimalDualOperator : public cuTVPrimalDualOperator<T> {
    typedef cuTVPrimalDualOperator<T> parent;

        cuScaleOperator<T,3> scaleOp;
//cuBoxFilterOperator<T,3> scaleOp;
    public:
        cuATVPrimalDualOperator() : parent() {}
        cuATVPrimalDualOperator(T alpha_ ) :  parent(alpha_){}
        virtual void primalDual(cuNDArray<T>* in, cuNDArray<T>* out,T sigma, bool accumulate) override {
            auto dims_small = *in->get_dimensions();
            for (int i = 0; i < 3; i++)
                dims_small[i] /= 2;

            cuNDArray<T> tmp_in(dims_small);
            cuNDArray<T> tmp_out(dims_small);

            scaleOp.mult_M(in,&tmp_in);
            parent::primalDual(&tmp_in,&tmp_out,sigma,false);
            scaleOp.mult_MH(&tmp_out,out,accumulate);



        };




    };
}