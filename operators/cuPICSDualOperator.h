#pragma once

#include "primalDualOperator.h"
#include "cuTVPrimalDualOperator.h"
#include <cuNDArray.h>
namespace Gadgetron {
    template<class T>
    class cuPICSPrimalDualOperator : public cuTVPrimalDualOperator<T> {
    typedef cuTVPrimalDualOperator<T> parent;

    public:
        cuPICSPrimalDualOperator() : parent() {}
        cuPICSPrimalDualOperator(T alpha_ ) :  parent(alpha_){}
        virtual void primalDual(cuNDArray<T>* in, cuNDArray<T>* out,T sigma, bool accumulate) override {

            auto tmp_in = *in;
            tmp_in -= *prior;
            parent::primalDual(&tmp_in,out,sigma,accumulate);

        };

        void set_prior(boost::shared_ptr<cuNDArray<T>> p){
            prior = p;
        }

    protected:
        boost::shared_ptr<cuNDArray<T>> prior;


    };
}