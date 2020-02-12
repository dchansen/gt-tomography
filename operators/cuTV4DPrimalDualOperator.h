#pragma once

#include "primalDualOperator.h"
#include "cuTVPrimalDualOperator.h"
#include <cuNDArray.h>
#include "cuGaussianFilterOperator.h"
namespace Gadgetron {
    template<class T>
    class cuTV4DPrimalDualOperator : public cuTVPrimalDualOperator<T> {
    typedef cuTVPrimalDualOperator<T> parent;



    public:
        cuTV4DPrimalDualOperator() : parent() {

        }
        cuTV4DPrimalDualOperator( T alpha_ ) :  parent(alpha_){

        }
        virtual void primalDual(cuNDArray<T>* in, cuNDArray<T>* out,T sigma, bool accumulate) override {

            if (!accumulate) clear(out);
            cuNDArray<T> tmp_in = *in;
            cuNDArray<T> tmp_out = *out;


            std::vector<size_t> dim3D {in->get_size(0),in->get_size(1),in->get_size(2)};
            size_t  elements3D = in->get_size(0)*in->get_size(1)*in->get_size(2);

            for (int i =0; i < in->get_size(3); i++){
                cuNDArray<T> view_in(dim3D,tmp_in.get_data_ptr()+elements3D*i);
                cuNDArray<T> view_in2(dim3D,in->get_data_ptr()+elements3D*((i+1)%in->get_size(3)));
                view_in -= view_in2;
            }

            parent::primalDual(&tmp_in,&tmp_out,sigma,false);

            *out += tmp_out;

            for (int i =0; i < in->get_size(3); i++){
                cuNDArray<T> view_out(dim3D,out->get_data_ptr()+elements3D*i);
                cuNDArray<T> view_out2(dim3D,tmp_out.get_data_ptr()+elements3D*((i-1+in->get_size(3))%in->get_size(3)));
                view_out -= view_out2;
            }



        };






    };
}