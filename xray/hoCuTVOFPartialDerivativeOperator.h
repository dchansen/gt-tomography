//
// Created by dch on 12/02/16.
//

#pragma once

#include <boost/smart_ptr/shared_array.hpp>
#include <hoCuNDArray.h>
#include "hoLinearResampleOperator_eigen.h"
#include "cuLinearResampleOperator.h"
#include "hoCuOFPartialDerivativeOperator.h"

namespace Gadgetron {
    template<class T>
    class hoCuTVOFPartialDerivativeOperator : public hoCuOFPartialDerivativeOperator<T> {
        typedef hoCuOFPartialDerivativeOperator<T> parent;
    public:
        hoCuTVOFPartialDerivativeOperator() : hoCuOFPartialDerivativeOperator<T>(), dX(0), dY(1), dZ(1) {};


        virtual void mult_M(hoCuNDArray<T>* in, hoCuNDArray<T>* out, bool accumulate) override{
            auto tmp_out = *in;
            std::cout << "Number of elements" << in->get_number_of_elements() << " " << out->get_number_of_elements() << std::endl;

            if (in->get_number_of_elements()*3 != out->get_number_of_elements())
                throw std::runtime_error("Input and output is inconsistent");

            parent::mult_M(in,&tmp_out,false);

            size_t elements = in->get_number_of_elements();
            hoCuNDArray<T> out_viewX(in->get_dimensions(),out->get_data_ptr());
            hoCuNDArray<T> out_viewY(in->get_dimensions(),out->get_data_ptr()+elements);
            hoCuNDArray<T> out_viewZ(in->get_dimensions(),out->get_data_ptr()+2*elements);
            dX.mult_M(&tmp_out, &out_viewX,accumulate);
            dY.mult_M(&tmp_out, &out_viewY,accumulate);
            dZ.mult_M(&tmp_out, &out_viewZ,accumulate);



        };

        virtual void mult_MH(hoCuNDArray<T>* in, hoCuNDArray<T>* out, bool accumulate) override {

            hoCuNDArray<T> tmp_out(out->get_dimensions());
            size_t elements = out->get_number_of_elements();
            hoCuNDArray<T> in_viewX(out->get_dimensions(),in->get_data_ptr());
            hoCuNDArray<T> in_viewY(out->get_dimensions(),in->get_data_ptr()+elements);
            hoCuNDArray<T> in_viewZ(out->get_dimensions(),in->get_data_ptr()+2*elements);
            dX.mult_MH(&in_viewX, &tmp_out,false);
            dY.mult_MH(&in_viewY, &tmp_out,true);
            dZ.mult_MH(&in_viewZ, &tmp_out,true);

            parent::mult_MH(&tmp_out,out,accumulate);


        }

        hoCuPartialDerivativeOperator<T,3> dX;
        hoCuPartialDerivativeOperator<T,3> dY;
        hoCuPartialDerivativeOperator<T,3> dZ;


    };
}


