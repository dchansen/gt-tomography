//
// Created by dch on 19/12/16.
//

#ifndef GT_TOMOGRAPHY_CUTVTFFT_H
#define GT_TOMOGRAPHY_CUTVTFFT_H

#include <linearOperator.h>

#include <cuNDArray.h>
#include "primalDualOperator.h"
namespace Gadgetron {

    class cuTVTFFT : public primalDualOperator<cuNDArray<float>> {


    public:

        cuTVTFFT() : primalDualOperator() {
            Ft = cuTFFT();
            TV = cuTVPrimalDualOperator<float>();

        }


        virtual void primalDual(cuNDArray<float>* in, cuNDArray<float>* out,float sigma, bool accumulate) override {

            cuNDArray<float> tmp(Ft.get_codomain_dimensions());
            Ft.mult_M(in,&tmp,false);
            {
                auto dim_order = std::vector<size_t>{1,2,3,4,0};

                auto tmp2 = permute(&tmp, &dim_order);
                tmp.reshape(tmp2->get_dimensions());

                TV.primalDual(tmp2.get(), &tmp, sigma, false);
            }
            auto dim_order = std::vector<size_t>{4,0,1,2,3};

            auto tmp2 = permute(&tmp,&dim_order);
            Ft.mult_MH(tmp2.get(),out,accumulate);






        };


        void set_domain_dimensions(std::vector<size_t> *dims){
            Ft.set_domain_dimensions(dims);
        }

        cuTVPrimalDualOperator<float> TV;
        cuTFFT Ft;

    };

}


#endif //GT_TOMOGRAPHY_CUTVTFFT_H
