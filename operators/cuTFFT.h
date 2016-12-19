//
// Created by dch on 19/12/16.
//

#ifndef GT_TOMOGRAPHY_CUTFFT_H
#define GT_TOMOGRAPHY_CUTFFT_H

#include <linearOperator.h>

#include <cuNDArray.h>

namespace Gadgetron {

    class cuTFFT : public linearOperator<cuNDArray < float>> {

    public:
        void mult_M(cuNDArray<float> *, cuNDArray<float> *, bool) override;

        void mult_MH(cuNDArray<float> *, cuNDArray<float> *, bool) override;

        virtual boost::shared_ptr< std::vector<size_t> > get_codomain_dimensions(){
            auto res = this->get_domain_dimensions();
            res->at(3) = res->at(3)+2;
            return res;
        }

        virtual void set_codomain_dimensions(std::vector<size_t>* size){
            throw std::runtime_error("Cannot set codomain dimension on cuTFFT");
        }


    };

}


#endif //GT_TOMOGRAPHY_CUTFFT_H
