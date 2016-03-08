//
// Created by dch on 12/02/16.
//

#ifndef GT_TOMOGRAPHY_HOCUOFPARTIALDERIVATIVEOPERATOR_H
#define GT_TOMOGRAPHY_HOCUOFPARTIALDERIVATIVEOPERATOR_H

#include <boost/smart_ptr/shared_array.hpp>
#include <hoCuNDArray.h>
#include "hoLinearResampleOperator_eigen.h"
#include "cuLinearResampleOperator.h"

namespace Gadgetron {
    template<class T>
    class hoCuOFPartialDerivativeOperator : public linearOperator<hoCuNDArray<T>> {
    public:
        hoCuOFPartialDerivativeOperator() : linearOperator<hoCuNDArray<T>>() {};

        void set_displacement_field(boost::shared_ptr<hoNDArray<T> > vf);

        virtual void mult_MH(hoCuNDArray<T>* in, hoCuNDArray<T>* out, bool accumulate) override;

        virtual void mult_M(hoCuNDArray<T>* in, hoCuNDArray<T>* out, bool accumulate) override;

    protected:
        //std::vector<hoLinearResampleOperator_eigen<T,3>> Rs;
        std::vector<cuLinearResampleOperator<T,3>> Rs;

    };
}


#endif //GT_TOMOGRAPHY_HOCUOFPARTIALDERIVATIVEOPERATOR_H
