#pragma once

#include "gpuoperators_export.h"
#include "cuNDArray_math.h"
#include "generalOperator.h"

#include "complext.h"
#include "cuScaleOperator.h"

namespace Gadgetron{

    template<class T, unsigned int D> class  cuATvOperator
            : public cuTvOperator<T,D>
    {


    public:
        typedef typename realType<T>::Type REAL;

        cuATvOperator() : cuTvOperator<T,D>(){

        }

        virtual ~cuATvOperator(){};

        virtual void gradient(cuNDArray<T>* in ,cuNDArray<T>* out, bool accumulate=false){
            auto dims_small = *in->get_dimensions();
            for (int i = 0; i < 3; i++)
                dims_small[i] /= 2;

            cuNDArray<T> tmp_in(dims_small);
            cuNDArray<T> tmp_out(dims_small);

            scaleOp.mult_M(in,&tmp_in);
            parent::gradient(&tmp_in,&tmp_out,false);
            scaleOp.mult_MH(&tmp_out,out,accumulate);


        };
        virtual REAL magnitude(cuNDArray<T>* in){
            auto dims_small = *in->get_dimensions();
            for (int i = 0; i < 3; i++)
                dims_small[i] /= 2;

            cuNDArray<T> tmp_in(dims_small);
            return parent::magnitude(&tmp_in);
        }

    protected:
    typedef cuTvOperator<T,D> parent;
    protected:
        REAL limit_;
        cuScaleOperator<T,D> scaleOp;
    };
}
