#pragma once

#include <cuNDArray_math.h>
#include <cuNDArray_fileio.h>
#include "cuSolverUtils.h"
#include "osMOMSolverD.h"
namespace Gadgetron {
    template<class T>
    class cuOSMOMSolverD : public osMOMSolverD<cuNDArray<T>> {
        typedef typename realType<T>::Type REAL;
        virtual void updateG(cuNDArray<T>& g, cuNDArray<T>& x,cuNDArray<T>& s, cuNDArray<T>& precon, REAL tau, REAL avg_lambda ) override;
    };
}