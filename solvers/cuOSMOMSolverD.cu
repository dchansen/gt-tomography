#include "cuOSMOMSolverD.h"
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>

using namespace Gadgetron;


template<class T> struct updateG_functor {

    typedef typename realType<T>::Type REAL;
    updateG_functor(REAL tau_, REAL avg_lambda_) : tau(tau_), avg_lambda(avg_lambda_){
    }

    __device__ __inline__ T operator() (thrust::tuple<T,T,T,T> tup){
        T g = thrust::get<0>(tup);
        T x = thrust::get<1>(tup);
        T s = thrust::get<2>(tup);
        T precon = thrust::get<3>(tup);

        T result = (x-tau*g)+tau/avg_lambda*s/precon;
        result /= precon*tau/avg_lambda+REAL(1);
        return result;


    }

    typename realType<T>::Type tau,avg_lambda;
};

template<class T> void cuOSMOMSolverD<T>::updateG(cuNDArray<T>& g, cuNDArray<T>& x,cuNDArray<T>& s, cuNDArray<T>& precon, REAL tau, REAL avg_lambda ){
    auto begin_iterator = thrust::make_zip_iterator(thrust::make_tuple(g.begin(),x.begin(),s.begin(),precon.begin()));
    auto end_iterator = thrust::make_zip_iterator(thrust::make_tuple(g.end(),x.end(),s.end(),precon.end()));
    thrust::transform(begin_iterator,end_iterator,x.begin(),updateG_functor<T>(tau,avg_lambda));
}

template class cuOSMOMSolverD<float>;
