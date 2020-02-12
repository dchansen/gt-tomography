//
// Created by dchansen on 11/7/19.
//

#pragma once

#include<vector_td.h>

template< class T, unsigned int M, unsigned int N>
struct Matrix{

    Gadgetron::vector_td<T,N> data[M];

    __host__ __device__ T& operator()(int m, int n){
        return data[m][n];
    }

    __host__ __device__ T operator()(int m, int n) const {
        return data[m][n];
    }

};


template<class T, unsigned int M, unsigned int N>
__host__ __device__ Gadgetron::vector_td<T,M> dot(const Matrix<T,N,M>& matrix, const Gadgetron::vector_td<T,N>& vec){
    using namespace Gadgetron;
    vector_td<T, M> result;
    for (int i = 0; i < M; i++)
        result[i] = dot(matrix.data[i],vec);

    return result;
}
