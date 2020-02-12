#pragma once
#include <complex>
#include <vector>
#include "vector_td.h"

    class quadratureKernels {
    public:
        static std::vector<std::complex<float>> kernels;
        //static std::vector<float> m11, m12, m13, m22, m23, m33;
        static std::vector<Gadgetron::vector_td<float,6>> Tensor;
        static std::vector<Gadgetron::floatd3> directions;

    };

