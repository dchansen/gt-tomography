#pragma once
#include <complex>
#include <vector>
#include "vector_td.h"

    class quadratureKernels {
    private:
        static std::vector<std::complex<float>> f1, f2, f3, f4, f5, f6;
        static std::vector<float> m11, m12, m13, m22, m23, m33;
        static std::vector<Gadgetron::floatd3> directions;

    };

