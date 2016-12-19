//
// Created by dch on 19/12/16.
//

#include "cuTFFT.h"
#include <cufft.h>
#include <cuNDFFT.h>
#include <cuNDArray_math.h>

using namespace Gadgetron;

void cuTFFT::mult_M(cuNDArray<float> * in, cuNDArray<float> * out, bool accumulate) {

    cufftHandle plan;

    cuNDArray<float> * out_tmp = out;
    if (accumulate)
        out_tmp = new cuNDArray<float>(out->get_dimensions());

    std::vector<int> tdim{in->get_size(3)};
    int istride = in->get_size(0)*in->get_size(1)*in->get_size(2);

    auto cuRes = cufftPlanMany(&plan, 1,tdim.data(),tdim.data(),istride,1,tdim.data(),istride,1,CUFFT_R2C,istride);
    if (cuRes != CUFFT_SUCCESS) {
        std::stringstream ss;
        ss << "cuNDFFT FFT plan failed: " << cuRes;
        throw std::runtime_error(ss.str());
    }

    auto result = cufftExecR2C(plan,(cufftReal*) in->get_data_ptr(), (cufftComplex*) out_tmp->get_data_ptr());
    if (result != CUFFT_SUCCESS) {
        std::stringstream ss;
        ss << "cuNDFFT FFT plan failed: " << result;
        throw std::runtime_error(ss.str());
    }

    *out_tmp *= 1.0f/std::sqrt(float(in->get_size(3)));
    if (accumulate){
        *out += *out_tmp;
        delete out_tmp;
    }
    cufftDestroy(plan);


}


void cuTFFT::mult_MH(cuNDArray<float> * in, cuNDArray<float> * out, bool accumulate) {

    cuNDArray<float> * out_tmp = out;
    if (accumulate)
        out_tmp = new cuNDArray<float>(out->get_dimensions());

    cufftHandle plan;

    std::vector<int> tdim{out->get_size(3)};
    int istride = out->get_size(0)*out->get_size(1)*out->get_size(2);

    auto cuRes = cufftPlanMany(&plan, 1,tdim.data(),tdim.data(),istride,1,tdim.data(),istride,1,CUFFT_C2R,istride);
    if (cuRes != CUFFT_SUCCESS) {
        std::stringstream ss;
        ss << "cuNDFFT FFT plan failed: " << cuRes;
        throw std::runtime_error(ss.str());
    }

    auto result = cufftExecC2R(plan,(cufftComplex*) in->get_data_ptr(), (cufftReal*) out_tmp->get_data_ptr());
    if (result != CUFFT_SUCCESS) {
        std::stringstream ss;
        ss << "cuNDFFT FFT plan failed: " << result;
        throw std::runtime_error(ss.str());
    }

    *out_tmp *= 1.0f/std::sqrt(float(out->get_size(3)));
    if (accumulate){
        *out += *out_tmp;
        delete out_tmp;
    }
    cufftDestroy(plan);

}