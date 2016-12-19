//
// Created by dch on 19/12/16.
//

#include "projection_utils.h"

using namespace Gadgetron;

cuNDArray<float> Gadgetron::downsample_projections(cuNDArray<float> *image, float factorX, float factorY) {

    NppiSize size = {image->get_size(0), image->get_size(1)};
    NppiPoint offset = {0, 0};
    NppiRect roi = {0,0,size.width,size.height};

    auto result_size = std::vector<size_t>{size_t(image->get_size(0) * factorX),
                                           size_t(image->get_size(1) * factorY), image->get_size(2)};

    cuNDArray<float> result(result_size);


    NppiSize output_size = {result_size[0], result_size[1]};

    auto batch_size_in = image->get_size(0)* image->get_size(1);
    auto batch_size_out = result_size[0]*result_size[1];
    for (auto i = 0u; i < image->get_size(2); i++) {

        auto status = nppiResize_32f_C1R((Npp32f *) (image->get_data_ptr() + i * batch_size_in),size,
                                         image->get_size(0) * sizeof(float), roi,
                                         (Npp32f *) (result.get_data_ptr() + i * batch_size_out),
                                         result_size[0] * sizeof(float),
                                         output_size, factorX, factorY, NPPI_INTER_LINEAR);
    }

    return result;
}