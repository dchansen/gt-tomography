//
// Created by dch on 19/12/16.
//

#include <nppdefs.h>
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
    NppiRect output_roi = {0,0,output_size.width,output_size.height};

    auto batch_size_in = image->get_size(0)* image->get_size(1);
    auto batch_size_out = result_size[0]*result_size[1];
    for (auto i = 0u; i < image->get_size(2); i++) {

        auto status = nppiResize_32f_C1R((Npp32f *) (image->get_data_ptr() + i * batch_size_in),
                                         image->get_size(0) * sizeof(float),size, roi,
                                         (Npp32f *) (result.get_data_ptr() + i * batch_size_out),
                                         result_size[0] * sizeof(float), output_size, output_roi, NPPI_INTER_LINEAR);
    }

    return result;
}