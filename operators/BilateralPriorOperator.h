#pragma once
#include "cuNDArray.h"
#include "cuBilateralFilter.h"

namespace Gadgetron {

    class BilateralPriorOperator : public linearOperator<cuNDArray<float>> {
    public:
        BilateralPriorOperator() : sigma_int(1),sigma_spatial(1) {}

        void mult_M(cuNDArray<float>* in, cuNDArray<float>* out, bool accumulate = false) override{
            auto out_tmp = out;
            if (accumulate) out_tmp = new cuNDArray<float>(in->get_dimensions());
            *out_tmp = *in;


            std::vector<size_t> dims3D{in->get_size(0),in->get_size(1),in->get_size(2)};

            auto out_ptr = out_tmp->get_data_ptr();
            auto in_ptr = in->get_data_ptr();

            for (int i = 0; i < in->get_size(3); i++) {

                cuNDArray<float> out3D(dims3D,out_ptr);
                bilateral_filter(&out3D, prior.get(), sigma_spatial, sigma_int);
                out_ptr += out3D.get_number_of_elements();
            }
            *out_tmp -= *in;

            if (accumulate) {
                *out += *out_tmp;
                delete out_tmp;
            }
        }

        void mult_MH(cuNDArray<float>* in, cuNDArray<float>* out, bool accumulate = false) override {

            if (!norm_image) update_norm();

            auto out_tmp = out;
            if (accumulate) out_tmp = new cuNDArray<float>(in->get_dimensions());
            *out_tmp = *in;


            std::vector<size_t> dims3D{in->get_size(0),in->get_size(1),in->get_size(2)};

            auto out_ptr = out_tmp->get_data_ptr();
            auto in_ptr = in->get_data_ptr();

            for (int i = 0; i < in->get_size(3); i++) {

                cuNDArray<float> out3D(dims3D,out_ptr);
                out3D /= *norm_image;
                bilateral_filter_unnormalized(&out3D, prior.get(), sigma_spatial, sigma_int);
                out_ptr += out3D.get_number_of_elements();
            }
            *out_tmp -= *in;

            if (accumulate) {
                *out += *out_tmp;
                delete out_tmp;
            }
        }

        void set_prior(boost::shared_ptr<cuNDArray<float>> p){
            prior = p;
            norm_image = boost::shared_ptr<cuNDArray<float>>();
        }

        void set_sigma_int(float sigma) {
            sigma_int = sigma;
            norm_image = boost::shared_ptr<cuNDArray<float>>();
        }
        void set_sigma_spatial(float sigma) {
            sigma_spatial = sigma;
            norm_image = boost::shared_ptr<cuNDArray<float>>();
        }
    protected:

        void update_norm(){
            norm_image = boost::make_shared<cuNDArray<float>>(prior->get_dimensions());
            fill(norm_image.get(),1.0f);
            bilateral_filter_unnormalized(norm_image.get(),prior.get(),sigma_spatial,sigma_int);
        }

        float sigma_int,sigma_spatial;
        boost::shared_ptr<cuNDArray<float>> prior;
        boost::shared_ptr<cuNDArray<float>> norm_image;


    };

};