//
// Created by dch on 12/02/16.
//

#include "hoCuOFPartialDerivativeOperator.h"
#include <boost/make_shared.hpp>
using namespace Gadgetron;


template<class T> void Gadgetron::hoCuOFPartialDerivativeOperator<T>::set_displacement_field(boost::shared_ptr<Gadgetron::hoNDArray<T>> vf) {

    size_t nphases = vf->get_size(4);

    auto vdims = *vf->get_dimensions();
    vdims.pop_back();
    size_t elements = vf->get_number_of_elements()/nphases;
    Rs = std::vector<hoLinearResampleOperator_eigen<T,3>>(nphases,hoLinearResampleOperator_eigen<T,3>());

    for (auto i = 0u; i < nphases; i++){
        std::cout << "Setting displacementfield " << i << std::endl;
        auto array_view = boost::make_shared<hoNDArray<T>>(vdims,vf->get_data_ptr()+i*elements);
        Rs[i].set_displacement_field(array_view);
    }


}

template<class T> void hoCuOFPartialDerivativeOperator<T>::mult_M(hoCuNDArray<T> *in, hoCuNDArray<T> *out, bool accumulate) {
    if (!in->dimensions_equal(out)) throw std::runtime_error("Dimensions of in and out must be equal");
    auto dims3d = *in->get_dimensions();
    dims3d.pop_back();
    size_t elements = in->get_number_of_elements()/in->get_size(3);
    if (!accumulate) clear(out);
    for (auto i = 0u; i < Rs.size(); i++){
        auto in_view = hoCuNDArray<T>(dims3d,in->get_data_ptr()+i*elements);
        auto in_copy = in_view;

        Rs[i].mult_M(&in_view,&in_copy);

        auto in_view2 = hoCuNDArray<T>(dims3d,in->get_data_ptr()+((i+1)%Rs.size())*elements);
        auto out_view = hoCuNDArray<T>(dims3d,out->get_data_ptr()+i*elements);
        out_view += in_copy;
        out_view -= in_view2;
    }
}
template<class T> void hoCuOFPartialDerivativeOperator<T>::mult_MH(hoCuNDArray<T> *in, hoCuNDArray<T> *out, bool accumulate) {
    if (!in->dimensions_equal(out)) throw std::runtime_error("Dimensions of in and out must be equal");
    auto dims3d = *in->get_dimensions();
    dims3d.pop_back();
    size_t elements = in->get_number_of_elements()/in->get_size(3);
    for (auto i = 0u; i < Rs.size(); i++){
        auto in_view = hoCuNDArray<T>(dims3d,in->get_data_ptr()+i*elements);
        auto in_view2 = hoCuNDArray<T>(dims3d,in->get_data_ptr()+((i-1)%Rs.size())*elements);
        auto in_copy = in_view;
        in_copy -= in_view2;
        auto out_view = hoCuNDArray<T>(dims3d,out->get_data_ptr()+i*elements);
        Rs[i].mult_MH(&in_copy,&out_view,accumulate);
        //out_view = in_copy;
    }
}


template class hoCuOFPartialDerivativeOperator<float>;