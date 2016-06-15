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
    std::cout << "Nphases " << nphases << std::endl;
    //Rs = std::vector<hoLinearResampleOperator_eigen<T,3>>(nphases,hoLinearResampleOperator_eigen<T,3>());
    Rs = std::vector<cuLinearResampleOperator<T,3>>();

    for (auto i = 0u; i < nphases; i++){
        Rs.emplace_back();
        std::cout << "Setting displacementfield " << i << std::endl;
        auto array_view = boost::make_shared<hoNDArray<T>>(vdims,vf->get_data_ptr()+i*elements);
        auto cu_array_view = boost::make_shared<cuNDArray<T>>(*array_view);
        Rs.back().set_displacement_field(cu_array_view);
        Rs.back().mult_MH_preprocess();
    }


}

template<class T> void hoCuOFPartialDerivativeOperator<T>::mult_M(hoCuNDArray<T> *in, hoCuNDArray<T> *out, bool accumulate) {
    if (!in->dimensions_equal(out)) throw std::runtime_error("Dimensions of in and out must be equal");
    if (in->get_size(3) != Rs.size()) throw std::runtime_error("Temporal dimension must match number of vector fields");
    auto dims3d = *in->get_dimensions();
    dims3d.pop_back();
    size_t elements = in->get_number_of_elements()/in->get_size(3);
    if (!accumulate) clear(out);
    for (int i = 0u; i < Rs.size(); i++){
        auto in_view = hoCuNDArray<T>(dims3d,in->get_data_ptr()+i*elements);
        auto cu_in = cuNDArray<T>(in_view);
        auto cu_in_copy = cu_in;

        Rs[i].mult_M(&cu_in,&cu_in_copy);

        auto in_copy = cu_in_copy.to_host();

        auto in_view2 = hoCuNDArray<T>(dims3d,in->get_data_ptr()+((i+1)%Rs.size())*elements);
        auto out_view = hoCuNDArray<T>(dims3d,out->get_data_ptr()+i*elements);
        out_view += *in_copy;
        out_view -= in_view2;
    }
}
template<class T> void hoCuOFPartialDerivativeOperator<T>::mult_MH(hoCuNDArray<T> *in, hoCuNDArray<T> *out, bool accumulate) {
    if (!in->dimensions_equal(out)) throw std::runtime_error("Dimensions of in and out must be equal");
    auto dims3d = *in->get_dimensions();
    dims3d.pop_back();
    size_t elements = in->get_number_of_elements()/in->get_size(3);
    for (int i = 0u; i < Rs.size(); i++){
        auto in_view = hoCuNDArray<T>(dims3d,in->get_data_ptr()+i*elements);
        auto in_view2 = hoCuNDArray<T>(dims3d,in->get_data_ptr()+((i-1+Rs.size())%Rs.size())*elements);
        auto cu_in2 = cuNDArray<T>(in_view2);

        auto out_view = hoCuNDArray<T>(dims3d,out->get_data_ptr()+i*elements);

        auto cu_in = cuNDArray<T>(in_view);
        auto cu_out = cuNDArray<T>(out_view);
        Rs[i].mult_MH(&cu_in,&cu_out,accumulate);
        axpy(T(-1),&cu_in2,&cu_out);
        //Rs[(i-1)%Rs.size()].mult_MH(&cu_in2,&cu_out,true);
        cu_out.to_host(&out_view);
        //out_view = in_copy;
    }
}


template class hoCuOFPartialDerivativeOperator<float>;