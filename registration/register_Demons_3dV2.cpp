//
// Created by dch on 06/06/16.
//

#include "cuDemonsSolver.h"
#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <hoCuNDArray.h>
#include "hoNDArray_fileio.h"
#include <cuLinearResampleOperator.h>
#include <boost/make_shared.hpp>
#include "cuNDArray_fileio.h"
#include "vector_td_io.h"

namespace po = boost::program_options;
using namespace Gadgetron;
using namespace std;

int main(int argc, char** argv){

    po::options_description desc("Allowed options");



    float sigma_diff,sigma_fluid,sigma_int,sigma_vdiff,alpha;
    bool composite;
    string image_filename, output_filename;
    int iterations,levels;
    vector_td<float,3> phys_dims;
    desc.add_options()
            ("help", "produce help message")
            ("image,f", po::value<string>(&image_filename)->default_value("reconstruction.real"), "4D image")
            ("output", po::value<string>(&output_filename)->default_value("vfield.real"), "Output vfield name")
            ("alpha,a",po::value<float>(&alpha)->default_value(4.0),"Maximum step length per iteration")
            ("sigma_diff",po::value<float>(&sigma_diff)->default_value(1),"Diffusion sigma for regularization")
            ("sigma_fluid",po::value<float>(&sigma_fluid)->default_value(0),"Fluid sigma for regularization")
            ("sigma_int",po::value<float>(&sigma_int)->default_value(0),"Intensity sigma for regularization (bilateral)")
            ("iterations,i",po::value<int>(&iterations)->default_value(30),"Number of iterations to use")
            ("levels",po::value<int>(&levels)->default_value(0),"Number of multiresolution levels to use")
            ("sigma_vdiff",po::value<float>(&sigma_vdiff)->default_value(0),"Vector field difference sigma for regularization (bilateral)")
            ("composite",po::value<bool>(&composite)->default_value(true),"Do proper vector composition when adding vector fields")
            ("physical_dims",po::value<vector_td<float,3>>(&phys_dims)->default_value(vector_td<float,3>(1,1,1)),"Physical dimension in mm")
            ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);


    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    std::cout << "Command line options:" << std::endl;
    for (po::variables_map::iterator it = vm.begin(); it != vm.end(); ++it){
        boost::any a = it->second.value();
        std::cout << it->first << ": ";
        if (a.type() == typeid(std::string)) std::cout << it->second.as<std::string>();
        else if (a.type() == typeid(int)) std::cout << it->second.as<int>();
        else if (a.type() == typeid(unsigned int)) std::cout << it->second.as<unsigned int>();
        else if (a.type() == typeid(float)) std::cout << it->second.as<float>();
        else if (a.type() == typeid(vector_td<float,3>)) std::cout << it->second.as<vector_td<float,3> >();
        else if (a.type() == typeid(vector_td<int,3>)) std::cout << it->second.as<vector_td<int,3> >();
        else if (a.type() == typeid(vector_td<unsigned int,3>)) std::cout << it->second.as<vector_td<unsigned int,3> >();
        else std::cout << "Unknown type" << std::endl;
        std::cout << std::endl;
    }

    floatd3 sigma_diff_v = floatd3(sigma_diff);
    sigma_diff_v /= phys_dims;

    cuDemonsSolver<float,3> demonsSolver;

    demonsSolver.set_sigmaDiff(sigma_diff_v);
    demonsSolver.set_sigmaFluid(sigma_fluid);
    demonsSolver.set_sigmaInt(sigma_int);
    demonsSolver.set_sigmaVDiff(sigma_vdiff);
    demonsSolver.set_compositive(composite);
    demonsSolver.set_exponential(true);
    demonsSolver.set_iterations(iterations);
    demonsSolver.set_alpha(alpha);
    //demonsSolver.use_normalized_gradient_field(0.01);


    auto image = hoCuNDArray<float>(*read_nd_array<float>(image_filename.c_str()));

    if (image.get_number_of_dimensions() != 4)
        throw std::runtime_error("Image must be 4D");

    auto dims = *image.get_dimensions();

    auto dims3D = std::vector<size_t>(dims.begin(),dims.end()-1);
    auto vdims = std::vector<size_t>{dims[0],dims[1],dims[2],3,dims[3]};
    auto vdims3D = std::vector<size_t>(vdims.begin(),vdims.end()-1);

    auto vfield = hoCuNDArray<float>(vdims);

    size_t elements = image.get_number_of_elements()/dims[3];
    size_t ntimes = dims[3];
    for (size_t i = 0; i < vdims[4]; i++){
        std::cout << "Doing registration " << i << std::endl;
        hoCuNDArray<float> movingImage(dims3D,image.get_data_ptr()+(i+1)%ntimes*elements);
        hoCuNDArray<float> staticImage(dims3D,image.get_data_ptr()+i*elements);

        cuNDArray<float> cuMov(movingImage);
        cuNDArray<float> cuStat(staticImage);

        auto cuVfield = demonsSolver.multi_level_reg(cuStat,cuMov,levels);

        auto vfieldView = hoCuNDArray<float>(vdims3D,vfield.get_data_ptr()+i*elements*3);
        vfieldView = cuVfield;
    }


    auto final_vfield = hoCuNDArray<float>(vdims3D,vfield.get_data_ptr()+(vdims[4]-1)*elements*3);
    auto cufinal_vfield = cuNDArray<float>(final_vfield);


    std::cout << "Accumulating " << std::endl;
    for (int i = vdims[4]-2; i >= 0; i--){
        std::cout << "Number " << i << std::endl;
        auto vfieldView = hoCuNDArray<float>(vdims3D,vfield.get_data_ptr()+i*elements*3);
        auto cuVfieldView = cuNDArray<float>(vfieldView);
        deform_vfield(cuVfieldView,cufinal_vfield);
        cufinal_vfield += cuVfieldView;
    }


    std::cout << "Writing output " << std::endl;
    hoCuNDArray<float> tmp(vdims3D);
    tmp = cufinal_vfield;
    write_nd_array(&tmp,"vfield.real");

    std::cout << " Done? " << std::endl;

    return 0;




}