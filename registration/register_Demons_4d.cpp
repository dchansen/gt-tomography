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

namespace po = boost::program_options;
using namespace Gadgetron;
using namespace std;

int main(int argc, char** argv){

    po::options_description desc("Allowed options");



    float sigma_diff,sigma_fluid,sigma_int,sigma_vdiff,alpha;
    bool composite;
    string image_filename;
    int iterations,levels;
    desc.add_options()
            ("help", "produce help message")
            ("image,f", po::value<string>(&image_filename)->default_value("reconstruction.real"), "4D image")
            ("alpha,a",po::value<float>(&alpha)->default_value(4.0),"Maximum step length per iteration")
            ("sigma_diff",po::value<float>(&sigma_diff)->default_value(1),"Diffusion sigma for regularization")
            ("sigma_fluid",po::value<float>(&sigma_fluid)->default_value(0),"Fluid sigma for regularization")
            ("sigma_int",po::value<float>(&sigma_int)->default_value(0),"Intensity sigma for regularization (bilateral)")
            ("iterations,i",po::value<int>(&iterations)->default_value(30),"Number of iterations to use")
            ("levels",po::value<int>(&levels)->default_value(0),"Number of multiresolution levels to use")
            ("sigma_vdiff",po::value<float>(&sigma_vdiff)->default_value(0),"Vector field difference sigma for regularization (bilateral)")
            ("composite",po::value<bool>(&composite)->default_value(true),"Do proper vector composition when adding vector fields")
            ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);


    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    cuDemonsSolver<float,3> demonsSolver;

    demonsSolver.set_sigmaDiff(sigma_diff);
    demonsSolver.set_sigmaFluid(sigma_fluid);
    demonsSolver.set_sigmaInt(sigma_int);
    demonsSolver.set_sigmaVDiff(sigma_vdiff);
    demonsSolver.set_compositive(composite);
    demonsSolver.set_exponential(true);
    demonsSolver.set_iterations(iterations);
    demonsSolver.set_alpha(alpha);
    demonsSolver.use_normalized_gradient_field(0.02);


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
        hoCuNDArray<float> movingImage(dims3D,image.get_data_ptr()+i*elements);
        hoCuNDArray<float> staticImage(dims3D,image.get_data_ptr()+(i+1)%ntimes*elements);

        cuNDArray<float> cuMov(movingImage);
        cuNDArray<float> cuStat(staticImage);

        auto cuVfield = demonsSolver.registration(cuStat,cuMov);

        auto vfieldView = hoCuNDArray<float>(vdims3D,vfield.get_data_ptr()+i*elements*3);
        vfieldView = cuVfield;



    }

    write_nd_array(&vfield,"vfield.real");




}