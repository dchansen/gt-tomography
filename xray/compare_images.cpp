//
// Created by dch on 13/03/17.
//

#include <boost/program_options.hpp>


#include "mssim.h"

#include <cuNDArray_fileio.h>
#include <cuNDArray_math.h>

namespace po = boost::program_options;

using namespace Gadgetron;

int main(int argc, char** argv) {

    std::string image_file;
    std::string reference_file;
    float sigma;
    po::options_description desc("Allowed options");

    desc.add_options()
            ("help", "produce help message")
            ("image",po::value<std::string>(&image_file))
            ("gold",po::value<std::string>(&reference_file))
            ("sigma",po::value<float>(&sigma)->default_value(1.5f));



	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

    auto image = read_nd_array<float>(image_file.c_str());
    auto gold = read_nd_array<float>(reference_file.c_str());



    auto dims = *image->get_dimensions();
    size_t phases = dims[3];

    std::vector<float> results;


    std::vector<size_t> dims3d(dims.begin(),dims.end()-1);

    size_t elements = std::accumulate(dims3d.begin(),dims3d.end(),1,std::multiplies<size_t>());

    for (size_t phase = 0; phase < phases; phase++){
        hoNDArray<float> image_view(dims3d,image->get_data_ptr()+phase*elements);
        hoNDArray<float> ref_view(dims3d,gold->get_data_ptr()+phase*elements);

        cuNDArray<float> cu_image(image_view);
        cuNDArray<float> cu_ref(ref_view);


        results.push_back(mssim(&cu_image,&cu_ref,floatd3(sigma,sigma,sigma)));
//
//        cu_image -= cu_ref;
//
//        results.push_back(nrm2(&cu_image));

    }


    float sum = std::accumulate(results.begin(),results.end(),0.0f)/results.size();

    std::cout << sum << std::endl;
//
//    for (auto r : results)
//        std::cout << r <<  " ";



}

