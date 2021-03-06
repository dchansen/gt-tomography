//
// Created by dch on 18/02/16.
//

#include <boost/program_options.hpp>
#include <cuNDArray.h>
#include <cuNDArray_fileio.h>
#include "CT_acquisition.h"
#include "CTProjectionOperator.h"
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <algorithm>
#include <cuCgSolver.h>
#include <cuGpBbSolver.h>
#include <cuNlcgSolver.h>

using namespace std;
using namespace Gadgetron;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

std::vector<string> get_dcm_files(std::string dir){

    fs::directory_iterator end_itr;
    const boost::regex filter(".*\.dcm");

    std::vector<std::string> files;
    for (fs::directory_iterator i (dir); i!= end_itr; i++){
        if (!fs::is_regular(i->status())) continue;
        boost::smatch what;

        if (!boost::regex_match(i->path().filename().string(),what,filter)) continue;

        //std::cout  << i->path().filename() << std::endl;
        files.push_back(i->path().string());
    }
    sort(files.begin(),files.end());

    return files;
}

int main(int argc, char** argv){

    floatd3 imsize_in_mm;
    float offset;
    uintd3 imageSize;
    int device;
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("dir", po::value<string>()->multitoken())

            ("size,s",po::value<uintd3>(&imageSize)->default_value(uintd3(512,512,1)),"Image size in pixels")
            ("dimensions,d",po::value<floatd3>(&imsize_in_mm)->default_value(floatd3(400,400,500)),"Image dimensions in mm. Overwrites voxelSize.")
            ("device",po::value<int>(&device)->default_value(0),"Number of the device to use (0 indexed)")
            ("offset",po::value<float>(&offset)->default_value(0),"Offset of image");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

    if (vm.count("help")) {
		cout << desc << "\n";
		return 1;
	}


    cudaSetDevice(device);
    cudaDeviceReset();

    auto files = read_dicom_projections(get_dcm_files(vm["dir"].as<string>()));
    std::cout << " Penguins ";
    std::cout << files->geometry.detectorFocalCenterAxialPosition.size() << std::endl;
    std::vector<float>& axials =  files->geometry.detectorFocalCenterAxialPosition;
    std::cout << "Axials size " << axials.size() << std::endl;
    auto mean_offset = std::accumulate(axials.begin(),axials.end(),0.0f)/axials.size();
    std::cout << "Mean offset " << mean_offset << std::endl;
    for ( auto & z : axials)
        z -= mean_offset+offset;

    //for (auto x  : axials)
    //    std::cout << x << std::endl;
    std::cout << "Axials start " << axials[0] << " Axials end " << axials.back() << std::endl;

    std::cout << "Minimum angle " << *std::min_element(files->geometry.detectorFocalCenterAngularPosition.begin(),files->geometry.detectorFocalCenterAngularPosition.end())  << " max " << *std::max_element(files->geometry.detectorFocalCenterAngularPosition.begin(),files->geometry.detectorFocalCenterAngularPosition.end()) << std::endl;

    auto E = boost::make_shared<CTProjectionOperator<cuNDArray>>();
    std::vector<size_t> imdims {imageSize[0],imageSize[1],imageSize[2]};
    cuNDArray<float> image(imdims);

    cuNDArray<float> projections(files->projections );

    std::cout << "Projection memory size " << projections.get_number_of_bytes()/(1024*1024) << "MB" << std::endl;
    E->set_domain_dimensions(image.get_dimensions().get());
    E->set_codomain_dimensions(projections.get_dimensions().get());
    std::cout << "Starting setup" << std::endl;
    E->setup(files,imsize_in_mm);
    std::cout << "Setup done" << std::endl;

    std::cout << "Projections size: " << projections.get_size(0) << " " << projections.get_size(1) << " " << projections.get_size(2) << std::endl;
    //E->mult_MH(&projections,&image,false);
    write_nd_array(&projections,"projections.real");
    //cuGpBbSolver<float> solver;
    //cuNlcgSolver<float> solver;
    cuCgSolver<float> solver;
    //solver.set_non_negativity_constraint(true);
    solver.set_max_iterations(50);
    solver.set_tc_tolerance(1e-8);

    solver.set_encoding_operator(E);
    solver.set_output_mode(cuCgSolver<float>::OUTPUT_VERBOSE);
    auto result = solver.solve(&projections);

    write_nd_array(result.get(),"test.real");

    //fill(result.get(),1.0f);
    auto proj2 = projections;
    E->mult_M(result.get(),&proj2,false);
    write_nd_array(&proj2,"projections2.real");
    float scaling = dot(&proj2,&projections)/dot(&projections,&projections);
    std::cout << "Scaling " << scaling << std::endl;
    proj2 /= scaling;
    proj2 -= projections;
    write_nd_array(&proj2,"projections3.real");

}
