//
// Created by dch on 18/02/16.
//

#include <boost/program_options.hpp>
#include <cuNDArray.h>
#include <cuNDArray_fileio.h>
#include "CT_acquisition.h"
#include "cuCTProjectionOperator.h"

using namespace std;
using namespace Gadgetron;

namespace po = boost::program_options;

int main(int argc, char** argv){

    floatd3 imsize_in_mm;
    uintd3 imageSize;
    po::options_description desc("Allowed options");
    desc.add_options()
            ("files", po::value<vector<string>>()->multitoken())

            ("size,s",po::value<uintd3>(&imageSize)->default_value(uintd3(512,512,1)),"Image size in pixels")
            ("dimensions,d",po::value<floatd3>(&imsize_in_mm)->default_value(floatd3(400,400,100)),"Image dimensions in mm. Overwrites voxelSize.");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

    auto files = read_dicom_projections(vm["files"].as<vector<string>>());
    std::cout << " Penguins ";
    std::cout << files->geometry.detectorFocalCenterAxialPosition.size() << std::endl;
    std::vector<float> axials =  files->geometry.detectorFocalCenterAxialPosition;
    std::cout << "Axials size " << axials.size() << std::endl;
    for (auto x  : axials)
        std::cout << x << std::endl;

    cuCTProjectionOperator E;
    E.setup(files,imsize_in_mm);

    cuNDArray<float> projections(files->projections );

    std::vector<size_t> imdims {imageSize[0],imageSize[1],imageSize[2]};
    cuNDArray<float> image(imdims);

    E.set_domain_dimensions(image.get_dimensions().get());
    E.set_codomain_dimensions(projections.get_dimensions().get());

    E.mult_MH(&projections,&image,false);

    write_nd_array(&image,"test.real");

}
