//
// Created by dch on 18/02/16.
//

#include <boost/program_options.hpp>
#include "CT_acquisition.h"

using namespace std;
using namespace Gadgetron;

namespace po = boost::program_options;

int main(int argc, char** argv){

    po::options_description desc("Allowed options");
    desc.add_options()
            ("files", po::value<vector<string>>());

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

    auto files = read_dicom_projections(vm["files"].as<vector<string>>());

}
