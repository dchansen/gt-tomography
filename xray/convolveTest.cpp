#include "hoCuNDArray_utils.h"
#include "radial_utilities.h"
#include "hoNDArray_fileio.h"
#include "cuNDArray_fileio.h"
#include "cuNDArray.h"
#include "cuNDArray_math.h"
#include "cuNDArray_utils.h"
#include "imageOperator.h"
#include "identityOperator.h"
#include "hoPartialDerivativeOperator.h"
#include "hoCuConebeamProjectionOperator.h"
#include "cuConvolutionOperator.h"
#include "hoCuIdentityOperator.h"
#include "hoCuNDArray_math.h"
#include "hoCuNDArray_blas.h"
#include "hoCuCgSolver.h"
#include "CBCT_acquisition.h"
#include "complext.h"
#include "encodingOperatorContainer.h"
#include "vector_td_io.h"
#include "hoCuPartialDerivativeOperator.h"
#include "GPUTimer.h"
#include "FFTRebinning.h"

#include <iostream>
#include <algorithm>
#include <sstream>
#include <math_constants.h>
#include <boost/program_options.hpp>
#include <boost/make_shared.hpp>
#include "cuNFFT.h"
#include "hoNDArray_math.h"

using namespace std;
using namespace Gadgetron;

typedef float REAL;
typedef cuNFFT_plan<REAL,2> PLAN;
namespace po = boost::program_options;

boost::shared_ptr<cuNDArray<vector_td<REAL,2>>> make_points(std::vector<size_t> & dims){
	hoNDArray<vector_td<REAL,2>> points({dims[0]*dims[1]});
	vector_td<REAL,2> * data = points.get_data_ptr();

	for (int n = 0; n < dims[1]; n++)
		for (int m = 0; m < dims[0]; m++)
			data[m+n*dims[0]] = vector_td<REAL,2>(REAL(m)/(dims[0]-1)-0.5f,REAL(n)/(dims[1]-1)-0.5f);

	auto result = boost::make_shared<cuNDArray<vector_td<REAL,2>>>(points);
	return result;

}

int
main(int argc, char** argv)
{
  string outputFile;
  string inputFile;


  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("input,a", po::value<string>(&inputFile)->default_value("lena.real"),"Input filename")
    ("output,f", po::value<string>(&outputFile)->default_value("reconstruction.real"), "Output filename")
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
    else if (a.type() == typeid(REAL)) std::cout << it->second.as<REAL>();
    else if (a.type() == typeid(vector_td<REAL,3>)) std::cout << it->second.as<vector_td<REAL,3> >();
    else if (a.type() == typeid(vector_td<int,3>)) std::cout << it->second.as<vector_td<int,3> >();
    else if (a.type() == typeid(vector_td<unsigned int,3>)) std::cout << it->second.as<vector_td<unsigned int,3> >();
    else std::cout << "Unknown type" << std::endl;
    std::cout << std::endl;
  }
  cudaDeviceReset();

  //Really weird stuff. Needed to initialize the device?? Should find real bug.
  cudaDeviceManager::Instance()->lockHandle();
  cudaDeviceManager::Instance()->unlockHandle();

  auto input = cuNDArray<float>(*read_nd_array<float>(inputFile.c_str()));
  //auto rInput = convert_to<float,REAL>(&input);
  auto cuInput = real_to_complex<complext<REAL>>(&input);
  auto points = make_points(*input.get_dimensions());

  PLAN plan;

  auto dims = from_std_vector<size_t,2>(*input.get_dimensions());
  plan.setup(dims,dims*size_t(2),7.5);
  plan.preprocess(points.get(),PLAN::NFFT_PREP_ALL);

  auto output = *cuInput;
  clear(&output);

  cuInput->reshape(std::vector<size_t>{cuInput->get_number_of_elements()});
  /*
  std::vector<size_t> dims_oversampled{cuInput->get_size(0)*2,cuInput->get_size(1)*2};
  output = cuNDArray<float_complext>(dims_oversampled);

  mirrorpad<float_complext,2>(cuInput.get(),&output);
  */
  //output.reshape(std::vector<size_t>{cuInput->get_number_of_elements()});

  plan.compute(cuInput.get(),&output,nullptr,PLAN::NFFT_FORWARDS_NC2C);
  //output.reshape(cuInput->get_dimensions());
  //cuNDFFT<REAL>::instance()->fft(&output);
  clear(cuInput.get());
  plan.compute(&output,cuInput.get(),nullptr,PLAN::NFFT_BACKWARDS_C2NC);
  output = *cuInput;

/*
  plan.deapodize(cuInput.get());
  cuNDFFT<REAL>::instance()->fft(cuInput.get());
  plan.convolve(cuInput.get(),&output,nullptr,PLAN::NFFT_CONV_C2NC,false);

  output.reshape(cuInput->get_dimensions());
  cuNDFFT<REAL>::instance()->ifft(&output);
  */
/*
  plan.convolve(cuInput.get(),&output,nullptr,PLAN::NFFT_CONV_NC2C,false);

  cuNDFFT<REAL>::instance()->fft(&output);
  plan.deapodize(&output,true);
  cuNDFFT<REAL>::instance()->ifft(&output);
  */
  //plan.convolve_NFFT_NC2C(&cuInput,&output,false);
  //plan.deapodize(&cuInput);
  /*
  cuNDFFT<REAL>::instance()->fft(&cuInput);
  output = cuInput;
  //plan.convolve(&cuInput,&output,nullptr,PLAN::NFFT_CONV_C2NC);

  cuNDFFT<REAL>::instance()->ifft(&output);
*/

/*
  plan.compute(&cuInput,&output,nullptr,PLAN::NFFT_FORWARDS_C2NC);
  //cuNDFFT<REAL>::instance()->ifft(&output);

  std::cout << "Output nrm " << nrm2(&output) << std::endl;
  clear(&cuInput);
  plan.compute(&output,&cuInput,nullptr,PLAN::NFFT_BACKWARDS_NC2C);

  std::cout << "Output nrm " << nrm2(&cuInput) << std::endl;
  //cuNDFFT<REAL>::instance()->ifft(&output);

  output = cuInput;
*/
  write_nd_array<REAL>( abs(&output).get(), outputFile.c_str());
}
