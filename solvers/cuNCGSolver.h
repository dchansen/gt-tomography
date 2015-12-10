#pragma once

#include "solver_utils.h"
#include "ncgSolver.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "cuNDArray_blas.h"
#include "real_utilities.h"
#include "vector_td_utilities.h"
#include "gpusolvers_export.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include <fstream>

#include "hoNDArray_fileio.h"
namespace Gadgetron{
  
  template <class T> class EXPORTGPUSOLVERS cuNCGSolver : public ncgSolver<cuNDArray<T> >
  {
  public:
    
    cuNCGSolver() : ncgSolver<cuNDArray<T> >() {}
    virtual ~cuNCGSolver() {}
    

    /*
    virtual void iteration_callback(cuNDArray<T>* x ,int iteration,typename realType<T>::Type value){
     	  if (iteration == 0){
     		  std::ofstream textFile("residual.txt",std::ios::trunc);
     	  	  textFile << value << std::endl;
     	  } else{
     		  std::ofstream textFile("residual.txt",std::ios::app);
     		  textFile << value << std::endl;
     	  }

     	  std::stringstream ss;
     	  ss << "NCG-" << iteration << ".real";
     	 write_nd_array(x->to_host().get(),ss.str().c_str());


       };
       */
  };
}
