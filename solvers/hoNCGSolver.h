#pragma once

#include "ncgSolver.h"
#include "hoNDArray_math.h"
#include "hoNDArray_fileio.h"
#include "complext.h"

namespace Gadgetron{

template<class T> class hoNCGSolver: public ncgSolver<hoNDArray<T> >{
	typedef typename realType<T>::Type REAL;
public:
	hoNCGSolver():ncgSolver<hoNDArray<T> >(){

	}

	virtual ~hoNCGSolver(){};

protected:
  virtual void solver_non_negativity_filter(hoNDArray<T> *x,hoNDArray<T> *g)
  {
    T* x_ptr = x->get_data_ptr();
    T* g_ptr = g->get_data_ptr();
    for (int i =0; i < x->get_number_of_elements(); i++){
      if ( real(x_ptr[i]) <= REAL(0) && real(g_ptr[i]) > 0) g_ptr[i]=T(0);
    }

  }

  virtual void iteration_callback(hoNDArray<T>* x,int i,REAL data_res,REAL reg_res){
//	  if (i == 0){
//		  std::ofstream textFile("residual.txt",std::ios::trunc);
//	  	  textFile << data_res << std::endl;
//	  } else{
//		  std::ofstream textFile("residual.txt",std::ios::app);
//		  textFile << data_res << std::endl;
//	  }
//	  std::stringstream ss;
//	  ss << "iteration-" << i << ".real";
//	  write_nd_array(x,ss.str().c_str());
  };
};
}
