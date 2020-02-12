#pragma once

#include "hoCuCgSolver.h"
#include "sbcSolver.h"

#include "complext.h"

namespace Gadgetron{

  template <class T> class hoCuSbcCgSolver : public sbcSolver< hoCuNDArray<typename realType<T>::Type >, hoCuNDArray<T>, hoCuCgSolver<T> >
  {
  public:
    hoCuSbcCgSolver() : sbcSolver<hoCuNDArray<typename realType<T>::Type >, hoCuNDArray<T>, hoCuCgSolver<T> >(),_it(0) {}
    virtual ~hoCuSbcCgSolver() {}
    /*
    virtual bool post_linear_solver_callback( hoCuNDArray<T>* x) {

    	std::stringstream ss;
			ss << "iteration-" << _it << ".real";
			write_nd_array(x,ss.str().c_str());
			_it++;
			return true;

    }
    */
  private:
     int _it;
  };
}
