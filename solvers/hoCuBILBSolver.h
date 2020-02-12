#pragma once

#include "BILBSolver.h"
//#include "hoNDArray_fileio.h"
//#include <sstream>

namespace Gadgetron{

template<class ARRAY_TYPE> class hoCuBILBSolver : public BILBSolver<ARRAY_TYPE>{

	typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
	typedef typename realType<ELEMENT_TYPE>::Type REAL;
public:
	virtual ~hoCuBILBSolver(){};
protected:
	virtual void solver_non_negativity_filter(ARRAY_TYPE *xdata,ARRAY_TYPE *gdata){
		ELEMENT_TYPE* x = xdata->get_data_ptr();
		ELEMENT_TYPE* g = gdata->get_data_ptr();
		for (int i = 0; i != xdata->get_number_of_elements(); ++i)
			if ( real(x[i]) <= REAL(0) && real(g[i]) > 0) g[i]=ELEMENT_TYPE(0);
	}
};
}
