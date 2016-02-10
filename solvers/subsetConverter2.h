/*
 * subsetConverter.h
 *
 *  Created on: Dec 10, 2015
 *      Author: dch
 */

#ifndef SUBSETCONVERTER_H_
#define SUBSETCONVERTER_H_

#include "subsetOperator.h"

namespace Gadgetron {
template<class ARRAY_TYPE> class subsetConverter: public subsetOperator<ARRAY_TYPE>{
private:
  typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
  typedef typename realType<ELEMENT_TYPE>::Type REAL;
public:

	subsetConverter(boost::shared_ptr<linearOperator<ARRAY_TYPE> > op_) : op(op_),subsetOperator<ARRAY_TYPE>(){};

	virtual ~subsetConverter(){};
	virtual void mult_M(ARRAY_TYPE* in, ARRAY_TYPE* out, int subset, bool accumulate){
		op->mult_M(in,out,accumulate);
	}
	virtual void mult_MH(ARRAY_TYPE* in, ARRAY_TYPE* out, int subset, bool accumulate){
		op->mult_MH(in,out,accumulate);
	}
	virtual void mult_MH_M(ARRAY_TYPE* in, ARRAY_TYPE* out, int subset, bool accumulate){
		op->mult_MH_M(in,out,accumulate);
	}


	virtual boost::shared_ptr< std::vector<size_t> > get_codomain_dimensions(int subset){
		return op->get_codomain_dimensions();
	}
/*
 	virtual void set_codomain_subsets(std::vector< std::vector<unsigned int> > & _dims){
		codomain_dimensions = std::vector< std::vector<unsigned int> >(_dims);
	}
*/
	int get_number_of_subsets(){return number_of_subsets;}

protected:
	boost::shared_ptr<linearOperator<ARRAY_TYPE> > op;
	int number_of_subsets;

};

}

#endif /* SUBSETCONVERTER_H_ */
