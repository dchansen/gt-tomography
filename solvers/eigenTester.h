#pragma once
#include "identityOperator.h"

namespace Gadgetron{
template <class ARRAY_TYPE> class eigenTester {
private:
	typedef typename ARRAY_TYPE::element_type ELEMENT_TYPE;
		typedef typename realType<ELEMENT_TYPE>::Type REAL;
		typedef ARRAY_TYPE ARRAY_CLASS;
public:
  eigenTester(){
	  tolerance = REAL(1e-8);
	  id_operator = boost::shared_ptr<identityOperator<ARRAY_TYPE> >(new identityOperator<ARRAY_TYPE>);
  }
  virtual ~eigenTester(){}

  ELEMENT_TYPE get_dominant_eigenvalue(){
	  boost::shared_ptr<ARRAY_TYPE> eigenVector = get_dominant_eigenvector();
	  return get_eigenvalue_from_vector(eigenVector.get());
  }


  ELEMENT_TYPE get_smallest_eigenvalue(ELEMENT_TYPE dominant_eigenvalue){
	  ELEMENT_TYPE beta = dominant_eigenvalue*2;
	  id_operator->set_weight(-beta);

	  regularization_operators_.push_back(id_operator);
	  std::cout << "ID operator weight " << id_operator->get_weight() << std::endl;
	  ELEMENT_TYPE eig1 = get_dominant_eigenvalue();
	  regularization_operators_.pop_back();
	  return eig1+beta;


  }
  ELEMENT_TYPE get_smallest_eigenvalue(){
  	  ELEMENT_TYPE eig = get_dominant_eigenvalue();
  	  return get_smallest_eigenvalue(eig);
    }
  // Add encoding operator to solver (only one allowed)
   inline bool set_encoding_operator( boost::shared_ptr< linearOperator< ARRAY_TYPE> > op)
   {
     if( !op.get() ){
       std::cout << "Error: linearSolver::add_matrix_operator : NULL operator provided" << std::endl;
       return false;
     }

     encoding_operator_ = op;

     return true;
   }
   inline void set_tolerance(REAL tolerance){
	   this->tolerance = tolerance;
   }
   // Add linear operator to solver (in addition to the encoding operator)
   inline bool add_linear_operator( boost::shared_ptr< linearOperator< ARRAY_TYPE> > op)
     {
       if( !op.get() ){
    	   std::cout << "Error: linearSolver::add_matrix_operator : NULL operator provided"  << std::endl;
         return false;
       }

       regularization_operators_.push_back(op);

       return true;
     }
	protected:
	 bool mult_MH_M( ARRAY_TYPE *in, ARRAY_TYPE *out )
	  {
	    // Basic validity checks
	    if( !in || !out ){
	      std::cout << "Error: cgSolver::mult_MH_M : invalid input pointer(s)" << std::endl;
	      return false;
	    }
	    if( in->get_number_of_elements() != out->get_number_of_elements() ){
	    	std::cout << "Error: cgSolver::mult_MH_M : array dimensionality mismatch"<< std::endl;
	      return false;
	    }

	    // Intermediate storage
	    ARRAY_TYPE q(in->get_dimensions());
	    clear(out);

	    //Use encoding operator

	    this->encoding_operator_->mult_MH_M( in, &q, false );
	    axpy( this->encoding_operator_->get_weight(), &q, out );


	    // Iterate over regularization operators
	    for( unsigned int i=0; i<this->regularization_operators_.size(); i++ ){

	      this->regularization_operators_[i]->mult_MH_M( in, &q, false );
	      axpy( this->regularization_operators_[i]->get_weight(), &q, out );

	    }

	    return true;
	  }
	 ELEMENT_TYPE get_eigenvalue_from_vector(ARRAY_TYPE* eigenVector){
		 ARRAY_TYPE out(*eigenVector);
		 clear(&out);
		 mult_MH_M(eigenVector,&out);
		 ELEMENT_TYPE dom1 = (*eigenVector)[amax(eigenVector)];
		 ELEMENT_TYPE dom2 = out[amax(&out)];
		 return dom2/dom1;

	 }

	  boost::shared_ptr<ARRAY_TYPE> get_dominant_eigenvector(){
		  std::cout << "Starting dominant eigenvector calculations "<< tolerance << std::endl;
		  ELEMENT_TYPE norm = ELEMENT_TYPE(1);
		  ELEMENT_TYPE norm_old = ELEMENT_TYPE(2);

		  ARRAY_TYPE* in = new ARRAY_TYPE;
		  std::vector<unsigned int> image_dims = *this->encoding_operator_->get_domain_dimensions();

		  in->create(&image_dims);

		  fill(in,ELEMENT_TYPE(1));

		  ARRAY_TYPE* out = new ARRAY_TYPE;
		  out->create(&image_dims);

		  while (fabs(norm-norm_old)/norm> tolerance){
			  norm_old=norm;
			  mult_MH_M(in,out);
			  norm = nrm2(out);
			  *out /= norm;
			  ARRAY_TYPE* tmp = in;
			  in = out;
			  out = tmp;
			  std::cout << "Relative change in eigenvalue: " << fabs(norm-norm_old)/norm << std::endl;


			  }
		  delete in;
		  return boost::shared_ptr<ARRAY_TYPE>(out);
		}



	protected:

	  // Single encoding operator
	  boost::shared_ptr< identityOperator<ARRAY_TYPE> > id_operator;
	  boost::shared_ptr< linearOperator< ARRAY_TYPE> > encoding_operator_;
	  REAL tolerance;
	  // Vector of linear regularization operators
	  std::vector< boost::shared_ptr< linearOperator< ARRAY_TYPE> > > regularization_operators_;

};
}
