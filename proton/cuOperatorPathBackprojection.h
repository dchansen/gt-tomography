#pragma once
#include "cuNDArray_operators.h"
#include "linearOperator.h"
#include "cuNDArray.h"
#include <vector>
#include "GPUTimer.h"


namespace Gadgetron{
template<class REAL>
class cuOperatorPathBackprojection : public linearOperator<cuNDArray<REAL> > {
 public:
 cuOperatorPathBackprojection() : linearOperator<cuNDArray<REAL> >() {
	 rescale_dirs = true;
	 cutoff = 0.02f;

    }
    virtual ~cuOperatorPathBackprojection() {}

    virtual void mult_M( cuNDArray<REAL>* in, cuNDArray<REAL>* out, bool accumulate = false );
    virtual void mult_MH( cuNDArray<REAL>* in, cuNDArray<REAL>* out, bool accumulate = false );
    virtual void mult_MH_M( cuNDArray<REAL>* in, cuNDArray<REAL>* out, bool accumulate = false );

    virtual void setup(boost::shared_ptr< cuNDArray<vector_td<REAL,3> > > splines, vector_td<REAL,3> physical_dims,boost::shared_ptr< cuNDArray< REAL > > projections, vector_td<REAL,3> origin, std::vector<size_t> img_dims, bool use_hull, REAL offset=0);
    virtual void setup(boost::shared_ptr< cuNDArray<vector_td<REAL,3> > > splines, vector_td<REAL,3> physical_dims, boost::shared_ptr< cuNDArray< REAL > > projections, boost::shared_ptr< cuNDArray<REAL> > weights,vector_td<REAL,3> origin, std::vector<size_t> img_dims, bool use_hull, REAL offset = 0);


    virtual void set_codomain_dimensions( std::vector<size_t> *dims ){
    	throw std::runtime_error("cuOperatorPathBackprojection::codomain dimension must be set through the setup function");
    }
    virtual boost::shared_ptr<cuNDArray<REAL> > calc_hull( cuNDArray<REAL>* projections, std::vector<size_t>& img_dims );

    boost::shared_ptr<cuNDArray<REAL> > get_hull(){ return hull_mask;}
    virtual boost::shared_ptr< linearOperator< cuNDArray<REAL> > > clone() {
       return linearOperator< cuNDArray<REAL> >::clone(this);
     }
    void rescale_directions(bool val){rescale_dirs=val;}
 protected:
    boost::shared_ptr< cuNDArray<vector_td<REAL,3> > > splines;

    boost::shared_ptr< cuNDArray<REAL> > weights;
    vector_td<REAL,3> physical_dims;
    vector_td<REAL,3> origin;

    bool rescale_dirs;
    REAL cutoff;

    boost::shared_ptr<cuNDArray<REAL> > hull_mask;

    REAL background;

};
}
