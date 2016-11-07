#pragma once
#include <boost/shared_ptr.hpp>

namespace Gadgetron {
    template<class ARRAY>
    class primalDualOperator {

    public:
        typedef typename ARRAY::element_type ELEMENT_TYPE;
        typedef typename realType<ELEMENT_TYPE>::Type REAL;
        virtual void primalDual(ARRAY* in,ARRAY *out, REAL sigma = 0, bool accumulate = false)=0;

        virtual void set_weight(REAL weight_){ weight = weight_;}
        virtual REAL get_weight(){ return weight;}

        primalDualOperator() : weight(1) {  };

        virtual void update_weights(ARRAY* x){};

    protected:
        REAL weight;
        boost::shared_ptr<ARRAY> weight_arr;


    };
}