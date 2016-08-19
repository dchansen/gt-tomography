#pragma once


namespace Gadgetron {
    template<class ARRAY>
    class primalDualOperator {

    public:
        typedef typename ARRAY::element_type ELEMENT_TYPE;
        typedef typename realType<ELEMENT_TYPE>::Type REAL;
        virtual void primalDual(ARRAY* in,ARRAY *out, REAL sigma = 0, bool accumulate = false)=0;




    };
}