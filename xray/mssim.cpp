//
// Created by dch on 13/03/17.
//

#include "mssim.h"
#include <cuNDArray_math.h>
#include "cuGaussianFilterOperator.h"
#include <vector_td_operators.h>
#include <cuNDArray_fileio.h>

using namespace Gadgetron;


static cuNDArray<float> CS(cuNDArray<float>* X, cuNDArray<float> * Y, floatd3 sigma){
    cuGaussianFilterOperator<float,3> gauss;
    gauss.set_sigma(sigma);
    float K2 = 0.03;

    float data_range = max(Y)-min(Y);

    cuNDArray<float> ux(X->get_dimensions());
    gauss.mult_M(X,&ux);

    cuNDArray<float> uy(Y->get_dimensions());
    gauss.mult_M(Y,&uy);


    cuNDArray<float> tmp(*X);
    tmp *= *X;
    cuNDArray<float> XX(Y->get_dimensions());
    gauss.mult_M(&tmp,&XX);

    tmp = *Y;
    tmp *= *Y;

    cuNDArray<float> YY(Y->get_dimensions());
    gauss.mult_M(&tmp,&YY);

    tmp = *Y;
    tmp *= *X;


    cuNDArray<float> YX(Y->get_dimensions());
    gauss.mult_M(&tmp,&YX);


    cuNDArray<float> uxuy = ux;
    uxuy *= uy;

    ux *= ux;
    uy *= uy;

    XX -= ux;

    YY -= uy;

    YX -= uxuy;

    float c2 = std::pow(K2*data_range,2);

    YX *= float(2);
    YX += c2;

    XX += YY;
    XX += c2;


    YX /= XX;

    return YX;



}


float Gadgetron::mssim(cuNDArray<float> *image, cuNDArray<float> *reference, floatd3 sigma, int scales) {
    cuNDArray<float> S(image->get_dimensions());
    fill(&S,1.0f);

    for (int i = 0; i < scales; i++){

        auto tmp = CS(image,reference,sigma*float(std::pow(2.0f,i)));
        S *= tmp;
    }


    cuGaussianFilterOperator<float,3> gauss;
    gauss.set_sigma(sigma*float(std::pow(2.0f,scales-1)));

    float K1 = 0.01;



    float data_range = max(reference)-min(reference);
    float c1 = std::pow(K1*data_range,2);
    cuNDArray<float> ux(image->get_dimensions());
    gauss.mult_M(image,&ux);


    cuNDArray<float> uy(image->get_dimensions());
    gauss.mult_M(reference,&uy);


    cuNDArray<float> A(ux);
    A *= uy;
    A *= float(2);
    A += c1;

    ux *= ux;
    uy *= uy;
    ux += uy;
    ux += c1;

    S *= A;
    S /= ux;

    return mean(&S);


}
