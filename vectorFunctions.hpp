#ifndef VECTORFUNCTIONS_H
#define VECTORFUNCTIONS_H

#include "includes.hpp"

using std::vector;
using std::cout;
using std::endl;

class VectorFunctions {
private:

public:
   VectorFunctions () = default;
   
   ~VectorFunctions () = default;
   
   long double vectorsum (vector< long double >);
   
   vector< long double > epower (vector< long double >);
   
   vector< long double > transpose (const long double *, int, int);
   
   vector< long double > softmax (vector< long double >);
   
   vector< long double > sigmoid_d (vector< long double >);
   
   long double sigmoid_d (long double);
   
   vector< long double > sigmoid (vector< long double >);
   
   vector< long double > dot (vector< long double >, vector< long double >,
                         int, int, int);
                         
   long double crossentropy_d(long double, long double, bool);
   
   long double crossEntropy (vector< long double >, vector< long double >,
                        vector< long double >, bool);
   
   long double meanSquaredError (vector< long double >, vector< long double >);
   
   long double sigmoid (const long double x);
};

#endif
