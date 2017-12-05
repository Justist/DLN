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
   
   double vectorsum (vector< double >);
   
   vector< double > epower (vector< double >);
   
   vector< double > transpose (const double *, int, int);
   
   vector< double > softmax (vector< double >);
   
   vector< double > sigmoid_d (vector< double >);
   
   double sigmoid_d (double);
   
   vector< double > sigmoid (vector< double >);
   
   vector< double > dot (vector< double >, vector< double >,
                         int, int, int);
   
   double crossEntropy (vector< double >, vector< double >,
                        vector< double >, bool);
   
   double meanSquaredError (vector< double >, vector< double >);
};

#endif
