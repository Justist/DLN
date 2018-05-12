#ifndef OVERLOADS_C
#define OVERLOADS_C

#include "includes.hpp"

using std::vector;

inline vector< long double > operator* (const vector< long double >& m1, const vector< long double >& m2) {
   /*  Returns the product of two vectors (elementwise multiplication).
       Inputs:
           m1: vector
           m2: vector
       Output: vector, m1 * m2, product of two vectors m1 and m2
   */
   const unsigned long VECTOR_SIZE = m1.size();
   vector< long double > product(VECTOR_SIZE);
   
   for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
      product[i] = m1[i] * m2[i];
   };
   
   return product;
}

inline vector< long double > operator* (const vector< long double >& m, const long double f) {
   /*  Returns the product of a long double and a vector (elementwise multiplication).
       Inputs:
           m: vector
           f: long double
       Output: vector, m * f, product of vector m and long double f
   */
   const unsigned long VECTOR_SIZE = m.size();
   vector< long double > product(VECTOR_SIZE);
   
   for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
      product[i] = m[i] * f;
   };
   
   return product;
}

inline vector< long double > operator* (const long double f, const vector< long double >& m) {
   /*
    * Front-end for the * operator which takes the long double as the second
    * element. This also allows m * f instead of solely f * m.
    */
   return m * f;
}

inline vector< long double > operator/ (const vector< long double >& m, const long double f) {
   /*  Returns the product of a long double and a vector (elementwise multiplication).
       Inputs:
           m: vector
           f: long double
       Output: vector, m / f, vector m divided bY long double f
   */
   const unsigned long VECTOR_SIZE = m.size();
   vector< long double > product(VECTOR_SIZE);
   
   for (unsigned i = 0; i < VECTOR_SIZE; ++i) {
      product[i] = m[i] / f;
   };
   
   return product;
}

inline vector< long double > operator- (const vector< long double >& m1, const vector< long double >& m2) {
   /*  Returns the difference between two vectors.
       Inputs:
           m1: vector
           m2: vector
       Output: vector, m1 - m2, difference between two vectors m1 and m2.
   */
   const unsigned long VECTOR_SIZE = m1.size();
   vector< long double > difference(VECTOR_SIZE);
   
   for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
      difference[i] = m1[i] - m2[i];
   };
   
   return difference;
}

inline vector< long double > operator- (const vector< long double >& m, const long double f) {
   /*  Returns the difference of a long double and a vector (elementwise subtraction).
       Inputs:
           m: vector
           f: long double
       Output: vector, m - f, containing the difference between each element in
               vector m and long double f
   */
   const unsigned long VECTOR_SIZE = m.size();
   vector< long double > difference(VECTOR_SIZE);
   
   for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
      difference[i] = m[i] - f;
   };
   
   return difference;
}

inline vector< long double > operator- (const long double f, const vector< long double >& m) {
   /*  Returns the difference of a long double and a vector (elementwise subtraction).
       Inputs:
           f: long double
           m: vector
       Output: vector, f - m, containing the difference between long double f and
               each element in vector m
   */
   const unsigned long VECTOR_SIZE = m.size();
   vector< long double > difference(VECTOR_SIZE);
   
   for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
      difference[i] = f - m[i];
   };
   
   return difference;
}

inline vector< long double > operator+ (const vector< long double >& m1, const vector< long double >& m2) {
   /*  Returns the elementwise sum of two vectors.
       Inputs:
           m1: a vector
           m2: a vector
       Output: a vector, sum of the vectors m1 and m2.
   */
   const unsigned long VECTOR_SIZE = m1.size();
   vector< long double > sum(VECTOR_SIZE);
   
   for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
      sum[i] = m1[i] + m2[i];
   };
   
   return sum;
}

inline vector< long double > operator+ (const vector< long double >& m, const long double f) {
   /*  Returns the sum of a long double and a vector (elementwise addition).
       Inputs:
           m: vector
           f: long double
       Output: vector, m + f, containing the sums of each element in vector m
               and long double f
   */
   const unsigned long VECTOR_SIZE = m.size();
   vector< long double > sum(VECTOR_SIZE);
   
   for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
      sum[i] = m[i] + f;
   };
   
   return sum;
}

inline vector< long double > operator+ (const long double f, const vector< long double >& m) {
   /*
    * Front-end for the + operator which takes the long double as the second
    * element. This also allows m + f instead of solely f + m.
    */
   return m + f;
}

#endif
