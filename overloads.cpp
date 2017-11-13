#ifndef OVERLOADS_C
#define OVERLOADS_C

#include "includes.hpp"

using std::vector;

inline vector<double> operator*(const vector<double>& m1, const vector<double>& m2){
    /*  Returns the product of two vectors (elementwise multiplication).
        Inputs:
            m1: vector
            m2: vector
        Output: vector, m1 * m2, product of two vectors m1 and m2
    */
    const unsigned long VECTOR_SIZE = m1.size();
    vector <double> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1[i] * m2[i];
    };
    
    return product;
}

inline vector<double> operator*(const vector<double>& m, const double f){
    /*  Returns the product of a double and a vector (elementwise multiplication).
        Inputs:
            m: vector
            f: double
        Output: vector, m * f, product of vector m and double f
    */
    const unsigned long VECTOR_SIZE = m.size();
    vector<double> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m[i] * f;
    };
    
    return product;
}

inline vector<double> operator*(const double f, const vector<double>& m){
    /*
     * Front-end for the * operator which takes the double as the second
     * element. This also allows m * f instead of solely f * m.
     */
    return m * f;
}

inline vector<double> operator/(const vector<double>& m, const double f){
    /*  Returns the product of a double and a vector (elementwise multiplication).
        Inputs:
            m: vector
            f: double
        Output: vector, m / f, vector m divided bY double f
    */
    const unsigned long VECTOR_SIZE = m.size();
    vector<double> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i < VECTOR_SIZE; ++i){
        product[i] = m[i] / f;
    };
    
    return product;
}

inline vector<double> operator-(const vector<double>& m1, const vector<double>& m2){
    /*  Returns the difference between two vectors.
        Inputs:
            m1: vector
            m2: vector
        Output: vector, m1 - m2, difference between two vectors m1 and m2.
    */
    const unsigned long VECTOR_SIZE = m1.size();
    vector <double> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = m1[i] - m2[i];
    };
    
    return difference;
}

inline vector<double> operator-(const vector<double>& m, const double f){
    /*  Returns the difference of a double and a vector (elementwise subtraction).
        Inputs:
            m: vector
            f: double
        Output: vector, m - f, containing the difference between each element in 
                vector m and double f
    */
    const unsigned long VECTOR_SIZE = m.size();
    vector<double> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = m[i] - f;
    };
    
    return difference;
}

inline vector<double> operator-(const double f, const vector<double>& m){
    /*  Returns the difference of a double and a vector (elementwise subtraction).
        Inputs:
            f: double
            m: vector
        Output: vector, f - m, containing the difference between double f and 
                each element in vector m
    */
    const unsigned long VECTOR_SIZE = m.size();
    vector<double> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = f - m[i];
    };
    
    return difference;
}

inline vector<double> operator+(const vector<double>& m1, const vector<double>& m2){
    /*  Returns the elementwise sum of two vectors.
        Inputs: 
            m1: a vector
            m2: a vector
        Output: a vector, sum of the vectors m1 and m2.
    */
    const unsigned long VECTOR_SIZE = m1.size();
    vector <double> sum (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum[i] = m1[i] + m2[i];
    };
    
    return sum;
}

inline vector<double> operator+(const vector<double>& m, const double f){
    /*  Returns the sum of a double and a vector (elementwise addition).
        Inputs:
            m: vector
            f: double
        Output: vector, m + f, containing the sums of each element in vector m 
                and double f
    */
    const unsigned long VECTOR_SIZE = m.size();
    vector<double> sum (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum[i] = m[i] + f;
    };
    
    return sum;
}

inline vector<double> operator+(const double f, const vector<double>& m){
    /*
     * Front-end for the + operator which takes the double as the second
     * element. This also allows m + f instead of solely f + m.
     */
    return m + f;
}

#endif
