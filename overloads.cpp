#ifndef OVERLOADS_C
#define OVERLOADS_C

#include "includes.hpp"

using std::vector;

inline vector<float> operator*(const vector<float>& m1, const vector<float>& m2){
    /*  Returns the product of two vectors (elementwise multiplication).
        Inputs:
            m1: vector
            m2: vector
        Output: vector, m1 * m2, product of two vectors m1 and m2
    */
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1[i] * m2[i];
    };
    
    return product;
}

inline vector<float> operator*(const vector<float>& m, const float f){
    /*  Returns the product of a float and a vector (elementwise multiplication).
        Inputs:
            m: vector
            f: float
        Output: vector, m * f, product of vector m and float f
    */
    const unsigned long VECTOR_SIZE = m.size();
    vector<float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m[i] * f;
    };
    
    return product;
}

inline vector<float> operator*(const float f, const vector<float>& m){
    /*
     * Front-end for the * operator which takes the float as the second
     * element. This also allows m * f instead of solely f * m.
     */
    return m * f;
}

inline vector<float> operator/(const vector<float>& m, const float f){
    /*  Returns the product of a float and a vector (elementwise multiplication).
        Inputs:
            m: vector
            f: float
        Output: vector, m / f, vector m divided bY float f
    */
    const unsigned long VECTOR_SIZE = m.size();
    vector<float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i < VECTOR_SIZE; ++i){
        product[i] = m[i] / f;
    };
    
    return product;
}

inline vector<float> operator-(const vector<float>& m1, const vector<float>& m2){
    /*  Returns the difference between two vectors.
        Inputs:
            m1: vector
            m2: vector
        Output: vector, m1 - m2, difference between two vectors m1 and m2.
    */
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = m1[i] - m2[i];
    };
    
    return difference;
}

inline vector<float> operator-(const vector<float>& m, const float f){
    /*  Returns the difference of a float and a vector (elementwise subtraction).
        Inputs:
            m: vector
            f: float
        Output: vector, m - f, containing the difference between each element in 
                vector m and float f
    */
    const unsigned long VECTOR_SIZE = m.size();
    vector<float> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = m[i] - f;
    };
    
    return difference;
}

inline vector<float> operator-(const float f, const vector<float>& m){
    /*  Returns the difference of a float and a vector (elementwise subtraction).
        Inputs:
            f: float
            m: vector
        Output: vector, f - m, containing the difference between float f and 
                each element in vector m
    */
    const unsigned long VECTOR_SIZE = m.size();
    vector<float> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = f - m[i];
    };
    
    return difference;
}

inline vector<float> operator+(const vector<float>& m1, const vector<float>& m2){
    /*  Returns the elementwise sum of two vectors.
        Inputs: 
            m1: a vector
            m2: a vector
        Output: a vector, sum of the vectors m1 and m2.
    */
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> sum (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum[i] = m1[i] + m2[i];
    };
    
    return sum;
}

inline vector<float> operator+(const vector<float>& m, const float f){
    /*  Returns the sum of a float and a vector (elementwise addition).
        Inputs:
            m: vector
            f: float
        Output: vector, m + f, containing the sums of each element in vector m 
                and float f
    */
    const unsigned long VECTOR_SIZE = m.size();
    vector<float> sum (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum[i] = m[i] + f;
    };
    
    return sum;
}

inline vector<float> operator+(const float f, const vector<float>& m){
    /*
     * Front-end for the + operator which takes the float as the second
     * element. This also allows m + f instead of solely f + m.
     */
    return m + f;
}

#endif
