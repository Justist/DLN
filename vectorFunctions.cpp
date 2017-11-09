#include "vectorFunctions.hpp"

float VectorFunctions::vectorsum(const vector<float> vec) {
   /*
    * Return the sum of the elements of a vector vec.
    * Input:
    *    vec: vector of floats
    * Output:
    *    sum, sum of the elements in given vector vec.
    */
   int sum = 0;
   for (auto n : vec) {sum += n;}
   return sum;
}

vector<float> VectorFunctions::transpose (float *m, const int C, const int R) {
    /*  Returns a transpose matrix of input matrix.
     *  Inputs:
     *      m: vector, input matrix
     *      C: int, number of columns in the input matrix
     *      R: int, number of rows in the input matrix
     *  Output: vector, transpose matrix mT of input matrix m
     */
    vector<float> mT (C*R);
    
    for(int n = 0; n!=C*R; n++) {
        int i = n/C;
        int j = n%C;
        mT[n] = m[R*j + i];
    }
    
    return mT;
}

vector<float> VectorFunctions::softmax(const vector<float> vec) {
    /*
     * Return the softmaxed version of given vector vec.
     * Input:
     *    vec: vector of floats
     * Output:
     *    smvec, softmaxed version of vec
     */
     vector<float> smvec;
     double maxelem = *std::max_element(std::begin(vec), std::end(vec));
     for(auto v : vec) {
        smvec.push_back(std::exp(v - maxelem));
     }
     return smvec;
}

vector<float> VectorFunctions::sigmoid_d(const vector<float>& m1) {
    /*  Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)), 
        where f(x) is sigmoid function.
        Input: m1, a vector.
        Output: x(1 - x) for every element of the input matrix m1.
    */
    const unsigned long VECTOR_SIZE = m1.size();
    vector<float> output (VECTOR_SIZE);
    
    for(unsigned i = 0; i < VECTOR_SIZE; i++) {
        output[i] = m1[i] * (1 - m1[i]);
    }
    
    return output;
}

vector<float> VectorFunctions::sigmoid(const vector<float>& m1) {
    /*  Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: 1/(1 + e^-x) for every element of the input matrix m1.
    */
    const unsigned long VECTOR_SIZE = m1.size();
    vector<float> output (VECTOR_SIZE);
    
    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[i] = 1 / (1 + exp(-m1[i]));
    }
    
    return output;
}

vector<float> VectorFunctions::dot (const vector<float>& m1, 
                                    const vector<float>& m2, 
                                    const int m1_rows, 
                                    const int m1_columns, 
                                    const int m2_columns) {
    /*  Returns the product of two matrices: m1 x m2.
     *  Inputs:
     *      m1: vector, left matrix of size m1_rows x m1_columns
     *      m2: vector, right matrix of size m1_columns x m2_columns 
     *          (the number of rows in the right matrix must be equal 
     *          to the number of the columns in the left one)
     *      m1_rows: int, number of rows in the left matrix m1
     *      m1_columns: int, number of columns in the left matrix m1
     *      m2_columns: int, number of columns in the right matrix m2
     *  Output: vector, m1 * m2, product of two vectors m1 and m2, 
     *          a matrix of size m1_rows x m2_columns
     */
    vector<float> output(m1_rows*m2_columns);
    
    for(int row = 0; row < m1_rows; row++) {
        for(int col = 0; col < m2_columns; col++) {
            output[row * m2_columns + col] = 0.f;
            for(int k = 0; k < m1_columns; k++) {
                output[row * m2_columns + col] += m1[row * m1_columns + k] * m2[k * m2_columns + col];
            }
        }
    }
    
    return output;
}

float VectorFunctions::crossEntropy(const vector<float> output, 
                                    const vector<float> labels) {
   /*  Returns the cross entropy between two vectors.
    *   Inputs:
    *       output: vector
    *       labels: vector
    *   Output: float, -sum(labels * log(output)), 
    *           cross entropy between output and labels.
    */
    if(output.size() != labels.size()) {
        std::cerr << "Labels and output not of same length!" << endl;
        std::cerr << "Labels length: " << labels.size() << endl;
        std::cerr << "Output length: " << output.size() << endl;
        exit(1);
    }
    const unsigned long VECTOR_SIZE = output.size();
    float crossent = 0.0;
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        crossent += labels[i] * log(output[i]);
    };
    return -crossent;
}
