#include "vectorFunctions.hpp"
#include "overloads.cpp"

long double VectorFunctions::vectorsum (const vector< long double > vec) {
   /*
    * Return the sum of the elements of a vector vec.
    * Input:
    *    vec: vector of floats
    * Output:
    *    sum, sum of the elements in given vector vec.
    */
   int sum = 0;
   for (auto n : vec) { sum += n; }
   return sum;
}

vector< long double > VectorFunctions::epower (const vector< long double > m) {
   /*
    * Returns a vector of float which is the given vector of floats but with
    * each element x being a power of e, such that every element x in the
    * given vector is an element e^x in the output vector.
    * Input:
    *    m: vector of floats
    * Output:
    *    vector of floats which are all e-powers
    */
   vector< long double > output;
   long double y;
   for (long double x : m) {
      y = exp(x);
      if (y != y) {
         std::cout << "in epower gaat het fout!" << std::endl;
         exit(0);
      }
      output.push_back(std::exp(x));
   }
   return output;
}

vector< long double > VectorFunctions::transpose (const long double *m, 
                                                  const int C, 
                                                  const int R) {
   /*
    *  Returns a transpose matrix of input matrix.
    *  Inputs:
    *      m: vector, input matrix
    *      C: int, number of columns in the input matrix
    *      R: int, number of rows in the input matrix
    *  Output: vector, transpose matrix mT of input matrix m
    */
   vector< long double > mT(C * R);
   
   for (int n = 0; n != C * R; n++) {
      int i = n / C;
      int j = n % C;
      mT[n] = m[R * j + i];
   }
   
   return mT;
}

vector< long double > VectorFunctions::softmax (const vector< long double > vec) {
   /*
    * Return the softmaxed version of given vector vec.
    * Input:
    *    vec: vector of floats
    * Output:
    *    smvec, softmaxed version of vec
    */
   vector< long double > smvec;
   long double maxelem = *std::max_element(std::begin(vec), std::end(vec));
   for (auto v : vec) {
      smvec.push_back(std::exp(v - maxelem));
   }
   return smvec;
}

vector< long double > VectorFunctions::sigmoid_d (const vector< long double > m) {
   /*  Returns the value of the sigmoid function derivative f'(x) = f(x)(1.0 - f(x)),
       where f(x) is sigmoid function.
       Input: m1, a vector.
       Output: x(1.0 - x) for every element of the input matrix m1.
   */
   const unsigned long VECTOR_SIZE = m.size();
   vector< long double > output(VECTOR_SIZE);
   
   for (unsigned int i = 0; i < VECTOR_SIZE; i++) {
      output[i] = sigmoid_d(m[i]);
   }
//    std::cout << "sigmoid_d input:" << m1[0] << std::endl;
   
   return output;
}

long double VectorFunctions::sigmoid_d (const long double f) {
   /*  Returns the value of the sigmoid function derivative f'(x) = f(x)(1.0 - f(x)),
       where f(x) is sigmoid function.
       Input: f, a float.
       Output: f1(1.0 - f).
   */
   return f * (1.0 - f);
}

vector< long double > VectorFunctions::sigmoid (const vector< long double > m) {
   /*  Returns the value of the sigmoid function f(x) = 1/(1.0 + e^-x).
       Input: m1, a vector.
       Output: 1/(1.0 + e^-x) for every element of the input matrix m1.
   */
   const unsigned long VECTOR_SIZE = m.size();
   vector< long double > output(VECTOR_SIZE);
   for (unsigned int i = 0; i != VECTOR_SIZE; ++i) {
      output[i] = sigmoid(m[i]);
   }
   return output;
}

long double VectorFunctions::sigmoid (const long double x) {
   /*
    * Calculate the sigmoid of given long double x.
    * Input:
    *    x, long double, number to take the sigmoid of.
    * Output:
    *    The sigmoid of x.
    */
//   long double epower = exp(-x);
//   if (epower < -1000) { epower = -1000; }
   long double y = 1.0 / (1.0 + exp(-x));
   if (y == 1.0) { y = 0.999999; } // Very cheesy
   if (y == 0.0) { y = 0.000001; }
   return y;
}

vector< long double > VectorFunctions::dot (const vector< long double > m1,
                                            const vector< long double > m2,
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
   vector< long double > output(m1_rows * m2_columns);
   
   for (int row = 0; row < m1_rows; row++) {
      for (int col = 0; col < m2_columns; col++) {
         output[row * m2_columns + col] = 0.f;
         for (int k = 0; k < m1_columns; k++) {
            output[row * m2_columns + col] += m1[row * m1_columns + k] * m2[k * m2_columns + col];
         }
      }
   }
   
   return output;
}

long double safeLog (const long double x) {
   /*
    * Return the log value of x.
    * If this would return NaN, then instead return 1.
    * Input:
    *    x, value to calculate the log-value of.
    * Output:
    *    The log value of x, or, if that'd be NaN, 1.
    */
   long double y = log(x);
   if (y != y) { return 1.0; }
   return y;
}

long double VectorFunctions::crossentropy_d(const long double output,
                                            const long double label,
                                            /*const vector< Node > nodeLayer,
                                            const unsigned int nodeNumber,*/
                                            const long double nodeValue,
                                            const bool softmax) {
   /*long double weightsum = 0.0;
   for (Node n : nodeLayer) {
      weightsum += n.weights[nodeNumber];
   }
   return (output - label) / (output - pow(output, 2.0)) * weightsum;*/
   
   // As per "Notes on Backpropagation" by Peter Sadowski
   return (output - label) * sigmoid(nodeValue);
}

// For the weights between the inputLayer and first hiddenlayer
// May differ with (and between) multiple hidden layers
long double VectorFunctions::crossentropy_d(const vector< long double > outputs,
                                            const vector< long double > labels,
                                            const Node targetNode,
                                            const Node originNode) {
   long double sum = 0.0;
   long double targetSigmoid = sigmoid(targetNode.value);
   for (unsigned int i = 0; i < outputs.size(); i++) {
      sum += (outputs[i] - labels[i]) * targetNode.weights[i] * 
             sigmoid_d(targetSigmoid) * sigmoid(originNode.value);
   }
   return sum;
}

long double VectorFunctions::crossEntropy (const vector< long double > output,
                                           const vector< long double > input,
                                           const vector< long double > labels,
                                           const bool softmax) {
   /*  
    * Returns the cross entropy between two vectors.
    * Inputs:
    *    output: vector
    *    labels: vector
    * Output: float, -sum(labels * log(output)), 
    *         cross entropy between output and labels.
    */
   if (output.size() != labels.size()) {
      std::cerr << "Labels and output not of same length!" << endl;
      std::cerr << "Labels length: " << labels.size() << endl;
      std::cerr << "Output length: " << output.size() << endl;
      exit(1);
   }
   
   long double out;

   if (softmax) {
      out = -vectorsum(labels * (input - log(vectorsum(epower(input)))));
   } else {
      long double crossent = 0.0;
      long double firstLog, secondLog;
      for (auto o = output.begin(), l = labels.begin(), e = output.end();
           o != e; o++, l++) {
         firstLog = /*(*o) == 0.0 ? 0.0 :*/ log(*o);
         secondLog = /*(1.0 - *o) == 0.0 ? 0.0 :*/ log(1.0 - *o);
         crossent += (*l * firstLog) + (1.0 - *l) * secondLog;
      }
      out = -crossent;
   }
   return out;
}

long double VectorFunctions::meanSquaredError (const vector< long double > output,
                                               const vector< long double > labels) {
   /*
    * Returns the mean square error between two vectors.
    * 
    */
   if (output.size() != labels.size()) {
      std::cerr << "Labels and output not of same length!" << endl;
      std::cerr << "Labels length: " << labels.size() << endl;
      std::cerr << "Output length: " << output.size() << endl;
      exit(1);
   }
   vector< long double > minus = output - labels;
   return vectorsum(minus * minus) / output.size();
}
