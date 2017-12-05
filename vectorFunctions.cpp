#include "vectorFunctions.hpp"
#include "overloads.cpp"

double VectorFunctions::vectorsum (const vector< double > vec) {
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

vector< double > VectorFunctions::epower (const vector< double > m) {
   /*
    * Returns a vector of float which is the given vector of floats but with
    * each element x being a power of e, such that every element x in the
    * given vector is an element e^x in the output vector.
    * Input:
    *    m: vector of floats
    * Output:
    *    vector of floats which are all e-powers
    */
   vector< double > output;
   for (double x : m) {
      output.push_back(std::exp(x));
   }
   return output;
}

vector< double > VectorFunctions::transpose (const double *m, const int C, const int R) {
   /*
    *  Returns a transpose matrix of input matrix.
    *  Inputs:
    *      m: vector, input matrix
    *      C: int, number of columns in the input matrix
    *      R: int, number of rows in the input matrix
    *  Output: vector, transpose matrix mT of input matrix m
    */
   vector< double > mT(C * R);
   
   for (int n = 0; n != C * R; n++) {
      int i = n / C;
      int j = n % C;
      mT[n] = m[R * j + i];
   }
   
   return mT;
}

vector< double > VectorFunctions::softmax (const vector< double > vec) {
   /*
    * Return the softmaxed version of given vector vec.
    * Input:
    *    vec: vector of floats
    * Output:
    *    smvec, softmaxed version of vec
    */
   vector< double > smvec;
   double maxelem = *std::max_element(std::begin(vec), std::end(vec));
   for (auto v : vec) {
      smvec.push_back(std::exp(v - maxelem));
   }
   return smvec;
}

vector< double > VectorFunctions::sigmoid_d (const vector< double > m1) {
   /*  Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
       where f(x) is sigmoid function.
       Input: m1, a vector.
       Output: x(1 - x) for every element of the input matrix m1.
   */
   const unsigned long VECTOR_SIZE = m1.size();
   vector< double > output(VECTOR_SIZE);
   
   for (unsigned i = 0; i < VECTOR_SIZE; i++) {
      output[i] = m1[i] * (1 - m1[i]);
   }
//    std::cout << "sigmoid_d input:" << m1[0] << std::endl;
   
   return output;
}

double VectorFunctions::sigmoid_d (const double f1) {
   /*  Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
       where f(x) is sigmoid function.
       Input: f1, a float.
       Output: f1(1 - f1).
   */
   return f1 * (1 - f1);
}

vector< double > VectorFunctions::sigmoid (const vector< double > m1) {
   /*  Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
       Input: m1, a vector.
       Output: 1/(1 + e^-x) for every element of the input matrix m1.
   */
   const unsigned long VECTOR_SIZE = m1.size();
   vector< double > output(VECTOR_SIZE);
   
   for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
      output[i] = 1 / (1 + exp(-m1[i]));
   }
   
   return output;
}

vector< double > VectorFunctions::dot (const vector< double > m1,
                                       const vector< double > m2,
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
   vector< double > output(m1_rows * m2_columns);
   
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

double safeLog (const double x) {
   /*
    * Return the log value of x.
    * If this would return NaN, then instead return 1.
    * Input:
    *    x, value to calculate the log-value of.
    * Output:
    *    The log value of x, or, if that'd be NaN, 1.
    */
   double y = log(x);
   if (y != y) { return 1; }
   return y;
}

double VectorFunctions::crossEntropy (const vector< double > output,
                                      const vector< double > input,
                                      const vector< double > labels,
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
   
   double out = 0.0;

//   FILE * testoutput;
//   testoutput = fopen("testoutput.txt", "w");
   
   if (softmax) {
      out = -vectorsum(labels * (input - log(vectorsum(epower(input)))));
   } else {
      double crossent = 0.0;
      double firstLog, secondLog;
      for (auto o = output.begin(), l = labels.begin(), e = output.end();
           o != e; o++, l++) {
         firstLog = (*o) == 0 ? 0 : log(*o);
         secondLog = (1 - *o) == 0 ? 0 : log(1 - *o);
         crossent += (*l * firstLog) + (1 - *l) * secondLog;
         //fprintf(testoutput, "crossent = %f\n", crossent);
         //fprintf(testoutput, "o = %f, l = %f, log(o) = %f, log(1 - o) = %f\n", 
         //        *o, *l, safeLog(*o), safeLog(1 - *o));
         //if(crossent != crossent) { printf("o: %f, l: %f", *o, *l); exit(0); }
      }
      out = -crossent;
   }
//   fclose(testoutput);
   return out;
//   const unsigned long VECTOR_SIZE = output.size();
//   for (unsigned i = 0; i != VECTOR_SIZE; ++i){
//      crossent += labels[i] * log(output[i]);
//   };
//   return -crossent;
}

double VectorFunctions::meanSquaredError (const vector< double > output,
                                          const vector< double > labels) {
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
   vector< double > minus = output - labels;
   return vectorsum(minus * minus) / output.size();
}
