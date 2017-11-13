#ifndef VECTORFUNCTIONS_H
#define VECTORFUNCTIONS_H

#include "includes.hpp"

using std::vector;
using std::cout;
using std::endl;

class VectorFunctions {
	private:
		
	public:
		VectorFunctions() {}
		~VectorFunctions() {}
	
		double vectorsum(const vector<double>);
		vector<double> epower(const vector<double>);
		vector<double> transpose (double*, const int, const int);
		vector<double> softmax(const vector<double>);
		vector<double> sigmoid_d(const vector<double>);
		double sigmoid_d(const double);
		vector<double> sigmoid(const vector<double>);
		vector<double> dot (const vector<double>, const vector<double>, 
                         const int, const int, const int);
		double crossEntropy(const vector<double>, const vector<double>, 
                          const vector<double>, const bool);
		double meanSquaredError(const vector<double>, const vector<double>);
};

#endif
