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
	
		float vectorsum(const vector<float>);
		vector<float> transpose (float*, const int, const int);
		vector<float> softmax(const vector<float>);
		vector<float> sigmoid_d(const vector<float>&);
		float sigmoid_d(const float);
		vector<float> sigmoid(const vector<float>&);
		vector<float> dot (const vector<float>&, const vector<float>&, 
                         const int, const int, const int);
		float crossEntropy(const vector<float>, const vector<float>);
		float meanSquaredError(const vector<float>, const vector<float>);
};

#endif
