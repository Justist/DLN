#ifndef NETWORK_H
#define NETWORK_h

#include <cstring>
#include <fstream>
#include <sstream>
#include "includes.hpp"
#include "vectorFunctions.hpp"

using std::vector;

class Network {
   private:
   	vector<vector<float>> weights;
   	vector<float> outputLayer;
   	int outSize;
   	float networkSeed;
   	float learningRate = 0.4;
   	VectorFunctions* VF;
   	
   	void initialiseOutputLayer();
   public:
   	//Network(int hiddenLayers, int layerLength, float seed);
   	Network(int outputlength, int hiddenLayers = 2, 
   	        int layerLength = 4, float seed = 42);
      Network(vector<vector<float>> initWeights, bool outputIncluded = true,
              float seed = 42);
      ~Network();
      
      vector<float> createOutput(const vector<float> input);
      vector<float> getWeightLayer(unsigned int layer);
      void backpropagate(const float errorRate);
      void run(const vector<float> input, const vector<float> labels);
      
      void exportNetwork(const std::string fileName);
      vector<vector<float>> importNetwork(const std::string fileName);
};

#endif
