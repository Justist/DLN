#ifndef NETWORK_H
#define NETWORK_h

#include "includes.hpp"
#include "vectorFunctions.hpp"

using std::vector;

class Network {
   private:
   	vector<vector<float>> weights;
   	
   	void initialiseOutputLayer(float seed);
   public:
   	//Network(int hiddenLayers, int layerLength, float seed);
   	Network(int hiddenLayers = 2, int layerLength = 4, float seed = 42);
      Network(vector<vector<float>> initWeights, bool outputIncluded = true,
              float seed = 42);
      ~Network();
};

#endif
