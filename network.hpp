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
      vector<vector<float>> deltas;
      vector<vector<float>> inputValues;
      vector<float> outputLayer;
      vector<float> outputDelta;
      vector<float> outputLayerInput;
      unsigned int layerSize;
      unsigned int outSize;
      float networkSeed;
      float learningRate = 0.3;
      float accuracy = 0.0;
      unsigned int aantalgoed = 0;
      unsigned int aantalslecht = 0;
      VectorFunctions* VF;
      vector<float> oldOutput;
      
      void initialiseOutputLayer();
   public:
      //Network(int hiddenLayers, int layerLength, float seed);
      Network(int outputlength, int hiddenLayers = 2, 
              int layerLength = 4, float seed = 42);
      //Network(vector<vector<float>> initWeights, bool outputIncluded = true,
      //        float seed = 42);
      ~Network();
      
      vector<float> createOutput(const vector<float> input);
      vector<float> getWeightLayer(unsigned int layer);
      void backpropagate(const float errorRate, const vector<float> output);
      void run(const vector<float> input, const vector<float> labels);
      
      void exportNetwork(const std::string fileName);
      vector<vector<float>> importNetwork(const std::string fileName);
      
      void printOutputAndLabels(const vector<float> output, 
                                const vector<float> labels);
      void updateAccuracy(const vector<float> output, 
                          const vector<float> labels);
};

#endif
