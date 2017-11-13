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
      vector<vector<double>> weights;
      vector<vector<double>> deltas;
      vector<vector<double>> inputValues;
      vector<double> outputLayer;
      vector<double> outputDelta;
      vector<double> outputLayerInput;
      unsigned int layerSize;
      unsigned int outSize;
      float networkSeed;
      float learningRate = 0.005;
      float accuracy = 0.0;
      unsigned int aantalgoed = 0;
      unsigned int aantalslecht = 0;
      VectorFunctions* VF;
//      vector<float> oldOutput;
      
      unsigned int teller = 0;
      
      void initialiseOutputLayer();
   public:
      Network(int outputlength, int hiddenLayers = 1, 
              int layerLength = 2, float seed = 420.42);
      ~Network();
      
      vector<double> createOutput(const vector<double> input);
      vector<double> getWeightLayer(unsigned int layer);
      void backpropagate(const double errorRate);
      void run(const vector<double> input, const vector<double> labels);
      
      void exportNetwork(const std::string fileName);
      void importNetwork(const std::string fileName);
      
      void printOutputAndLabels(const vector<double> output, 
                                const vector<double> labels);
      void updateAccuracy(const vector<double> output, 
                          const vector<double> labels);
};

#endif
