#ifndef NETWORK_H
#define NETWORK_H

#include "includes.hpp"
#include "vectorFunctions.hpp"

using std::vector;

class Network {
   private:
      vector< vector< Node>> hiddenlayers;
      vector< Node > inputLayer;
      vector< OutputNode > outputLayer;
      unsigned int hiddenLayerSize;
      unsigned int hiddenLayerAmount;
      unsigned int outputLayerSize;
      unsigned int networkSeed;
      double weightInit = 0.5;
      float learningRate = 0.5;
      float accuracy = 0.0;
      unsigned int aantalgoed = 0;
      unsigned int aantalslecht = 0;
      VectorFunctions *VF;
      
      unsigned int teller = 0;
   
   public:
      explicit Network (unsigned int outputlength, unsigned int hiddenLayers = 1,
                        unsigned int layerLength = 2, unsigned int seed = 420);
      
      ~Network ();
      
      void clearValues (vector< Node >& nodeLayer);
      
      void clearValues (vector< vector< Node>>& nodeLayers);
      
      template< typename Iterator >
      void initialiseWeights (Iterator& begin, Iterator& end,
                              int weightSize);
      
      void createOutput (vector< double > input);
      
      //void backpropagate(double errorRate);
      void run (vector< double > input, vector< double > labels);
      
      //void exportNetwork(std::string fileName);
      //void importNetwork(std::string fileName);
      
      void printOutputAndLabels (vector< double > output,
                                 vector< double > labels);
      
      void updateAccuracy (vector< double > output,
                           vector< double > labels);
   
   vector< double > returnOutputValues ();
};

#endif
