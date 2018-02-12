#ifndef NETWORK_H
#define NETWORK_H

#include "includes.hpp"
#include "vectorFunctions.hpp"

using std::vector;

class Network {
   private:
      vector< vector< Node > > hiddenlayers;
      vector< Node > inputLayer;
      vector< OutputNode > outputLayer;
      unsigned int hiddenLayerSize;
      unsigned int hiddenLayerAmount;
      unsigned int outputLayerSize;
      unsigned int networkSeed;
      unsigned int iterations;
      long double weightInit = 0.5;
      float learningRate;
      unsigned int aantalgoed = 0;
      unsigned int aantalslecht = 0;
      VectorFunctions *VF;
      
      unsigned int teller = 0;
   
   public:
      explicit Network (unsigned int outputlength, unsigned int hiddenLayers = 1,
                        unsigned int layerLength = 4, unsigned int seed = 420,
                        float alpha = 0.5);
      
      ~Network ();
      
      void clearValues (vector< Node >& nodeLayer);
      
      void clearValues (vector< vector< Node>>& nodeLayers);
      
      template< typename Iterator >
      void initialiseWeights (Iterator& begin, Iterator& end,
                              unsigned long weightSize);
      
      void createOutput (vector< long double > input);
   
      void backpropagate (vector< long double >,
                          vector< long double >,
                          bool);
      void run (vector< long double > input, vector< long double > labels);
   
      void exportNetwork (std::string fileName);
   
      void importNetwork (std::string fileName);
      
      void printOutputAndLabels (vector< long double > output,
                                 vector< long double > labels,
                                 double error);
      
      void updateAccuracy (vector< long double > output,
                           vector< long double > labels);
   
      vector< long double > returnOutputValues ();
      
      
      
      float accuracy = 0.0;
};

#endif
