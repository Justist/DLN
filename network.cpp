#include "network.hpp"
#include "overloads.cpp"

Network::Network(int outputLength, int hiddenLayers, int layerLength, float seed) {
   /*
    * Create a neural network by initialising multiple layers of weights.
    * Each layer is a vector of floats, and the network is a vector of layers.
    * The network is stored in class variable weights.
    * Input:
    *    outputLength: The length the outputvector should have.
    *    hiddenLayers: The amount of hidden layers to be contained in the network.
    *    layerLength: The length of each of the hidden layers.
    *    seed: Random seed for the random number generation.
    */
   outSize = outputLength;
   networkSeed = seed;
   srand(networkSeed);
   
   VF = new VectorFunctions();
   
   for(int i = 0; i < hiddenLayers; i++) {
      vector<float> layer(layerLength);
      for(int j = 0; j < layerLength; j++) {
         layer[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      }
      weights.push_back(layer);
   }
   initialiseOutputLayer();
}

/*Network::Network(int hiddenLayers, int layerLength, float seed) {
   / *
    * Front-end for the constructor.
    * This front-end requires no arguments to be supplied.
    * Input:
    *    hiddenLayers: The amount of hidden layers to be contained in the network.
    *                  Default value is 2.
    *    layerLength: The length of each of the hidden layers.
    *                 Default value is 4.
    *    seed: Random seed for the random number generation.
    *          Default value is 42.
    * /
   Network(hiddenLayers, layerLength, seed);
}*/

Network::Network(vector<vector<float>> initWeights, bool outputIncluded, float seed) {
   /*
    * Create a neural network by copying another.
    * Input:
    *    initWeights: A neural network in the same form as the program uses.
    *                 This is explained in the first constructor.
    *    outputIncluded: Whether or not the initWeights contains weights for
    *                    the output layer or only for the hidden layers.
    *                    Default value is true.
    *    seed: Random seed for the random number generation when generating 
    *          the output layer, if needed.
    *          Default value is 42.
    */
   weights = initWeights;
   networkSeed = seed;
   
   VF = new VectorFunctions();
   
   if(!outputIncluded) { initialiseOutputLayer(); }
}

Network::~Network() {}

void Network::initialiseOutputLayer() {
   /*
    * Initialise the output weights of the network.
    */
   srand(networkSeed);
   vector<float> layer(outSize);
   for(int j = 0; j < outSize; j++) {
         layer[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
   }
   outputLayer = layer;
}

vector<float> Network::getWeightLayer(unsigned int layer) {
   /*
    * Return the requested layer of weights.
    * Input:
    *    layer: Non-negative integer depicting the layer requested.
    * Output:
    *    vector<float> containing the requested layer.
    */
   if(layer > weights.size()) {
      std::cerr << "Layer does not exist!\nGiving layer 0 instead." << std::endl;
      layer = 0;
   }
   return weights[layer];
}

vector<float> Network::createOutput(const vector<float> input) {
   /*
    * Runs the input through the network.
    * Input:
    *    input: Vector of floats to which the weights are applied to calculate
    *           an output vector.
    * Output:
    *    vector of floats, output of the network
    */
   vector<float> output = input;
   for(vector<float> w : weights) {
      output = VF->dot(w, output, w.size(), 1, 1);
   }
   output = VF->sigmoid(VF->dot(outputLayer, output, outSize, 1, 1));
   return output;
}

void Network::exportNetwork(const std::string fileName) {
   /*
    * Export the network to a given file.
    * Format for exporting is, when given network
    * a1 b1
    * a2 b2
    * a3 b3
    * is as follows:
    * a1 a2 a3 
    * b1 b2 b3 
    * 
    * Input:
    *    fileName, string, name of the file the network will be written to.
    */
   std::ofstream of(fileName);
   for(vector<float> layer : weights) {
      for(float weight : layer) {
         of << weight << " ";
      }
      of << "\n";
   }
}

vector<vector<float>> Network::importNetwork(const std::string fileName) {
   /*
    * Import a network in given file and return it.
    * It is returned so the user can also specify if the outputlayer is
    * included.
    * The network to be imported should be in the format as described
    * in the comments of function Network::exportNetwork.
    * Input:
    *    fileName, string, name of the file the network will be read from.
    * Output:
    *    a vector<vector<float>> containing the network.
    */
   std::ifstream inF(fileName);
   vector<vector<float>> network;
   float f;
   std::string line = "";
   while (std::getline(inF, line, '\n')) {
      std::stringstream ss(line);
      vector<float> layer;
      while(!ss.fail()) {
         ss >> f;
         layer.push_back(f);
      }
      network.push_back(layer);
   }
   return network;
}

void Network::backpropagate(const float errorRate) {
   /*
    * Backpropagate through the network and adjust the weights based on the
    * given error rate and the learning rate.
    * Input:
    *    errorRate, float, depicting the error rate based on which the weights
    *    are to be adjusted.
    */
   ;
}

void Network::run(const vector<float> input, const vector<float> labels) {
   /*
    * Higher level function to run the input through the network and compare
    * the output to the given labels. It just calls the functions needed to do so.
    * Input:
    *    input, vector<float>, the input to run through the network.
    *    labels, vector<float>, the labels to compare the output to.
    */
   vector<float> output = createOutput(input);
   float error = VF->crossEntropy(output, labels);
   backpropagate(error);
}
