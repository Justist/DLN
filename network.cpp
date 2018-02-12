#include "network.hpp"
#include "overloads.cpp"

void checkForNan (const long double number, const std::vector< long double >& rest) {
   /*
    * Debug function.
    */
   if (number != number) {
      for (long double x : rest) {
         std::cout << x << " ";
      }
      std::cout << std::endl;
      exit(1);
   }
}

Network::Network (unsigned int outputLength, unsigned int hiddenLayers,
                  unsigned int layerLength, unsigned int seed, float alpha) {
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
   // Raise an error when one of these float exceptions occur.
   feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);
   
   outputLayerSize = outputLength;
   networkSeed = seed;
   srand(networkSeed);
   hiddenLayerSize = layerLength;
   hiddenLayerAmount = hiddenLayers;
   learningRate = alpha;
   
   std::cout << "\33c\r" << std::endl; //clean the terminal
   printf("Training the network with %d layers containing %d nodes, with seed %d \
           \nand learning rate %.4f\n\n", hiddenLayerAmount, hiddenLayerSize,
                                          networkSeed, learningRate);
   
   VF = new VectorFunctions();
   
   // Fill the hiddenlayers with nodes
   int size = hiddenLayerSize;
   for (unsigned int i = 0; i < hiddenLayers; i++) {
      if(i == hiddenLayers - 1) {
         // Last layer may have different size of weightvectors.
         size = outputLayerSize;
      }
      vector< long double > defaultWeights(size, weightInit);
      Node node = {defaultWeights, 0.0, 1.0};
      Node biasNode = {defaultWeights, -1.0, 1.0};
      vector< Node > nodeLayer(hiddenLayerSize, node);
      nodeLayer[0] = biasNode;
      hiddenlayers.push_back(nodeLayer);
   }
   // Fill the outputlayer with nodes
   OutputNode outputNode = {0.0, 1.0}; // Value and delta
   for (unsigned int j = 0; j < outputLayerSize; j++) {
      outputLayer.push_back(outputNode);
   }
}

Network::~Network () = default;

template< typename Iterator >
void Network::initialiseWeights (Iterator& begin, Iterator& end,
                                 const unsigned long weightSize) {
   /*
    * Initialise the weights of a given (part of a) vector of Nodes to weightInit.
    * Input:
    *    begin, the iterator pointing to the start of the vector (part)
    *    end, the iterator pointing to the end of the vector (part)
    *    weightSize, the size of the layer above, so the amount of weights the current node should have
    */
   vector< long double > defaultWeights(weightSize, weightInit);
   for (Iterator& it = begin; it != end; ++it) {
      (*it).weights = defaultWeights;
   }
}

void Network::clearValues (vector< vector< Node>>& nodeLayers) {
   /*
    * Call clearValues for each vector of Nodes in the network.
    * Used to clear the network before feeding new input.
    * Input:
    *     nodeLayers, a vector of vectors of Nodes.
    *       Basically the whole network.
    */
   for (vector< Node >& nodeLayer : nodeLayers) {
      clearValues(nodeLayer);
   }
}

void Network::clearValues (vector< Node >& nodeLayer) {
   /*
    * Set the value of each node in the given nodeLayer to 0.
    * The bias node, the first one, is set to -1.
    * Input:
    *     nodeLayer, vector of Nodes
    */
   for (Node& node : nodeLayer) {
      node.value = 0.0;
   }
   nodeLayer[0].value = -1.0;
}

void Network::createOutput (const vector< long double > input) {
   /*
    * Runs the input through the network.
    * Input:
    *    input: Vector of floats to which the weights are applied to calculate
    *           an output vector.
    * Output:
    *    vector of floats, output of the network
    */
   // Clear the network, set all values to 0.0 except for the bias nodes
   clearValues(hiddenlayers);
   for (OutputNode& node : outputLayer) { node.value = 0.0; }
   
   // Then put the input values in the inputLayer
   // Count the bias node
   unsigned long inputSize = input.size() + 1;
   unsigned long inputLayerSize = inputLayer.size();
   
   // If the input is larger than the inputLayer, make the inputLayer larger
   if (inputSize > inputLayerSize) {
      // Keep in mind the bias node
      inputLayer.resize(inputSize);
      auto begin = inputLayer.begin() + inputLayerSize;
      auto end = inputLayer.end();
      initialiseWeights(begin, end, hiddenLayerSize);
   }
   
   // Then set the inputLayer to the input.
   inputLayer[0].value = -1.0;
   for (unsigned int i = 0; i < input.size(); i++) {
      inputLayer[i+1].value = input[i];
   }
   
   // First calculate the first layer of the hidden layers with the input
   for (Node inputNode : inputLayer) {
      // Don't update the bias node
      for (unsigned int j = 1; j < hiddenLayerSize; j++) {
         hiddenlayers[0][j].value += inputNode.value * inputNode.weights[j];
      }
   }
   
   // Then do so for each hidden layer except the last
   for (unsigned int k = 0; k < hiddenLayerAmount - 1; k++) {
      for (Node hiddenNode : hiddenlayers[k]) {
         for (unsigned int l = 1; l < hiddenLayerSize; l++) {
            hiddenlayers[k+1][l].value += hiddenNode.value * hiddenNode.weights[l];
         }
      }
   }
   
   // Then for the last layer
   for (Node hiddenNode : hiddenlayers[hiddenLayerAmount - 1]) {
      for (unsigned int m = 0; m < outputLayerSize; m++) {
         outputLayer[m].value += hiddenNode.value * hiddenNode.weights[m];
      }
   }
}

void Network::exportNetwork (const std::string fileName) {
   /*
    * Export the network to a given file.
    * First print "x y z", where:
    *    x is the size of the output layer
    *    y is the size of the hidden layers
    *    z is the amount of hidden layers
    * Then print the weights between each pair of layers,
    * starting with the topmost between the last hidden and
    * the output layer. For each such batch of weights, per
    * Node the vector of weights is printed on one line.
    * So for the topmost weights, y rows containing x elements
    * are printed.
    * Then the other weights are printed in similar fashion, each
    * following directly on the other, giving z sets of such.
    * Then at last the input weights are printed. As these may be
    * variable in the program, no length has the be given and all
    * the last rows are considered to be part of it.
    *
    * TODO: Rewrite so it actually says what it does.
    *
    * Input:
    *    fileName, string, name of the file the network will be written to.
    */
   std::ofstream of(fileName);
   of << outputLayerSize << " "
      << hiddenLayerSize << " "
      << hiddenLayerAmount << "\n";
      
   for (Node node : inputLayer) {
      for (long double weight : node.weights) {
         of << weight << " ";
      }
      of << "\n";
   }
   
   for (vector< Node > layer : hiddenlayers) {
      for (Node node : layer) {
         for (long double weight : node.weights) {
            of << weight << " ";
         }
         of << "\n";
      }
   }
   
}

//void Network::importNetwork(const std::string fileName) {
//   /*
//    * Import a network in given file and return it.
//    * It is returned so the user can also specify if the outputlayer is
//    * included.
//    * The network to be imported should be in the format as described
//    * in the comments of function Network::exportNetwork.
//    * Input:
//    *    fileName, string, name of the file the network will be read from.
//    * Output:
//    *    a vector<vector<float>> containing the network.
//    */
//   std::ifstream inF(fileName);
//   vector<vector<long double>> network;
//   long double f;
//   std::string line = "";
//   while (std::getline(inF, line, '\n')) {
//      std::stringstream ss(line);
//      vector<long double> layer;
//      while(!ss.fail()) {
//         ss >> f;
//         layer.push_back(f);
//      }
//      network.push_back(layer);
//   }
//   weights = network;
//   outputLayer = weights.back();
//   weights.pop_back();
//}

void Network::backpropagate (const long double errorRate,
                             const vector< long double > outputs,
                             const vector< long double > labels,
                             const bool softmax) {
   /*
    * Backpropagate through the network and adjust the weights based on the
    * given error rate and the learning rate.
    * Input:
    *    errorRate, float, depicting the error rate based on which the weights
    *    are to be adjusted.
    */
   /*// Update the delta of the output layer
   for (unsigned int i = 0; i < outputLayer.size(); i++) {
      outputLayer[i].delta = errorRate * 
                             VF->crossentropy_d(output[i], 
                                                labels[i],
                                                hiddenlayers[hiddenLayerAmount - 1],
                                                i,
                                                softmax);
   }
   // Update the weights and deltas of the last hidden layer
   for (Node& node : hiddenlayers[hiddenLayerAmount - 1]) {
      node.delta = 1.0;
      for (unsigned int i = 0; i < outputLayerSize; i++) {
         node.weights[i] += learningRate * outputLayer[i].delta * VF->sigmoid(node.value);
         node.delta += VF->sigmoid_d(node.value) * node.weights[i] * outputLayer[i].delta;
      }
   }
   // Do the same for all other hidden layers
   for (int j = hiddenLayerAmount - 2; j >= 0; j--) {
      for (Node& node : hiddenlayers[j]) {
         node.delta = 1.0;
         for (unsigned int i = 0; i < hiddenLayerSize; i++) {
            node.weights[i] += learningRate * hiddenlayers[j + 1][i].delta *
                               VF->sigmoid(node.value);
            node.delta += VF->sigmoid_d(node.value) * node.weights[i] *
                          hiddenlayers[j + 1][i].delta;
         }
      }
   }
   // Then do so for the weights of the input layer
   // No delta is updated, because there are no layers beneath this one
   for (Node& node : inputLayer) {
      for (unsigned int i = 0; i < hiddenLayerSize; i++) {
         node.weights[i] += learningRate * hiddenlayers[0][i].delta *
                            VF->sigmoid_d(node.value);
      }
   }*/
   
   // As per "Notes on Backpropagation" by Peter Sadowski
   // First update the weights to the outputLayer
   for (unsigned int i = 0; i < outputLayerSize; i++) {
      const long double o = outputs[i];
      const long double l = labels[i];
      for (Node& n : hiddenlayers[hiddenLayerAmount - 1]) {
         n.weights[i] += learningRate * VF->crossentropy_d(o, l, 
                                                           n.value, softmax);
      }
   }
   
   // Then the weights between the hidden layers
   // TODO
   
   // And lastly the weights from the inputLayer
   for (unsigned int j = 0; j < hiddenLayerSize; j++) {
      for (Node& n : inputLayer) {
         n.weights[j] += learningRate * VF->crossentropy_d(outputs, labels, 
                                                           hiddenlayers[0][j], n);
      }
   }
}

void clearXLines (const unsigned int x) {
   /*
    * Clears the current line and the x lines above.
    * Input:
    *    x, the amount of lines above the current line to clear.
    */
   std::cout << "\33[2K\r"; // go to start of line with the cursor
   for (unsigned int i = 0; i < x; i++) {
      std::cout << "\33[A\r"; // go 1 line up with the cursor
      std::cout << "\33[2K\r"; // clear the current line
   }
}

void Network::printOutputAndLabels (const vector< long double > output,
                                    const vector< long double > labels) {
   /*
    * Print both the output and the labels (what the output should be),
    * so a comparison can be made by the user.
    * First it calls a function to clear the previous output, then prints
    * both the output and the labels, and then the updated accuracy.
    * Inputs:
    *    output, the output of the network
    *    labels, what the output should be
    */
   unsigned int lines = outputLayerSize;
   //clearXLines(lines);
   std::cout << "\33[" + std::to_string(lines) + "A\r";
   std::cout << "Output\t\tLabel" << std::endl;
   for (auto o = begin(output), l = begin(labels), e = end(output);
        o != e; o++, l++) {
      printf("%.6f\t%.0f\n", *o, *l);
   }
   printf("Accuracy: %.6f en teller: %d\n", accuracy, teller++);
}

void Network::updateAccuracy (const vector< long double > output,
                              const vector< long double > labels) {
   /*
    * Update the accuracy of the network, based on the output and what the 
    * output should be.
    * Input:
    *    output, the output of the network
    *    labels, what the output should be
    */
   long double comp;
   const unsigned long VECTOR_SIZE = output.size();
   for (unsigned long i = 0; i < VECTOR_SIZE; i++) {
      comp = output[0] + labels[0];
      if ((comp >= 1.5) || (comp < .5)) { aantalgoed++; }
      else { aantalslecht++; }
   }
   
   accuracy = aantalgoed / (float) (aantalgoed + aantalslecht);
}

vector<long double> Network::returnOutputValues () {
   /*
    * Convert the outputLayer, consisting of OutputNodes,
    * to a vector of long doubles which are the values of these
    * OutputNodes.
    * Output:
    *    vector of long doubles, corresponding to the values of
    *       the OutputNodes in outputLayer.
    */
   vector < long double > outputValues;
   for (OutputNode node : outputLayer) {
      outputValues.push_back(node.value);
   }
   return outputValues;
}

void Network::run (const vector< long double > input, const vector< long double > labels) {
   /*
    * Higher level function to run the input through the network and compare
    * the output to the given labels. It just calls the functions needed to do so.
    * Input:
    *    input, vector<float>, the input to run through the network.
    *    labels, vector<float>, the labels to compare the output to.
    */
   createOutput(input);
   exportNetwork("testweights");
   vector< long double > output = VF->sigmoid(returnOutputValues());
//   for (auto x : output) { std::cerr << x << " ";}
//   std::cerr << std::endl;
//   exit(0);
   const bool softmax = false;
   long double error = VF->crossEntropy(output, input, labels, softmax);
   
   updateAccuracy(output, labels);
   printOutputAndLabels(output, labels);
   backpropagate(error, output, labels, softmax);
}
