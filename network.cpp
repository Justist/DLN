#include "network.hpp"
#include "overloads.cpp"

Network::Network (unsigned int outputLength, unsigned int hiddenLayers,
                  unsigned int layerLength, unsigned int seed) {
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
   
   std::cout << "\33c\r" << std::endl; //clean the terminal
   printf("Training the network with %d layers containing %d nodes, with seed %d \
           \nand learning rate %.4f\n\n", hiddenLayers, layerLength,
                                          seed, learningRate);
   outputLayerSize = outputLength;
   networkSeed = seed;
   srand(networkSeed);
   hiddenLayerSize = layerLength;
   hiddenLayerAmount = hiddenLayers;
   
   VF = new VectorFunctions();
   
   // Fill the hiddenlayers with nodes
   vector< double > defaultWeights(hiddenLayerSize, weightInit);
   Node node = {defaultWeights, 0.0, 0.0};
   Node biasNode = {defaultWeights, -1.0, 0.0};
   for (unsigned int i = 0; i < hiddenLayers; i++) {
      vector< Node > weightLayer(hiddenLayerSize, node);
      weightLayer[0] = biasNode;
      hiddenlayers.push_back(weightLayer);
   }
   // Fill the outputlayer with nodes
   OutputNode outputNode = {0.0, 0.0};
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
   vector< double > defaultWeights(weightSize, weightInit);
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

void Network::createOutput (const vector< double > input) {
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
   unsigned long inputSize = input.size();
   unsigned long inputLayerSize = inputLayer.size();
   
   // If the input is larger than the inputLayer, make the inputLayer larger
   if (inputSize > inputLayerSize) {
      inputLayer.resize(inputSize);
      auto begin = inputLayer.begin() + inputLayerSize;
      auto end = inputLayer.end();
      initialiseWeights(begin, end, hiddenLayerSize);
   }
   
   // Then set the inputLayer to the input.
   for (unsigned int i = 0; i < input.size(); i++) {
      inputLayer[i].value = input[i];
   }
   
   // First calculate the first layer of the hidden layers with the input
   for (Node inputNode : inputLayer) {
      for (unsigned int j = 1; j < hiddenLayerSize; j++) {
         hiddenlayers[0][j].value += inputNode.value * inputNode.weights[j];
         if (hiddenlayers[0][j].value != hiddenlayers[0][j].value) {
            printf("input value %.5f, weights %.5f\n", inputNode.value, inputNode.weights[j]);
            exit(0);
         }
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
         if (outputLayer[m].value != outputLayer[m].value) {
            printf("out value %.5f, weights %.5f\n", hiddenNode.value, hiddenNode.weights[m]);
            exit(0);
         }
      }
   }
}

//void Network::exportNetwork(const std::string fileName) {
//   /*
//    * Export the network to a given file.
//    * Format for exporting is, when given network
//    * a1 b1
//    * a2 b2
//    * a3 b3
//    * is as follows:
//    * a1 a2 a3 
//    * b1 b2 b3 
//    * 
//    * Input:
//    *    fileName, string, name of the file the network will be written to.
//    */
//   std::ofstream of(fileName);
//   for(weightLayer layer : weights) {
//      for(weightMap weight : layer) {
//         of << weight << " ";
//      }
//      of << "\n";
//   }
//}

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
//   vector<vector<double>> network;
//   double f;
//   std::string line = "";
//   while (std::getline(inF, line, '\n')) {
//      std::stringstream ss(line);
//      vector<double> layer;
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

void Network::backpropagate (const double errorRate) {
   /*
    * Backpropagate through the network and adjust the weights based on the
    * given error rate and the learning rate.
    * Input:
    *    errorRate, float, depicting the error rate based on which the weights
    *    are to be adjusted.
    */
   // Update the delta of the output layer
   for (OutputNode& on : outputLayer) {
      on.delta = errorRate * VF->sigmoid_d(on.value);
   }
   // Update the weights and deltas of the last hidden layer
   for (Node node : hiddenlayers[hiddenLayerAmount - 1]) {
      node.delta = 0.0;
      for (unsigned int i = 0; i < outputLayerSize; i++) {
         node.weights[i] += learningRate * outputLayer[i].delta * VF->sigmoid(node.value);
         node.delta += VF->sigmoid_d(node.value) * node.weights[i] * outputLayer[i].delta;
      }
   }
   // Do the same for all other hidden layers
   for (int j = hiddenLayerAmount - 2; j >= 0; j--) {
      for (Node& node : hiddenlayers[j]) {
         node.delta = 0.0;
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
                            VF->sigmoid(node.value);
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

void Network::printOutputAndLabels (const vector< double > output,
                                    const vector< double > labels) {
   /*
    * Print both the output and the labels (what the output should be),
    * so a comparison can be made by the user.
    * First it calls a function to clear the previous output, then prints
    * both the output and the labels, and then the updated accuracy.
    * Inputs:
    *    output, the output of the network
    *    labels, what the output should be
    */
   unsigned int lines = 2 + outputLayerSize;
   //clearXLines(lines);
   std::cout << "\33[" + std::to_string(lines) + "A\r";
   std::cout << "Output\t\tLabel" << std::endl;
   for (auto o = begin(output), l = begin(labels), e = end(output);
        o != e; o++, l++) {
      printf("%.6f\t%.0f\n", *o, *l);
   }
   printf("Accuracy: %.6f en teller: %d\n", accuracy, teller++);
}

void Network::updateAccuracy (const vector< double > output,
                              const vector< double > labels) {
   /*
    * Update the accuracy of the network, based on the output and what the 
    * output should be.
    * Input:
    *    output, the output of the network
    *    labels, what the output should be
    */
   double comp;
   const unsigned long VECTOR_SIZE = output.size();
   for (unsigned long i = 0; i < VECTOR_SIZE; i++) {
      comp = output[0] + labels[0];
      if ((comp >= 1.5) || (comp < .5)) { aantalgoed++; }
      else { aantalslecht++; }
   }
   
   accuracy = aantalgoed / (float) (aantalgoed + aantalslecht);
}

vector < double> Network::returnOutputValues () {
   vector < double > outputValues;
   for (OutputNode node : outputLayer) {
      outputValues.push_back(node.value);
   }
   return outputValues;
}

void Network::run (const vector< double > input, const vector< double > labels) {
   /*
    * Higher level function to run the input through the network and compare
    * the output to the given labels. It just calls the functions needed to do so.
    * Input:
    *    input, vector<float>, the input to run through the network.
    *    labels, vector<float>, the labels to compare the output to.
    */
   createOutput(input);
   vector < double > output = returnOutputValues();
   double error = VF->crossEntropy(output, input, labels, false);
   
   updateAccuracy(output, labels);
   printOutputAndLabels(output, labels);
   backpropagate(error);
}