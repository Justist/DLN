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
    
   std::cout << "\33c\r" << std::endl; //clean the terminal
   printf("Training the network with %d layers containing %d nodes, with seed %.2f \
           \nand learning rate %.4f\n\n\n\n\n", hiddenLayers, layerLength, 
                                                seed, learningRate);
   outSize = outputLength;
   networkSeed = seed;
   srand(networkSeed);
   layerSize = layerLength;
   
   VF = new VectorFunctions();
   
   for(int i = 0; i < hiddenLayers; i++) {
      vector<double> weightLayer(layerSize), 
                     deltaLayer(layerSize), 
                     inputLayer(layerSize);
      for(int j = 0; j < layerLength; j++) {
         weightLayer[j] = static_cast <double> (rand()) / 
                          static_cast <double> (RAND_MAX);
         deltaLayer[j] = 1.0;
         inputLayer[j] = 0.0;
      }
      weights.push_back(weightLayer);
      deltas.push_back(deltaLayer);
      inputValues.push_back(inputLayer);
   }
   initialiseOutputLayer();
}

Network::~Network() {}

void Network::initialiseOutputLayer() {
   /*
    * Initialise the output weights of the network.
    */
   srand(networkSeed);
   vector<double> layer(outSize);
   for(unsigned int j = 0; j < outSize; j++) {
         layer[j] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
   }
   outputLayer = layer;
}

vector<double> Network::getWeightLayer(unsigned int layer) {
   /*
    * Return the requested layer of weights.
    * Input:
    *    layer: Non-negative integer depicting the layer requested.
    * Output:
    *    vector<float> containing the requested layer.
    */
   if(layer > layerSize) {
      std::cerr << "Layer does not exist!\nGiving layer 0 instead." << std::endl;
      layer = 0;
   }
   return weights[layer];
}

vector<double> Network::createOutput(const vector<double> input) {
   /*
    * Runs the input through the network.
    * Input:
    *    input: Vector of floats to which the weights are applied to calculate
    *           an output vector.
    * Output:
    *    vector of floats, output of the network
    */
   vector<double> output = input;
   for(auto lw = weights.begin(), li = inputValues.begin(), ew = weights.end(); lw != ew; lw++, li++) {
      *li = VF->dot(*lw, output, layerSize, 1, 1);
      output = *li;
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
   for(vector<double> layer : weights) {
      for(double weight : layer) {
         of << weight << " ";
      }
      of << "\n";
   }
}

void Network::importNetwork(const std::string fileName) {
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
   vector<vector<double>> network;
   double f;
   std::string line = "";
   while (std::getline(inF, line, '\n')) {
      std::stringstream ss(line);
      vector<double> layer;
      while(!ss.fail()) {
         ss >> f;
         layer.push_back(f);
      }
      network.push_back(layer);
   }
   weights = network;
   outputLayer = weights.back();
   weights.pop_back();
}

void Network::backpropagate(const double errorRate/*, const vector<float> output*/) {
   /*
    * Backpropagate through the network and adjust the weights based on the
    * given error rate and the learning rate.
    * Input:
    *    errorRate, float, depicting the error rate based on which the weights
    *    are to be adjusted.
    */
   vector<double> u;
   //For-loop over the layers.
   for(auto lw = weights.begin(), ld = deltas.begin(), li = inputValues.begin(), 
            ew = weights.end(); lw != ew; lw++, ld++, li++) {
      //For-loop over the elements in each layer.
      u = *li * *ld * learningRate * errorRate;
      
      std::cout << "\33[6A\r" << "u: ";
      for(float x : u) { printf("%.6f", x); }
      if(u[0] != u[0]) { printf("\n%f, %f, %f, %f", VF->sigmoid_d(*li)[0], (*ld)[0], learningRate, errorRate); exit(0); }
      std::cout << "\33[6B\r";
//      exit(0);
//      if(VF->vectorsum(u) == 0.0) { 
//         for(auto x = u.begin(), e = u.end(); x != e; x++) { *x = 0.01; } 
//      }
      
      *lw = *lw + u;
      *ld = u;
      //*lw = VF->sigmoid_d(*lw);
   }
}

void clearXLines(const unsigned int x) {
   /*
    * Clears the current line and the x lines above.
    * Input:
    *    x, the amount of lines above the current line to clear.
    */
   std::cout << "\33[2K\r"; // go to start of line with the cursor
   for(unsigned int i = 0; i < x; i++) {
      std::cout << "\33[A\r"; // go 1 line up with the cursor
      std::cout << "\33[2K\r"; // clear the current line
   }
}

void Network::printOutputAndLabels(const vector<double> output, 
                                   const vector<double> labels) {
   /*
    * Print both the output and the labels (what the output should be),
    * so a comparison can be made by the user.
    * First it calls a function to clear the previous output, then prints
    * both the output and the labels, and then the updated accuracy.
    * Inputs:
    *    output, the output of the network
    *    labels, what the output should be
    */
   unsigned int lines = 2 + outSize;
   //clearXLines(lines);
   std::cout << "\33[" + std::to_string(lines) + "A\r";
   std::cout << "Output\t\tLabel" << std::endl;
   for(auto o = begin(output), l = begin(labels), e = end(output); 
            o != e; o++, l++) {
      printf("%.6f\t%.0f\n", *o, *l);
   }
   printf("Accuracy: %.6f en teller: %d\n", accuracy, teller++);
}

void Network::updateAccuracy(const vector<double> output, 
                             const vector<double> labels) {
   /*
    * Update the accuracy of the network, based on the output and what the 
    * output should be.
    * Input:
    *    output, the output of the network
    *    labels, what the output should be
    */
   float comp = 0.0;
   const unsigned long VECTOR_SIZE = output.size();
   for(unsigned long i = 0; i < VECTOR_SIZE; i++) {
      comp = output[0] + labels[0];
      if((comp >= 1.5) || (comp < .5)) aantalgoed++;
      else aantalslecht++;
   }

   accuracy = aantalgoed / (float) (aantalgoed + aantalslecht);
}

void Network::run(const vector<double> input, const vector<double> labels) {
   /*
    * Higher level function to run the input through the network and compare
    * the output to the given labels. It just calls the functions needed to do so.
    * Input:
    *    input, vector<float>, the input to run through the network.
    *    labels, vector<float>, the labels to compare the output to.
    */
   vector<double> output = createOutput(input);
   double error = VF->crossEntropy(output, input, labels, false);
   
   updateAccuracy(output, labels);
   printOutputAndLabels(output, labels);
   backpropagate(error);
}
