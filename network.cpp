#include "network.hpp"
#include "overloads.cpp"

Network::Network(int outputLength, int hiddenLayers, int layerLength, float seed) {
	/*
	 * Create a neural network by initialising multiple layers of weights.
	 * Each layer is a vector of floats, and the network is a vector of layers.
	 * The network is stored in class variable weights.
	 * Input:
	 * 	outputLength: The length the outputvector should have.
	 * 	hiddenLayers: The amount of hidden layers to be contained in the network.
	 * 	layerLength: The length of each of the hidden layers.
	 * 	seed: Random seed for the random number generation.
	 */
	outSize = outputLength;
	networkSeed = seed;
	srand(networkSeed);
	
	VectorFunctions VF();
	
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
	 * 	hiddenLayers: The amount of hidden layers to be contained in the network.
	 * 	              Default value is 2.
	 * 	layerLength: The length of each of the hidden layers.
	 * 	             Default value is 4.
	 * 	seed: Random seed for the random number generation.
	 * 	      Default value is 42.
	 * /
	Network(hiddenLayers, layerLength, seed);
}*/

Network::Network(vector<vector<float>> initWeights, bool outputIncluded, float seed) {
	/*
	 * Create a neural network by copying another.
	 * Input:
	 * 	initWeights: A neural network in the same form as the program uses.
	 * 	             This is explained in the first constructor.
	 * 	outputIncluded: Whether or not the initWeights contains weights for
	 * 	                the output layer or only for the hidden layers.
	 * 	                Default value is true.
	 * 	seed: Random seed for the random number generation when generating 
	 * 	      the output layer, if needed.
	 * 	      Default value is 42.
	 */
	weights = initWeights;
	networkSeed = seed;
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

vector<float> Network::createOutput(const vector<float> input) {
	/*
	 * Runs the input through the network.
	 * Input:
	 * 	input: Vector of floats to which the weights are applied to calculate
	 * 	       an output vector.
	 * Output:
	 * 	vector of floats, output of the network
	 */
	 vector<float> output(outSize);
	 for(auto o : output) {
	 	*o = VF.dot(...);
	 }
}
