#include "network.hpp"
#include "overloads.cpp"

Network::Network(int hiddenLayers, int layerLength, float seed) {
	/*
	 * Create a neural network by initialising multiple layers of weights.
	 * Each layer is a vector of floats, and the network is a vector of layers.
	 * The network is stored in class variable weights.
	 * Input:
	 * 	hiddenLayers: The amount of hidden layers to be contained in the network.
	 * 	layerLength: The length of each of the hidden layers.
	 * 	seed: Random seed for the random number generation.
	 */
	srand(seed);
	for(int i = 0; i < hiddenLayers; i++) {
		vector<float> layer(layerLength);
		for(int j = 0; j < layerLength; j++) {
			layer[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
		weights.push_back(layer);
	}
	initialiseOutputLayer(seed);
}

/*Network::Network(int hiddenLayers, int layerLength, float seed) {
	/*
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
	if(!outputIncluded) { initialiseOutputLayer(seed); }
}

Network::~Network() {}

void Network::initialiseOutputLayer(float seed) {
	/*
	 * Misschien wil ik nog ergens outputLength initialiseren...
	 */
	/*for(int j = 0; j < outputLength; j++) {
			layer[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}*/
	;
}
