#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "Includes.hpp"

class Network {

private:

   /* Variables */

   // Layer of nodes which contain the input for the network.
   // Every input is a double.
   vecdo inputs;
   
   // The weights on the edges from the input layer to the 
   // first hidden layer.
   vecvecdo weightsFromInputs;
   
   // A vector containing the layers of nodes which are the hidden layers.
   // Each node consists only of a double.
   vecvecdo hiddenLayers;
   
   // The weights on the edges between the hidden layers.
   // Definitely the most complex data structure of this program.
   vector< vecvecdo > weightsHiddenLayers;
   
   // The weights on the edges between the last hidden layer and the output node.
   // This output node then contains the result of the calculations in the network,
   // based on the given input.
   vecvecdo weightsToOutput;
   
   // The value which is expected to be returned, to be compared to 
   // the calculatedOutput
   double expectedOutput;
   
   // The alpha, or the learning rate, of the network. 
   // The higher this value is, the more changes are allowed in the weights 
   // between each propagation through the network.
   // Changing this value may have impact on the training time.
   double alpha;
   
   // The output of the network. As of currently, this is only a single number,
   // but later additions or experiments might require more outputs.
   double calculatedOutput;
   
   // The scheme according to which the weights of the network are initialised.
   // To better understand this, please read the accompanying paper.
   std::string scheme;
   
public:
   

};

#endif
