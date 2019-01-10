#include "SchemeNetwork.hpp"

SchemeNetwork::SchemeNetwork(vecdo inputs,
                             vecvecdo wFI,
                             vecvecdo hL,
                             std::vector< vecvecdo > wHL,
                             vecvecdo wTO,
                             double eO,
                             double alpha,
                             double cO,
                             std::string& scheme) :
                            Network(vecdo inputs,
                                    vecvecdo wFI,
                                    vecvecdo hL,
                                    std::vector< vecvecdo > wHL,
                                    vecvecdo wTO,
                                    double eO,
                                    double alpha,
                                    double cO) {
   _scheme = scheme;
}

SchemeNetwork::SchemeNetwork(const uint16_t inputs,
                             const uint16_t hiddenLayers,
                             const uint16_t hiddenNodes,
                             const uint16_t outputs,
                             const double alpha,
                             const uint16_t seed,
                             const std::string& scheme/* = ""*/) {
   /*
    * Construct a network using the parameters.
    * This is basically only calling the functions
    * which set the weights.
    * If a scheme is used, first schemeVector is filled
    * so that the scheme is reflected in the weights.
    * The '+ 1' after inputs and hiddenNodes is to account
    * for the bias nodes, which are added to the network.
    */
   const auto hiddenPlusBias = static_cast<const uint16_t>(hiddenNodes + 1);
   vecvecdo wFI(inputs + 1, vecdo(hiddenNodes));
   std::vector< vecvecdo > wHL(hiddenLayers,
                          vecvecdo(hiddenPlusBias,
                                   vecdo(hiddenNodes)));
   vecvecdo wTO(hiddenPlusBias, vecdo(outputs));
   
   //I am annoyed I have to take this step
   _weightsFromInputs = wFI;
   _weightsHiddenLayers = wHL;
   _weightsToOutput = wTO;
   
   _alpha = alpha;
   
   vecdo schemeVector = {};
   if (scheme.length() > 0) {
      schemeVector = initialiseWeightsByScheme(scheme, seed);
      _scheme = scheme;
   }
   
   initialiseWeights(seed, //seed
                     schemeVector); //scheme weights
}

vecdo SchemeNetwork::initialiseWeightsByScheme(const std::string& scheme,
                                               const unsigned int seed) {
   /*
    * Function takes a scheme in format "aaabbbcccddd" etc,
    * where equal letters represent the same 'random' 
    * weight in that position.
    * Then it makes a vector of equal length with on each
    * position a 'random' weight, according to this scheme.
    */
   char current = scheme[0];
   vecdo weights(scheme.length(), -1.0);
   weights[0] = General::randomWeight(seed);
   for (unsigned int i = 1; i < scheme.length(); i++) {
      if (current == scheme[i]) {
         weights[i] = weights[i-1];
      } else {
         weights[i] = General::trueRandomWeight(seed, weights);
      }
   }
   return weights;
}
