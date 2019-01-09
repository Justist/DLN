#include "Network.hpp"

Network::Network(const vecdo inputs,
                 const vecvecdo wFI,
                 const vecvecdo hL,
                 const std::vector< vecvecdo > wHL,
                 const vecvecdo wTO,
                 const double eO,
                 const double alpha,
                 const double cO,
                 const std::string scheme) {
   _inputs = inputs;
   _weightsFromInputs = wFI;
   _hiddenLayers = hL;
   _weightsHiddenLayers = wHL;
   _weightsToOutput = wTO;
   _expectedOutput = eO;
   _alpha = alpha;
   _calculatedOutput = cO;
   _scheme = scheme;
}