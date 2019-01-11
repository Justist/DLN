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

void Network::initialiseWeights(const uint16_t seed,
                                const vecdo& schemeWeights/* = {}*/) {
   /*
    * Initialise the weights of the given weightLayers.
    * If a scheme is given, fill the layers using that
    * scheme, else fill it with random values.
    * The layers are returned by reference.
    */
    
   const auto inputNodes   = amInputNodes() + 1;
   const auto hiddenNodes  = amHiddenNodes() + 1;
   const auto hiddenLayers = amHiddenLayers();
   const auto outputNodes  = amOutputNodes();
   
   bool useScheme = false;
   if (!schemeWeights.empty()) { useScheme = true; }
   for (uint16_t i = 0; i < inputNodes; i++) {
      for (uint16_t h = 0; h < hiddenNodes - 1; h++) {
         _weightsFromInputs[i][h] = useScheme ?
                                   schemeWeights[i*hiddenNodes + (h - 1)] :
                                   General::randomWeight(seed);
      }
   }

   for (uint16_t l = 0; l < hiddenLayers - 1; l++) {
      for (uint16_t hp = 0; hp < hiddenNodes; hp++) {
         for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
            _weightsHiddenLayers[l][hp][hn] = 
               useScheme ?
                  schemeWeights[l*hiddenNodes + hp + hn] :
                  General::randomWeight(seed);
         }
      }
   }

   for (uint16_t h = 0; h < hiddenNodes; h++) {
      for (uint16_t o = 0; o < outputNodes; o++) {
         _weightsToOutput[h][o] = useScheme ?
                                  schemeWeights[o*hiddenNodes +
                                                h +
                                                (hiddenNodes - 1)*inputNodes] :
                                  General::randomWeight(seed);
      }
   }
}

void Network::forward() {
   /*
    * Basically a forward propagation through the network.
    * _calculatedOutput contains the result of the
    * propagation.
    */
   const auto hiddenLayers = amHiddenLayers();
   const auto hiddenNodes  = amHiddenNodes();

   for (uint16_t h = 0; h < hiddenNodes - 1; h++) {
      //bias has value -1
      _hiddenLayers[0][h + 1] = -_weightsFromInputs[0][h];
      for (uint16_t i = 1; i < _inputs.size(); i++) {
         _hiddenLayers[0][h + 1] +=
            _weightsFromInputs[i][h] * _inputs[i];
      }
   }

   //hp is hidden previous
   //hn is hidden next
   //for the previous and next hidden layer
   for (uint16_t l = 0; l < hiddenLayers - 1; l++) {
      for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
         //bias has value -1
         _hiddenLayers[l + 1][hn + 1] = -_weightsHiddenLayers[l + 1][0][hn];
         for (uint16_t hp = 1; hp < hiddenNodes; hp++) {
            _hiddenLayers[l + 1][hn + 1] +=
               _weightsHiddenLayers[l][hp][hn] * 
               General::sigmoid(_hiddenLayers[l][hp]);
         }
      }
   }

   // only 1 output
   _calculatedOutput = -_weightsToOutput[0][0];
   for (uint16_t h = 1; h < hiddenNodes; h++) {
      _calculatedOutput += _weightsToOutput[h][0] *
                           General::sigmoid(_hiddenLayers[hiddenLayers - 1][h]);
   }
}

void Network::train() {
   /*
    * Both forward and backward propagation through the
    * network. First the forward propagation is done in
    * forward(), as testing it is done by forward
    * propagation.
    * For the backward propagation some optimisation may
    * be possible, but it works for now.
    */
   const auto hiddenLayers = amHiddenLayers();
   const auto hiddenNodes  = amHiddenNodes();
   
   double previous;

   // Forward
   forward();

   // Backward
   const double deltaOutput =
      General::sigmoid_d(_calculatedOutput) *
      (_expectedOutput - General::sigmoid(_calculatedOutput));
   vecvecdo deltas(hiddenLayers, vecdo(hiddenNodes, 0.0));

   for (uint16_t h = 0; h < hiddenNodes; h++) {
      deltas[hiddenLayers - 1][h] += _weightsToOutput[h][0] * deltaOutput;
      deltas[hiddenLayers - 1][h] *=
         General::sigmoid_d(_hiddenLayers[hiddenLayers - 1][h]);
      _weightsToOutput[h][0] +=
         _alpha * 
         General::sigmoid(_hiddenLayers[hiddenLayers - 1][h]) * 
         deltaOutput;
   }

   for (int16_t l = hiddenLayers - 2; l >= 0; l--) {
      for (uint16_t hp = 0; hp < hiddenNodes; hp++) {
         for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
            deltas[l][hp] +=
               _weightsHiddenLayers[l][hp][hn] * 
               deltas[l + 1][hn + 1];
         }
         deltas[l][hp] *= General::sigmoid_d(_hiddenLayers[l][hp]);
         for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
            _weightsHiddenLayers[l][hp][hn] +=
               _alpha * 
               General::sigmoid(_hiddenLayers[l][hp]) * 
               deltas[l + 1][hn + 1];
         }
      }
   }

   for (uint16_t h = 0; h < hiddenNodes - 1; h++) {
      for (uint16_t i = 0; i < _inputs.size(); i++) {
         // Totally not cheating around small numbers
         double weight = _weightsFromInputs[i][h];
         double addition = _alpha * _inputs[i] * deltas[0][h + 1];
         if (weight > 0 && weight < pow(10, -100)) {
            _weightsFromInputs[i][h] = addition;
         } else if (weight < 0 && weight > pow(-10, -100)) {
            _weightsFromInputs[i][h] = addition;
         } else {
            _weightsFromInputs[i][h] += addition;
         }
      }
   }
}
