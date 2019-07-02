#include "Network.hpp"

Network::Network(const vecdo&                   inputs,
                 const vecvecdo&                wFI,
                 const vecvecdo&                hL,
                 const std::vector< vecvecdo >& wHL,
                 const vecvecdo&                wTO,
                 const double&                  eO,
                 const double&                  alpha,
                 const double&                  cO,
                 const std::string&             scheme) {
   _inputs              = inputs;
   _weightsFromInputs   = wFI;
   _hiddenLayers        = hL;
   _weightsHiddenLayers = wHL;
   _weightsToOutput     = wTO;
   _expectedOutput      = eO;
   _alpha               = alpha;
   _calculatedOutput    = cO;
   _scheme              = scheme;
}

void Network::initialiseWeights(const uint16_t seed,
                                const vecdo& schemeWeights/* = {}*/) {
   /*
    * Initialise the weights of the given weightLayers.
    * If a scheme is given, fill the layers using that
    * scheme, else fill it with random values.
    * The layers are returned by reference.
    */
    
   const auto inputNodes   = amInputNodes();
   const auto hiddenNodes  = amHiddenNodes();
   const auto hiddenLayers = amHiddenLayers();
   const auto outputNodes  = amOutputNodes();
   
   bool useScheme = false;
   if (!schemeWeights.empty()) { useScheme = true; }
   
   for (uint16_t i = 0; i < inputNodes; i++) {
      for (uint16_t h = 0; h < hiddenNodes - 1; h++) {
         _weightsFromInputs[i][h] = useScheme ?
                                   schemeWeights[i*hiddenNodes + h] :
                                   General::randomWeight(seed);
      }
   }

   for (uint16_t l = 0; l < hiddenLayers - 1; l++) {
      for (uint16_t hp = 0; hp < hiddenNodes; hp++) {
         for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
            _weightsHiddenLayers[l][hp][hn] = 
               useScheme ?
                  schemeWeights[(inputNodes * (hiddenNodes - 1)) +
                                (l * hiddenNodes) + hp + hn] :
                  General::randomWeight(seed);
         }
      }
   }

   for (uint16_t h = 0; h < hiddenNodes; h++) {
      for (uint16_t o = 0; o < outputNodes; o++) {
         _weightsToOutput[h][o] = useScheme ?
                                  schemeWeights[
                                       (inputNodes * (hiddenNodes - 1)) +
                                        ((hiddenLayers - 1) *
                                         hiddenNodes * (hiddenNodes - 1)) +
                                       (o * hiddenNodes + h)]:
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
   
   /*
   std::cerr << "Inputs: ";
   for (auto i : _inputs) { std::cerr << i << "; "; }
   std::cerr << std::endl;
   
   std::cerr << "Weights: ";
   for (uint16_t h = 0; h < hiddenNodes - 1; h++) {
      for (uint16_t i = 0; i < _inputs.size(); i++) {
         std::cerr << _weightsFromInputs[i][h] << "; ";
      }
   }
   std::cerr << std::endl;
   */
   
   // Last node of a layer is the bias node, having a constant value of -1.

   const auto inputSize = _inputs.size();
   for (uint16_t h = 0; h < hiddenNodes; h++) {
      //bias has value -1
      _hiddenLayers[0][h] = -_weightsFromInputs[inputSize - 1][h];
      for (uint16_t i = 0; i < inputSize - 1; i++) {
         _hiddenLayers[0][h] +=
            _weightsFromInputs[i][h] * General::sigmoid(_inputs[i]);
      }
   }

   //hp is hidden previous
   //hn is hidden next
   //for the previous and next hidden layer
   for (uint16_t l = 0; l < hiddenLayers - 1; l++) {
      for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
         //bias has value -1
         _hiddenLayers[l + 1][hn] = -_weightsHiddenLayers[l][hiddenNodes - 1][hn];
         for (uint16_t hp = 0; hp < hiddenNodes - 1; hp++) {
            _hiddenLayers[l + 1][hn] +=
               _weightsHiddenLayers[l][hp][hn] * 
               General::sigmoid(_hiddenLayers[l][hp]);
         }
      }
   }

   // only 1 output
   _calculatedOutput = -_weightsToOutput[hiddenNodes - 1][0];
   for (uint16_t h = 0; h < hiddenNodes - 1; h++) {
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
   const auto inputNodes   = amInputNodes();
   const auto hiddenNodes  = amHiddenNodes();
   const auto outputNodes  = amOutputNodes();
   
   // Forward
   forward();

   // Backward
   const double deltaOutput =
      General::sigmoid_d(_calculatedOutput) *
      (_expectedOutput - General::sigmoid(_calculatedOutput));
   vecvecdo deltas(hiddenLayers, vecdo(hiddenNodes, 0.0));

   for (uint16_t h = 0; h < hiddenNodes; h++) {
      for (uint16_t o = 0; o < outputNodes; o++) {
         deltas[hiddenLayers - 1][h] += _weightsToOutput[h][o] * deltaOutput;
         _weightsToOutput[h][o] +=
              _alpha                                                *
              General::sigmoid(_hiddenLayers[hiddenLayers - 1][h])  *
              deltaOutput;
      }
   
      deltas[hiddenLayers - 1][h] *=
           General::sigmoid_d(_hiddenLayers[hiddenLayers - 1][h]);
   }

   for (auto l = static_cast<int16_t>(hiddenLayers - 2); l >= 0; l--) {
      for (uint16_t hp = 0; hp < hiddenNodes; hp++) {
         for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
            deltas[l][hp] +=
               _weightsHiddenLayers[l][hp][hn] * 
               deltas[l + 1][hn];
         }
         deltas[l][hp] *= General::sigmoid_d(_hiddenLayers[l][hp]);
         for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
            _weightsHiddenLayers[l][hp][hn] +=
               _alpha * 
               General::sigmoid(_hiddenLayers[l][hp]) * 
               deltas[l + 1][hn];
         }
      }
   }

   for (uint16_t h = 0; h < hiddenNodes - 1; h++) {
      for (uint16_t i = 0; i < inputNodes; i++) {
         _weightsFromInputs[i][h] += _alpha *
                                     General::sigmoid(_inputs[i]) *
                                     deltas[0][h];
         /*
         // Totally not cheating around small numbers
         double weight = _weightsFromInputs[i][h];
         double addition = _alpha * General::sigmoid(_inputs[i]) * deltas[0][h];
         if (weight > 0 && weight < pow(10, -100)) {
            _weightsFromInputs[i][h] = addition;
         } else if (weight < 0 && weight > pow(-10, -100)) {
            _weightsFromInputs[i][h] = addition;
         } else {
            _weightsFromInputs[i][h] += addition;
         }
         */
      }
   }
}

void Network::writeDot(const std::string& filename) {
    /*
     * Writes the network to a file in the DOT format.
     * This way, that file can be opened by GraphViz,
     * and the network can be visualised.
     */
   
    const auto hiddenLayers = amHiddenLayers();
    const auto inputNodes   = amInputNodes();
    const auto hiddenNodes  = amHiddenNodes();
    const auto outputNodes  = amOutputNodes();
    
    FILE *of = NULL;
    while (of == NULL) {
       perror("fopen");
       usleep(10); //Let us not clog the system with fopen calls
       of = fopen(filename.c_str(), "w");
    }
    // Printed at the start
    fprintf(of, "digraph{\n");

    /* First apply labels to all the nodes. */

    for (uint16_t iindex = 0; iindex < inputNodes; iindex++) {
        fprintf(of, "i%d [label = %f];\n", iindex, _inputs[iindex]);
    }

    for (uint16_t hlindex = 0; hlindex < hiddenLayers; hlindex++) {
        for (uint16_t hnindex = 0; hnindex < hiddenNodes; hnindex++) {
            fprintf(of, "h%d%d [label = %f];\n", hlindex, hnindex, _hiddenLayers[hlindex][hnindex]);
        }
    }

    // TODO place for loop when calculatedOutput is a vector
    fprintf(of, "o0 [label = %f];\n", _calculatedOutput);

    /* Then put in all the edges. */

    for (uint16_t i = 0; i < inputNodes; i++) {
        for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
            fprintf(of, "i%d -> h0%d [label = %f];\n", i, hn, _weightsFromInputs[i][hn]);
        }
    }
    
    // -1 to account for the fact there is 1 layer more than edges in between
    for (uint16_t hl = 0; hl < hiddenLayers - 1; hl++) {
        for (uint16_t hn1 = 0; hn1 < hiddenNodes; hn1++) {
            for (uint16_t hn2 = 0; hn2 < hiddenNodes - 1; hn2++) {
                fprintf(of, "h%d%d -> h%d%d [label = %f];\n", hl, hn1, hl, hn2, _weightsHiddenLayers[hl][hn1][hn2]);
            }
        }
    }

    const uint16_t lastHiddenLayer = hiddenLayers - 1;
    for (uint16_t hn = 0; hn < hiddenNodes; hn++) {
       for (uint16_t out = 0; out < outputNodes; out++) {
          fprintf(of, "h%d%d -> o%d [label = %f];\n", lastHiddenLayer, hn, out, _weightsToOutput[hn][out]);
       }
    }


    /* And then set the ranks of all nodes. */

    fprintf(of, "{ rank=same;");
    for (uint16_t i = 0; i < _inputs.size(); i++) { fprintf(of, " i%d,", i); }
    // Move filepointer 1 back
    fseek(of, -1, SEEK_CUR);
    fprintf(of, " }\n");

    for (uint16_t hl = 0; hl < _hiddenLayers.size(); hl++) {
        fprintf(of, "{ rank=same;");
        for (uint16_t hn = 0; hn < _hiddenLayers[0].size(); hn++) { fprintf(of, " h%d%d,", hl, hn); }
        fseek(of, -1, SEEK_CUR);
        fprintf(of, " }\n");
    }

    // In case of multiple output nodes, also implement that

    // Printed at the end
    fprintf(of, "}");
    fclose(of);
}
