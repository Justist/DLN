#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "Includes.hpp"

#include "General.cpp"

class Network {

private:
   
   /* Variables */
   
   // Layer of nodes which contain the input for the network.
   // Every input is a double.
   vecdo _inputs;
   
   // The weights on the edges from the input layer to the
   // first hidden layer.
   vecvecdo _weightsFromInputs;
   
   // A vector containing the layers of nodes which are the hidden layers.
   // Each node consists only of a double.
   vecvecdo _hiddenLayers;
   
   // The weights on the edges between the hidden layers.
   // Definitely the most complex data structure of this program.
   std::vector< vecvecdo > _weightsHiddenLayers;
   
   // The weights on the edges between the last hidden layer and the output node.
   // This output node then contains the result of the calculations in the network,
   // based on the given input.
   vecvecdo _weightsToOutput;
   
   // The value which is expected to be returned, to be compared to
   // the calculatedOutput
   double _expectedOutput;
   
   // The alpha, or the learning rate, of the network.
   // The higher this value is, the more changes are allowed in the weights
   // between each propagation through the network.
   // Changing this value may have impact on the training time.
   double _alpha;
   
   // The output of the network. As of currently, this is only a single number,
   // but later additions or experiments might require more outputs.
   double _calculatedOutput;
   
   // The scheme according to which the weights of the network are initialised.
   // To better understand this, please read the accompanying paper.
   std::string _scheme;

public:
   
   Network(vecdo inputs,
           vecvecdo wFI,
           vecvecdo hL,
           std::vector< vecvecdo > wHL,
           vecvecdo wTO,
           double eO,
           double alpha,
           double cO,
           std::string scheme);
           
   void initialiseWeights(uint16_t seed,
                          const vecdo& schemeWeights = {});
   // Forward propagation for the network
   void forward();
   // Backward propagation for the network
   // Also called training
   void train();
           
   /* Information callers */

   uint16_t amInputNodes() const
      { return static_cast<uint16_t>(_inputs.size()); }
   uint16_t amHiddenNodes() const
      { return static_cast<uint16_t>(_hiddenLayers[0].size()); }
   uint16_t amHiddenLayers() const
      { return static_cast<uint16_t>(_hiddenLayers.size()); }
   uint16_t amOutputNodes() const { return 1; }
   
   /* Getters */
   
   const vecdo&  inputs() const { return _inputs; }
   const double& inputs(const uint16_t i) const
   { return _inputs[i]; }
   
   const vecvecdo& weightsFromInputs() const
   { return _weightsFromInputs; }
   const vecdo&    weightsFromInputs(const uint16_t i) const
   { return _weightsFromInputs[i]; }
   const double&   weightsFromInputs(const uint16_t i, const uint16_t j) const
   { return _weightsFromInputs[i][j]; }
   
   const vecvecdo& hiddenLayers() const
   { return _hiddenLayers; }
   const vecdo&    hiddenLayers(const uint16_t i) const
   { return _hiddenLayers[i]; }
   const double&   hiddenLayers(const uint16_t i,
                                const uint16_t j) const
   { return _hiddenLayers[i][j]; }
   
   const std::vector< vecvecdo >& weightsHiddenLayers() const
   { return _weightsHiddenLayers; }
   const vecvecdo&                weightsHiddenLayers(const uint16_t i) const
   { return _weightsHiddenLayers[i]; }
   const vecdo&                   weightsHiddenLayers(const uint16_t i,
                                                      const uint16_t j) const
   { return _weightsHiddenLayers[i][j]; }
   const double&                  weightsHiddenLayers(const uint16_t i,
                                                      const uint16_t j,
                                                      const uint16_t k) const
   { return _weightsHiddenLayers[i][j][k]; }
   
   const vecvecdo& weightsToOutput() const
   { return _weightsToOutput; }
   const vecdo&   weightsToOutput(const uint16_t i) const
   { return _weightsToOutput[i]; }
   const double&  weightsToOutput(const uint16_t i,
                                  const uint16_t j) const
   { return _weightsToOutput[i][j]; }
   
   const double& expectedOutput() const { return _expectedOutput; }
   
   const double& alpha() const { return _alpha; }
   
   const double& calculatedOutput() const { return _calculatedOutput; }
   
   const std::string& scheme() const { return _scheme; }
   
   /* Setters */
   
   void inputs(const vecdo& a) { _inputs = a; }
   void inputs(const uint16_t i, const double& a) { _inputs[i] = a; }
   
   void weightsFromInputs(const vecvecdo& a)
   { _weightsFromInputs = a; }
   void weightsFromInputs(const uint16_t i,
                          const vecdo& a)
   { _weightsFromInputs[i] = a; }
   void weightsFromInputs(const uint16_t i,
                          const uint16_t j,
                          const double& a)
   { _weightsFromInputs[i][j] = a; }
   
   void hiddenLayers(const vecvecdo& a)
   { _hiddenLayers = a; }
   void hiddenLayers(const uint16_t i,
                     const vecdo& a)
   { _hiddenLayers[i] = a; }
   void hiddenLayers(const uint16_t i,
                     const uint16_t j,
                     const double& a)
   { _hiddenLayers[i][j] = a; }
   
   void weightsHiddenLayers(const std::vector< vecvecdo >& a)
   { _weightsHiddenLayers = a; }
   void weightsHiddenLayers(const uint16_t i, const vecvecdo& a)
   { _weightsHiddenLayers[i] = a; }
   void weightsHiddenLayers(const uint16_t i,
                            const uint16_t j,
                            const vecdo& a)
   { _weightsHiddenLayers[i][j] = a; }
   void weightsHiddenLayers(const uint16_t i,
                            const uint16_t j,
                            const uint16_t k,
                            const double& a)
   { _weightsHiddenLayers[i][j][k] = a; }
   
   void weightsToOutput(const vecvecdo& a)
   { _weightsToOutput = a; }
   void weightsToOutput(const uint16_t i, const vecdo& a)
   { _weightsToOutput[i] = a; }
   void weightsToOutput(const uint16_t i,
                        const uint16_t j,
                        const double& a)
   { _weightsToOutput[i][j] = a; }
   
   void expectedOutput(const double& a) { _expectedOutput = a; }
   
   void alpha(const double& a) { _alpha = a; }
   
   void calculatedOutput(const double& a) { _calculatedOutput = a;}
   
   void scheme(const std::string& a) { _scheme = a; }
   
};

#endif
