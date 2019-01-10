#ifndef SCHEMENETWORK_HPP
#define SCHEMENETWORK_HPP

#include "Includes.hpp"

#include "General.cpp"
#include "Network.hpp"

class SchemeNetwork : Network {
   private:
   
      // The scheme according to which the weights of the network are initialised.
      // To better understand this, please read the accompanying paper.
      std::string _scheme;
      
   public:
   
      SchemeNetwork(const uint16_t inputs,
                    const uint16_t hiddenLayers,
                    const uint16_t hiddenNodes,
                    const uint16_t outputs,
                    const double alpha,
                    const uint16_t seed,
                    const std::string& scheme = "");
   
      vecdo initialiseWeightsByScheme(const std::string& scheme,
                                      const unsigned int seed)
                                      
      const std::string& scheme() const { return _scheme; }
      
      void scheme(const std::string& a) { _scheme = a; }
   
};

#endif
