#ifndef TESTS_HPP
#define TESTS_HPP

#include "Includes.hpp"
#include "Network.hpp"

class Tests {
   private:
      
   public:
      Tests() = default;
   
      void XOR(vecdo&, double&);
      void XORTest(Network n,
                   const bool toFile = false,
                   std::string filename = "",
                   const std::string& writeMode = "w",
                   const bool seedTest = false,
                   const int seed = -1,
                   const std::string& addition = "");
};

#endif
