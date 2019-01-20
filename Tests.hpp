#ifndef TESTS_HPP
#define TESTS_HPP

#include "Includes.hpp"

#include "General.cpp"
#include "Network.hpp"

class Tests {
   private:
      template <typename T>
      void Print(FILE* of,
                 T toWrite,
                 bool toFile);
   
      void PrintResults(const vecvecdo& inputs,
                        const vecdo& outputs,
                        bool toFile/* = false*/,
                        const std::string& filename/* = ""*/,
                        const std::string& writeMode/* = "w"*/,
                        const std::string& firstString/* = "In: "*/,
                        const std::string& secondString/* = "Out: "*/,
                        bool equalSize/* = true*/);
   public:
      Tests() = default;
   
      void XOR(vecdo& inputs, double& output);
      void XORTest(Network n,
                   bool toFile = false,
                   std::string filename = "",
                   const std::string& writeMode = "w",
                   bool seedTest = false,
                   int seed = -1,
                   const std::string& addition = "");
   
      void ABC(vecdo& inputs, double& output);
};

#endif
