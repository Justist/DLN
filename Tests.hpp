#include <utility>

#include <utility>

#ifndef TESTS_HPP
#define TESTS_HPP

#include "Includes.hpp"

#include "General.cpp"
#include "Network.hpp"

class Tests {
   public:
      struct TestParameters {
         /*
          * Used for the calling of the test functions in the
          * function run(). 
          */
         Network network;
         bool toFile;
         std::string fileName;
         std::string writeMode;
         bool seedtest;
         int seed;
         std::string epoch;
         std::string addition;
         
         TestParameters(Network n,
                        const bool t,
                        std::string f,
                        std::string w,
                        const bool st,
                        const int s,
                        const std::string& e = "",
                        std::string a = "")
                        :
                        network(std::move(n)),
                        toFile(t),
                        fileName(std::move(f)),
                        writeMode(std::move(w)),
                        seedtest(st),
                        seed(s),
                        addition(std::move(a))
         {
            if (!e.empty()) { epoch = e; }
         };
      };
   
      Tests() = default;
   
      void XOR(vecdo& inputs, double& output);
   
      void ABC(vecdo& inputs, double& output);
      
      double runTest(TestParameters tp, const std::string& test);
      
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
      
      double XORTest(TestParameters tp);
      double ABCTest(TestParameters tp);
};

#endif
