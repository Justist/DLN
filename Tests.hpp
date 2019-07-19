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
         const char *writeMode;
         bool seedtest;
         int seed;
         std::string epoch;
         std::string addition;
         
         TestParameters(Network n,
                        const bool t,
                        std::string f,
                        const char *w,
                        const bool st,
                        const int s,
                        const std::string& e = "",
                        std::string a = "")
                        :
                        network(std::move(n)),
                        toFile(t),
                        fileName(std::move(f)),
                        writeMode(w),
                        seedtest(st),
                        seed(s),
                        addition(std::move(a))
         {
            if (!e.empty()) { epoch = e; }
         };
      };
   
      Tests() = default;
   
      void runSmallTest(vecdo& inputs, 
                               double& output, 
                               const std::string& test);
      
      double runTest(TestParameters tp, 
                     const std::string& test,
                     bool print = true);
      
   private:
      template <typename T>
      void Print(FILE* of,
                 T toWrite,
                 bool toFile);
   
      void PrintResults(const vecvecdo& inputs,
                        const vecdo& outputs,
                        bool toFile = false,
                        const std::string& filename = "",
                        const char *writeMode = "w",
                        const std::string& firstString = "In: ",
                        const std::string& secondString = "Out: ",
                        bool equalSize = true);

      void XOR(vecdo& inputs, double& output);
      void ABC(vecdo& inputs, double& output);
      double ABCFormula(int16_t a,
                        int16_t b,
                        int16_t c,
                        double x);

      double XORTest(TestParameters tp, bool print);
      double ABCTest(TestParameters tp, bool print);
};

#endif
