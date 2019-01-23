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
         
         TestParameters(const Network& n,
                        const bool t,
                        const std::string& f,
                        const std::string& w,
                        const bool st,
                        const int s,
                        const std::string& e = "",
                        const std::string& a = "") 
                        :
                        network(n),
                        toFile(t),
                        fileName(f),
                        writeMode(w),
                        seedtest(st),
                        seed(s),
                        addition(a)
         {
            if (e != "") { epoch = e; }
         };
      };
   
      Tests() = default;
   
      void XOR(vecdo& inputs, double& output);
   
      void ABC(vecdo& inputs, double& output);
      
      void runTest(TestParameters tp, const std::string& test);
      
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
      
      void XORTest(TestParameters tp);
      void ABCTest(TestParameters tp);
};

#endif
