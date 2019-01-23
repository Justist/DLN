#include "Tests.hpp"

template <typename T>
void Tests::Print(FILE* of,
                  const T toWrite,
                  const bool toFile) {
   std::ostringstream oss;
   oss << toWrite;
   if (toFile) {
      fprintf(of, "%s,", oss.str().c_str());
   } else {
      printf("%s,", oss.str().c_str());
   }
}

void Tests::PrintResults(const vecvecdo& inputs,
                         const vecdo& outputs,
                         const bool toFile = false,
                         const std::string& filename = "",
                         const std::string& writeMode = "w",
                         const std::string& firstString = "In: ",
                         const std::string& secondString = "Out: ",
                         const bool equalSize = true) {
   /*
    * Prints the inputs and outputs of the
    */
   unsigned long inputSize  = inputs.size();
   unsigned long outputSize = outputs.size();
   if (equalSize) {
      assert(inputSize == outputSize && 
      "No equal size of input and output vector!");
   }
   FILE *of = nullptr;
   if (toFile) {
      of = fopen(filename.c_str(), writeMode.c_str());
   }
   for (auto i = 0; i < inputSize; i++) {
      Print(of, firstString, toFile);
      for (double x : inputs[i]) {
         Print(of, x, toFile);
      }
      Print(of, secondString, toFile);
      Print(of, outputs[i], toFile);
      Print(of, "\n", toFile);
   }
}

void Tests::XOR(vecdo& inputs, double& output) {
   /*
    * Create input and expected output for the XOR
    * problem. Two numbers are generated, either 0 or 1,
    * and if they are equal then the expected output is 0,
    * otherwise 1.
    * If the numbers are 0, they are changed to -1 so the
    * network can use these numbers.
    */
   int a = rand() % 2 == 0;
   int b = rand() % 2 == 0;
   output = (a + b) % 2;
   if (a == 0) { a = -1; }
   if (b == 0) { b = -1; }
   inputs = {-1.0, static_cast<double>(a), static_cast<double>(b)};
}

void Tests::ABC(vecdo& inputs, double& output) {
   /*
    * Create input and expected output for the ABC-formula.
    * This is a formula to calculate how many times a line,
    * drawn by a formula of the form y = ax^2 + bx + c where a
    * is unequal to 0, has a value for x where y is equal to 0.
    * This function creates 3 numbers between -100 and 100,
    * which correspond with the a, b, and c mentioned above.
    * Then it calculates how many values of x give a value
    * of y equal to 0 using the following formulas:
    * x = (-b + sqrt(b^2 - 4ac)) / 2a and
    * x = (-b - sqrt(b^2 - 4ac)) / 2a
    */
   const int16_t min = -100;
   const uint16_t max = 100;
   auto a = static_cast<int16_t>(min + (rand() % max - min + 1));
   while (a == 0) { a = static_cast<int16_t>(min + (rand() % max - min + 1)); }
   const auto b = static_cast<int16_t>(min + (rand() % max - min + 1));
   const auto c = static_cast<int16_t>(min + (rand() % max - min + 1));
   
   inputs = {-1.0,
             static_cast<double>(a),
             static_cast<double>(b),
             static_cast<double>(c)};
   
   const double x1 = (-b + sqrt(b * b - 4 * a * c)) / 2 * a;
   const double x2 = (-b - sqrt(b * b - 4 * a * c)) / 2 * a;
   
   output = static_cast<double>((x1 == 0.0) + (x2 == 0.0));
}

void Tests::runTest(TestParameters tp, const std::string& test) {
   if(test == "xor") { XORTest(tp); }
   if(test == "abc") { ABCTest(tp); }
}

void Tests::XORTest(TestParameters tp) {
   /*
    * Given the trained network, calculate the error by
    * doing one forward propagation and comparing the
    * output with the expected output for each possible
    * input.
    * The error is then either printed to a file or to
    * the terminal.
    */
   
   vecvecdo inputs;
   vecdo outputs;
   
   Network n = tp.network;
   
   if (tp.fileName.empty()) {
      tp.fileName = "i" + std::to_string(n.amInputNodes()) +
                    "l" + std::to_string(n.amHiddenLayers()) +
                    "h" + std::to_string(n.amHiddenNodes()) +
                    "a" + std::to_string(n.alpha()) +
                    ".xoroutput";
   }
   tp.fileName.insert(tp.fileName.find(".xoroutput"), tp.addition);
   FILE * of = fopen(tp.fileName.c_str(), tp.writeMode.c_str());
   double outputDifference;
   double error = 0.0;
   for (float i = -1; i <= 1; i += 2) {
      for (float j = -1; j <= 1; j += 2) {
         n.inputs({-1.0, i, j});
         n.expectedOutput(i != j);
         n.forward();
         outputDifference = n.expectedOutput() -
                            General::sigmoid(n.calculatedOutput());
         error += outputDifference > 0 ? outputDifference : 1.0 - outputDifference;
         if (!tp.seedtest) {
            inputs.push_back({i, j});
            outputs.push_back(General::sigmoid(n.calculatedOutput()));
         }
      }
   }
   PrintResults(inputs, outputs, tp.toFile, tp.fileName, tp.writeMode);
   inputs.clear();
   outputs.clear();
   outputs.push_back(error);
   if (tp.seedtest) {
      inputs.push_back({static_cast<double>(tp.seed)});
      PrintResults(inputs, outputs, tp.toFile, tp.fileName, 
                   tp.writeMode, "seed: ", "error: ");
   } else {
      PrintResults(inputs, outputs, tp.toFile, tp.fileName, 
                   tp.writeMode, "", "error: ", false);
   }
   fclose(of);
}

void Tests::ABCTest(TestParameters tp) {
   /*
    * Tests the trained networks performance on calculating the
    * ABC formula. To do this, a few sets of variables are each tested
    * on the network, and then the answer supplied by the network
    * is compared to the answer it should be, and the difference
    * between the two makes the error.
    * Each set of variables consists of values a, b, and c, which
    * correspond to the function y = ax^2 + bx + c. The outcome
    * of the network should correspond with the amount of values
    * x can have for which y is equal to 0, given that a is unequal
    * to 0.
    * For each possible outcome, being 0, 1, or 2, there will be
    * multiple sets of variables to ensure the network works well
    * on both positive and negative inputs.
    */
   vecvecdo inputs;
   vecdo outputs;
   
   Network n = tp.network;
   
   if (tp.fileName.empty()) {
      tp.fileName = "i" + std::to_string(n.amInputNodes()) +
                    "l" + std::to_string(n.amHiddenLayers()) +
                    "h" + std::to_string(n.amHiddenNodes()) +
                    "a" + std::to_string(n.alpha()) +
                    ".abcoutput";
   }
   tp.fileName.insert(tp.fileName.find(".abcoutput"), tp.addition);
   FILE * of = fopen(tp.fileName.c_str(), tp.writeMode.c_str());
   
   const vecvecdo testcases = {
        // a, b, c, output
        {9, 12, 5, 0},
        {20, 1, 20, 0},
        {1, 10, 25, 1},
        {1, -2, 1, 1},
        {5, -44, 1, 2},
        {-1, 30, -8, 2},
        {-5, -20, -4, 2},
        {3, 8, 4, 2},
   };
   
   double outputDifference;
   double error = 0.0;
   
   for (vecdo test : testcases) {
      n.inputs({-1.0, test[0], test[1], test[2]});
      n.expectedOutput(test[3]);
      n.forward();
      outputDifference = n.expectedOutput() -
                         General::sigmoid(n.calculatedOutput());
      error += outputDifference > 0 ? outputDifference : 1.0 - outputDifference;
      if (!tp.seedtest) {
         inputs.push_back({test[0], test[1], test[2]});
         outputs.push_back(General::sigmoid(n.calculatedOutput()));
      }
   }
   
   PrintResults(inputs, outputs, tp.toFile, tp.fileName, tp.writeMode);
   inputs.clear();
   outputs.clear();
   outputs.push_back(error);
   if (tp.seedtest) {
      inputs.push_back({static_cast<double>(tp.seed)});
      PrintResults(inputs, 
                   outputs, 
                   tp.toFile, 
                   tp.fileName, 
                   tp.writeMode, 
                   "seed: ", 
                   "error: ");
   } else {
      PrintResults(inputs, 
                   outputs, 
                   tp.toFile, 
                   tp.fileName, 
                   tp.writeMode, 
                   "", 
                   "error: ", 
                   false);
   }
   fclose(of);
}
