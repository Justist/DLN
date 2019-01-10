#include "Tests.hpp"

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

void Tests::XORTest(Network n,
                    const bool toFile/* = false*/,
                    std::string filename/* = ""*/,
                    const std::string& writeMode/* = "w"*/,
                    const bool seedTest/* = false*/,
                    const int seed/* = -1*/,
                    const std::string& addition/* = ""*/) {
   /*
    * Given the trained network, calculate the error by
    * doing one forward propagation and comparing the
    * output with the expected output for each possible
    * input.
    * The error is then either printed to a file or to
    * the terminal.
    */
   if (filename.empty()) {
      filename = "i" + std::to_string(n.amInputNodes()) +
                 "l" + std::to_string(n.amHiddenLayers()) +
                 "h" + std::to_string(n.amHiddenNodes()) +
                 "a" + std::to_string(n.alpha()) +
                 ".xoroutput";
   }
   filename.insert(filename.find(".xoroutput"), addition);
   FILE * of = fopen(filename.c_str(), writeMode.c_str());
   double error = 0.0;
   for (int8_t i = -1; i <= 1; i += 2) {
      for (int8_t j = -1; j <= 1; j += 2) {
         n.inputs({-1.0, static_cast<float>(i), static_cast<float>(j)});
         n.expectedOutput(i != j);
         n.forward();
         error += abs(n.expectedOutput() -
                      General::sigmoid(n.calculatedOutput()));
         if (!seedTest) {
            if (toFile) {
               fprintf(of,
                       "x: %d, y: %d, gives %.6f\n",
                       i, j, General::sigmoid(n.calculatedOutput()));
            } else {
               printf("x: %d, y: %d, gives %.6f\n",
                      i, j, General::sigmoid(n.calculatedOutput()));
            }
         }
      }
   }
   if (toFile) {
      if (seedTest) {
         fprintf(of, "seed: %d, error: %.6f\n", seed, error);
      } else {
         fprintf(of, "error: %.6f\n", error);
      }
   } else { printf("error: %.6f\n", error); }
   fclose(of);
}
