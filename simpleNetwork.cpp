#include "Includes.hpp"

#include "General.cpp"
#include "Network.hpp"
#include "Tests.hpp"

/*// Global variables to enable multithreading
unordered_set<std::string> globalSchemes;
mutex mtx;*/

vecdo initialiseWeightsByScheme(const std::string& scheme,
                                const unsigned int seed) {
   /*
    * Function takes a scheme in format "aaabbbcccddd" etc,
    * where equal letters represent the same 'random' 
    * weight in that position.
    * Then it makes a vector of equal length with on each
    * position a 'random' weight, according to this scheme.
    */
   char current = scheme[0];
   vecdo weights(scheme.length(), -1.0);
   weights[0] = General::randomWeight(seed);
   for (unsigned int i = 1; i < scheme.length(); i++) {
      if (current == scheme[i]) {
         weights[i] = weights[i-1];
      } else {
         weights[i] = General::trueRandomWeight(seed, weights);
      }
   }
   return weights;
}

inline void pullScheme(Network& n) {
   /*
    * "Pull" the weights of the network together according to the scheme of the network.
    * This means that the weights which have been assigned the same letter will get a
    * small nudge to come closer to each other.
    * The nudge is half the distance to the average of their weights.
    */
    std::string scheme = n.scheme();
    auto schemeLength = static_cast<unsigned int>(scheme.length());
    vecdo weightSums(schemeLength, 0.0);
    auto wFIFlat = General::flatten(n.weightsFromInputs());
    auto wHLFlat = General::flatten(n.weightsHiddenLayers());
    auto wTOFlat = General::flatten(n.weightsToOutput());
    vecdo allWeightsFlat = General::flatten({wFIFlat, wHLFlat, wTOFlat});
    std::vector<unsigned int> letterCount(schemeLength, 0);
    unsigned int index = 0;
    
    for (unsigned int i = 0; i < schemeLength; i++) {
       index = static_cast<unsigned int>(scheme[i] - 'A');
       weightSums[index] += allWeightsFlat[i];
       letterCount[index]++;
    }
    
    // Take the averages of the weightSums
    vecdo weightAverages(schemeLength, 0.0);
    for (unsigned int j = 0; j < schemeLength; j++) {
       index = static_cast<unsigned int>(scheme[j] - 'A');
       weightAverages[index] = letterCount[index] > 0 ? weightSums[index] / 
       letterCount[index] : 0;
    }
    
    // Then use these to nudge the weights
    for (unsigned int k = 0; k < schemeLength; k++) {
      index = static_cast<unsigned int>(scheme[k] - 'A');
//      for (double w : wFIFlat) {
//         printf("%f ", w);
//      }
//      cout << endl;
//      for (double w : wHLFlat) {
//         printf("%f ", w);
//      }
//      cout << endl;
//      for (double w : wTOFlat) {
//         printf("%f ", w);
//      }
//      cout << endl;
//      cout << n.scheme << endl;
//      throw;
      
      allWeightsFlat[k] -= (allWeightsFlat[k] - 
                            weightAverages[index]) / 2.0;
    }
    
    // Then update the weights according to the flat weight vector
    n.initialiseWeights(0, //seed (not relevant in this case)
                      allWeightsFlat); //scheme weights
}

std::unordered_set<std::string> generateInitialSchemes(std::string scheme, 
                                                       uint8_t multitask = 0) {
   /*
    * First generate all the needed schemes.
    * This may get hard for larger collections of weights
    * due to ASCII limitations, but when limited to about
    * 50 different possible weights this should work.
    * Each element in the scheme string may not be larger
    * than its index + 'A', so that there will be no
    * duplicate schemes.
    */
   std::unordered_set<std::string> schemes = {scheme};
   std::unordered_set<std::string> newSchemes;
   const auto schemeLength = static_cast<const uint16_t>(scheme.length());
   for (int i = schemeLength - 1; i >= multitask; i--) {
      scheme[i]++;
      if (scheme[i] > ('A' + i) ||
         (i > 0 && scheme[i] >
                   (scheme[i-1] + 1))) {
         return schemes;
      }
      schemes.insert(scheme);
      newSchemes = generateInitialSchemes(scheme);
      schemes.reserve(schemes.size() +
                      distance(newSchemes.begin(), newSchemes.end()));
      schemes.insert(newSchemes.begin(), newSchemes.end());
   }
   return schemes;
}

/*unordered_set<std::string> multiTaskSchemes(uint16_t length) {
   *//*
    * Generate the schemes using threads, so it will take less time.
    *//*
   unordered_set<std::string> allSchemes = {};
   char startLetter = 'A';
   std::string startScheme;
   std::vector< future< void > > threads(length);
   for(uint16_t i = 0; i < length; i++) {
      startScheme = startLetter * length;
      threads[i] = async(launch::async,
                         [allSchemes,
                          startScheme]
                         {
         generateInitialSchemes(startScheme, 1); 
      });
      // TODO Some way to stitch those together
   }
}*/

void run(Network n,
         const uint64_t maxEpochs,
         const uint16_t seed,
         const bool toFile,
         std::string fileName = "") {
   /*
    * Given the Network, train the network on the
    * XOR problem in the given amount of epochs.
    * Afterwards, call the function to test it.
    */
   vecdo inputVector;
   double expectedOutput;
   uint64_t currentEpoch = 0;
   
   Tests tests;

   while (currentEpoch < maxEpochs) {
      tests.XOR(inputVector, expectedOutput);
      n.inputs(inputVector);
      n.expectedOutput(expectedOutput);
      n.train();
      currentEpoch++;
      if (currentEpoch % (maxEpochs / 20) == 0) {
         if (!fileName.empty()) {
            fileName = regex_replace(fileName,
                                     std::regex("e" + std::to_string(maxEpochs)),
                                     "e" + std::to_string(currentEpoch));
         }
         pullScheme(n);
         tests.XORTest(n, //network
                       toFile, //whether to write to a file
                       fileName.empty() ? "simple.xoroutput" :
                                          fileName, //filename
                       "a", //writing mode for the file
                       true, //seedtest
                       seed, //seed
                       "e" + std::to_string(currentEpoch) //amount of epochs
                       );
      }
   }

   //For the seedtest
   tests.XORTest(n, //network
                 toFile, //whether to write to a file
                 fileName.empty() ? "simple.xoroutput" :
                 fileName, //filename
                 "a", //writing mode for the file
                 true, //seedtest
                 seed); //seed
}

void runSchemes(const std::unordered_set<std::string> schemes,
                const uint16_t inputs,
                const uint16_t hiddenLayers,
                const uint16_t hiddenNodes,
                const uint16_t outputs,
                const uint64_t epochs,
                const uint16_t seed,
                const double alpha,
                const bool toFile) {
   /*
    * Given the set of schemes, run an identical network
    * on each of the schemes for the given seed.
    * It also creates the name of the file for the results
    * to be written to.
    */
   std::string fileName;
   const std::string folder = "nudgingtest2hidden3/";
   __attribute__((unused)) const auto unused =
               static_cast<const uint16_t>(system(("mkdir " + folder).c_str()));
   for(const std::string& scheme : schemes) {
      fileName = folder +
                 "w" + scheme                            +
                 "e" + std::to_string(epochs)            +
                 "a" + General::to_string_prec(alpha, 2) +
                 "i" + std::to_string(inputs)            +
                 "l" + std::to_string(hiddenLayers)      +
                 "h" + std::to_string(hiddenNodes)       +
                 "o" + std::to_string(outputs)           +
                 ".xoroutput";
      Network network = new Network(inputs,
                                    hiddenLayers,
                                    hiddenNodes,
                                    outputs,
                                    alpha,
                                    seed,
                                    scheme);
      run(network, epochs, seed, toFile, fileName);
   }
}

float progress = 0.0; //global
void updateStatusBar (const double percent) {
   /*
    * Update the progressbar by percent percent, then
    * reprint it.
    * This seems to currently print on multiple lines,
    * but that might be platform specific.
    */
   progress += percent;

   const int barWidth = 70;
   const auto amountProg = static_cast<const unsigned long>(barWidth * progress);

   std::cout << "[" << std::string(amountProg, '#')
             << std::string(barWidth - amountProg, ' ') << "] "
             << int(progress * 100.0) << "%\r";
   std::cout.flush();
}

int main (const int argc, const char **argv) {

   // Raise an error when one of these float exceptions occur.
   feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);

   if (!(argc == 4 || (argc == 2 && !strcmp(argv[1], "schemes")))) {
      printf("Usage: %s <epochs> <alpha> <seed>\n", argv[0]);
      return 1;
   }

   // + 1 for the bias node
   const uint16_t inputs = 2;
   const uint16_t hiddenLayers = 2;
   const uint16_t hiddenNodes = 2;
   const uint16_t outputs = 1;
   uint64_t epochs;
   double alpha;
   uint16_t seed;
   if (argc == 4) {
      epochs = atoi(argv[1]);
      alpha = atof(argv[2]);
      seed = atoi(argv[3]);
   } else {
      epochs = 20000;
      alpha = 0.5;
      seed = 1230;
   }

   const uint16_t hiddenPlusBias = hiddenNodes + 1;
   const uint16_t amountWeights =
      ((inputs + 1) * hiddenNodes) +
      (hiddenPlusBias * hiddenNodes * (hiddenLayers - 1)) +
      (hiddenPlusBias * outputs);
   const std::string initialScheme(amountWeights, 'A');
   const std::unordered_set<std::string> schemes = 
      generateInitialSchemes(initialScheme);
   
   // Do we write the results to a file?
   const bool toFile = true;
   // Do we run the program for multiple seeds?
   const bool seedRun = true;

   if (seedRun) {
      float progress = 0.0;
      updateStatusBar(progress);
      const uint16_t startseed = 100, endseed = 1000, stepseed = 10;
      const uint32_t steps = (endseed / stepseed) - ((startseed - 1) / stepseed);

      std::vector< std::future< void > > threads(steps);

      for (uint16_t s = startseed; s <= endseed; s += stepseed) {
         threads[(s/10 - 10)] = async(std::launch::async,
                                      [schemes,
                                       inputs,
                                       hiddenLayers,
                                       hiddenNodes,
                                       outputs,
                                       epochs,
                                       s,
                                       alpha,
                                       toFile] {
            runSchemes(schemes, inputs, hiddenLayers, hiddenNodes, outputs,
                       epochs, s, alpha, toFile);
            updateStatusBar(1.0 / (const float) steps);
         });
      }
   } else {
      runSchemes(schemes,
                 inputs,
                 hiddenLayers,
                 hiddenNodes,
                 outputs,
                 epochs,
                 seed,
                 alpha,
                 toFile);
   }
   return 0;
}
