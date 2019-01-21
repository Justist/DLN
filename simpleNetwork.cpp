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

Network makeNetwork(const uint16_t inputs,
                    const uint16_t hiddenLayers,
                    const uint16_t hiddenNodes,
                    const uint16_t outputs,
                    const double alpha,
                    const uint16_t seed,
                    const std::string& scheme = "") {
   /*
    * Construct a network using the parameters.
    * This is basically only calling the functions
    * which set the weights.
    * If a scheme is used, first schemeVector is filled
    * so that the scheme is reflected in the weights.
    * The '+ 1' after inputs and hiddenNodes is to account
    * for the bias nodes, which are added to the network.
    */
   const auto hiddenPlusBias = static_cast<const uint16_t>(hiddenNodes + 1);
   vecvecdo wFI(inputs + 1, vecdo(hiddenNodes));
   std::vector< vecvecdo > wHL(hiddenLayers,
                          vecvecdo(hiddenPlusBias,
                                   vecdo(hiddenNodes)));
   vecvecdo wTO(hiddenPlusBias, vecdo(outputs));
   vecdo schemeVector = {};
   if (scheme.length() > 0) {
      schemeVector = initialiseWeightsByScheme(scheme, seed);
   }
   
   // Here we construct a temporary network and feed it to this function
   Network tempNetwork(vecdo(inputs),
                       wFI,
                       vecvecdo(hiddenPlusBias,
                                vecdo(hiddenNodes)),
                       wHL,
                       wTO,
                       0.0,
                       alpha,
                       0.0,
                       scheme);
   
   tempNetwork.initialiseWeights(seed, //seed
                                 schemeVector); //scheme weights

   return tempNetwork;
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
         std::string fileName,
         const bool seedtest,
         const std::string& test) {
   /*
    * Given the Network, train the network on the
    * XOR problem in the given amount of epochs.
    * Afterwards, call the function to test it.
    */
   vecdo inputVector;
   double expectedOutput;
   uint64_t currentEpoch = 0;
   
   const std::unordered_set< std::string > acceptableTests = {
      // Might be lengthier in the future
      "xor",
      "abc"
   };
   
   assert(acceptableTests.find(test) != acceptableTests.end() && 
          "Given test is not implemented (yet)!");
   
   Tests tests;
   
   if (fileName.empty()) { fileName = "simple." + test + "output"; }
   Tests::TestParameters param(n, toFile, fileName, "a", true, seed, "");

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
         //pullScheme(n);
         if (test == "xor") {
            tests.XORTest(n, //network
                          toFile, //whether to write to a file
                          fileName.empty() ? "simple.xoroutput" :
                                             fileName, //filename
                          "a", //writing mode for the file
                          true, //seedtest
                          seed, //seed
                          "e" + std::to_string(currentEpoch) //amount of epochs
                          );
         } else if (test == "abc") {
            tests.ABCTest(n, //network
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
                const bool toFile,
                const bool seedtest,
                const std::string& test = "xor") {
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
      fileName = folder                                  +
                 "w" + scheme                            +
                 "e" + std::to_string(epochs)            +
                 "a" + General::to_string_prec(alpha, 2) +
                 "i" + std::to_string(inputs)            +
                 "l" + std::to_string(hiddenLayers)      +
                 "h" + std::to_string(hiddenNodes)       +
                 "o" + std::to_string(outputs)           +
                 ".xoroutput";
      run(
         makeNetwork(inputs,
                     hiddenLayers,
                     hiddenNodes,
                     outputs,
                     alpha,
                     seed,
                     scheme),
         epochs, seed, toFile, fileName, seedtest, test
      );
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

struct inputArgs {
   bool schemes;
   uint64_t epochs;
   double alpha;
   uint16_t seed;
   std::string test;
};

void usage(const std::string& programName) {
   printf("Usage: %s [-s] [-eadt]() [-h]", programName.c_str());
   const char* toPrint = R"(
   -s           : If given, the program uses schemes. 
   -e <epochs>  : The amount of epochs to be run.
   -a <alpha>   : The alpha of the network.
   -d <seed>    : The seed of the network.
   -t <test>    : The test to be run.
   )";
}

inputArgs parseArgs(const int argc, char **argv) {
   inputArgs ia;
   
   int c;
   
   ia.schemes = false;
   ia.epochs = 20000;
   ia.alpha = 0.5;
   ia.seed = 1230;
   ia.test = "xor";
   
   while ((c = getopt (argc, argv, "se:a:d:t:")) != -1) {
      switch (c) {
         case 's':
            ia.schemes = true;
            break;
         case 'e':
            if (optarg) { ia.epochs = std::atol(optarg); }
            break;
         case 'a':
            if (optarg) { ia.alpha = std::atof(optarg); }
            break;
         case 'd':
            if (optarg) { ia.seed = std::atoi(optarg); }
            break;
         case 't':
            if (optarg) { ia.test = optarg; }
            break;
         default:
            usage(argv[0]);
            throw("Undefined option!");
      }
    }
    
    return ia;
}

int main (const int argc, char **argv) {

   // Raise an error when one of these float exceptions occur.
   feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);

   try { 
      inputArgs ia = parseArgs(argc, argv);
   } catch (std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
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
      epochs = static_cast<uint64_t>(atoi(argv[1]));
      alpha = atof(argv[2]);
      seed = static_cast<uint16_t>(atoi(argv[3]));
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
   const bool seedtest = true;

   if (seedtest) {
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
                       epochs, s, alpha, toFile, seedtest);
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
                 toFile,
                 seedtest);
   }
   return 0;
}
