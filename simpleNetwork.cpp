#include "Includes.hpp"

#include "General.cpp"
#include "Network.hpp"
#include "Tests.hpp"

/*// Global variables to enable multithreading
unordered_set<std::string> globalSchemes;
mutex mtx;*/

struct InputArgs {
   bool schemes;
   uint8_t layers;
   uint8_t nodes;
   uint64_t epochs;
   double alpha;
   uint16_t seed;
   std::string test;
   std::string folder;
};

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
   const auto hiddenPlusBias = static_cast<uint16_t>(hiddenNodes + 1);
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
   const auto schemeLength = static_cast<uint16_t>(scheme.length());
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
         const std::string& test,
         const bool convergenceTest = true) {
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
   double error;
   
   if (fileName.empty()) { fileName = "simple." + test + "output"; }
   Tests::TestParameters param(n, toFile, fileName, "a", true, seed, "");

   while (currentEpoch < maxEpochs) {
      tests.runSmallTest(inputVector, expectedOutput, test);
      n.inputs(inputVector);
      n.expectedOutput(expectedOutput);
      n.train();
      param.network = n;
      currentEpoch++;
      if (convergenceTest && currentEpoch % 10 == 0) {
         error = tests.runTest(param, test, false);
         if (error < 0.1) {
            if (!param.fileName.empty()) {
            param.fileName = regex_replace(param.fileName,
                                           std::regex("e" +
                                                      std::to_string(maxEpochs)),
                                           "e" + std::to_string(currentEpoch));
            }
            tests.runTest(param, test, true); //to print the result
            break;
         }
      } if (currentEpoch % (maxEpochs / 20) == 0) {
         if (!param.fileName.empty()) {
            param.fileName = regex_replace(param.fileName,
                                           std::regex("e" +
                                                      std::to_string(maxEpochs)),
                                           "e" + std::to_string(currentEpoch));
         }
         tests.runTest(param, test, true);
      }
   }
}

void runSchemes(const std::unordered_set<std::string> schemes,
                const InputArgs ia,
                const uint16_t inputs,
                const uint16_t outputs,
                const bool toFile,
                const bool seedtest,
                const uint64_t seed) {
   /*
    * Given the set of schemes, run an identical network
    * on each of the schemes for the given seed.
    * It also creates the name of the file for the results
    * to be written to.
    */
   std::string fileName;
   __attribute__((unused)) const auto unused =
               static_cast<const uint16_t>(system(("mkdir " + ia.folder).c_str()));
   for(const std::string& scheme : schemes) {
      fileName = ia.folder                                  +
                 "w" + scheme                               +
                 "e" + std::to_string(ia.epochs)            +
                 "a" + General::to_string_prec(ia.alpha, 2) +
                 "i" + std::to_string(inputs)               +
                 "l" + std::to_string(ia.layers)            +
                 "h" + std::to_string(ia.nodes)             +
                 "o" + std::to_string(outputs)              +
                 "." + ia.test                              + 
                 "output";
      run(
         makeNetwork(inputs,
                     ia.layers,
                     ia.nodes,
                     outputs,
                     ia.alpha,
                     ia.seed,
                     scheme),
         ia.epochs, ia.seed, toFile, fileName, seedtest, ia.test
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
   const auto amountProg = static_cast<unsigned long>(barWidth * progress);

   std::cout << "[" << std::string(amountProg, '#')
             << std::string(barWidth - amountProg, ' ') << "] "
             << int(progress * 100.0) << "%\r";
   std::cout.flush();
}

void usage(const std::string& programName) {
   printf("Usage: %s [-s] [-lneadt]() [-h]\n", programName.c_str());
   const char* toPrint = R"(
   -s            : If given, the program uses schemes.
   -l <integer>  : The amount of hidden layers in the network.
   -n <integer>  : The amount of hidden nodes in each hidden layer.
   -e <integer>  : The amount of epochs to be run.
   -a <double>   : The alpha of the network.
   -d <integer>  : The seed of the network.
   -t <string>   : The test to be run.
   -f <string>   : The name of the folder to store the results in.
   )";
   printf("%s\n", toPrint);
}

InputArgs parseArgs(const int argc, char **argv) {
   InputArgs ia;
   
   int c;
   
   ia.schemes = false;
   ia.layers = 2;
   ia.nodes = 2;
   ia.epochs = 20000;
   ia.alpha = 0.5;
   ia.seed = 1230;
   ia.test = "xor";
   ia.folder = "output/";
   
   while ((c = getopt (argc, argv, "sl:n:e:a:d:t:f:")) != -1) {
      switch (c) {
         case 's':
            ia.schemes = true;
            break;
         case 'l':
            if (optarg) { ia.layers = static_cast<uint8_t>(
                                        std::atoi(optarg)); }
            break;
         case 'n':
            if (optarg) { ia.nodes  = static_cast<uint8_t>(
                                        std::atoi(optarg)); }
            break;
         case 'e':
            if (optarg) { ia.epochs = static_cast<uint64_t>(
                                        std::atol(optarg)); }
            break;
         case 'a':
            if (optarg) { ia.alpha  = std::atof(optarg); }
            break;
         case 'd':
            if (optarg) { ia.seed   = static_cast<uint16_t>(
                                        std::atoi(optarg)); }
            break;
         case 't':
            if (optarg) { ia.test   = optarg; }
            break;
         case 'f':
            if (optarg) { ia.folder = optarg; }
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

   InputArgs ia;

   try { 
      ia = parseArgs(argc, argv);
   } catch (std::exception& e) {
      fprintf(stderr, "%s\n", e.what());
      return -1;
   }

   // + 1 for the bias node
   const uint8_t inputs = 3;
   const uint8_t outputs = 1;

   const auto hiddenPlusBias = static_cast<uint16_t>(ia.nodes + 1);
   const auto amountWeights =
           static_cast<uint16_t>(
                   ((inputs + 1) * ia.nodes)                     +
                   (hiddenPlusBias * ia.nodes * (ia.layers - 1)) +
                   (hiddenPlusBias * outputs));
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
                                       ia,
                                       inputs,
                                       outputs,
                                       s,
                                       toFile] {
            runSchemes(schemes, ia, inputs, outputs,
                       toFile, seedtest, s);
            updateStatusBar(1.0 / (float) steps);
         });
      }
   } else {
      runSchemes(schemes,
                 ia,
                 inputs,
                 outputs,
                 toFile,
                 seedtest,
                 ia.seed);
   }
   //to prevent the statusbar from staying at the bottom of the terminal
   std::cout << std::endl; 
   return 0;
}
