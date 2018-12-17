#include <algorithm>
#include <cassert>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace std;

typedef vector< double > vecdo;
typedef vector< vecdo > vecvecdo;

// Global variables to enable multithreading
unordered_set<string> globalSchemes;
mutex mtx;

struct Network {
   vecdo inputs;
   vecvecdo weightsFromInputs;
   vecvecdo hiddenLayers;
   vector< vecvecdo > weightsHiddenLayers;
   // even though the inner vector is of length 1, this enables a uniform
   // initialisation function. In the future outputsize may differ as well.
   vecvecdo weightsToOutput;
   double expectedOutput;
   double alpha;
   double calculatedOutput;
   string scheme;
};

template <typename T>
std::string to_string_prec(const T a_value, const uint8_t n = 3) {
   std::ostringstream out;
   out << std::setprecision(n) << a_value;
   return out.str();
}

inline double sigmoid(const double x) {
   return 1.0 / (1.0 + exp(-x));
}

inline double sigmoid_d(const double x) {
   const double y = sigmoid(x);
   return y * (1.0 - y);
}

inline double randomWeight(unsigned int seed) {
   return -1 + 2 * (static_cast<double>(rand_r(&seed)) /RAND_MAX);
}

inline double trueRandomWeight(unsigned int seed, vecdo pastWeights) {
   /*
    * Ensure the generated weights are 'truly' different.
    * This is ensured by having all weights differ by at least 'margin'.
    */
   const double margin = 0.01; //change if needed
   double randomNumber = -1.0;
   bool stop = false;
   while (!stop) {
      stop = true;
      randomNumber = randomWeight(seed);
      for (double weight : pastWeights) {
         if (randomNumber == weight + margin || randomNumber == weight - margin) {
            stop = false;
            break;
         }
      }
   }
   assert(randomNumber != -1.0);
   return randomNumber;
}

vecdo initialiseWeightsByScheme(const string scheme,
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
   weights[0] = randomWeight(seed);
   for (unsigned int i = 1; i < scheme.length(); i++) {
      if (current == scheme[i]) {
         weights[i] = weights[i-1];
      } else {
         weights[i] = trueRandomWeight(seed, weights);
      }
   }
   return weights;
}

void initialiseWeights(vecvecdo& wFI,
                       vector< vecvecdo >& wHL,
                       vecvecdo& wTO,
                       const uint16_t inputs,
                       const uint16_t hiddenLayers,
                       const uint16_t hiddenNodes,
                       const uint16_t outputs,
                       const uint16_t seed,
                       const vecdo schemeWeights = {}) {
   /*
    * Initialise the weights of the given weightLayers.
    * If a scheme is given, fill the layers using that
    * scheme, else fill it with random values.
    * The layers are returned by reference.
    */
   bool useScheme = false;
   if (!schemeWeights.empty()) { useScheme = true; }
   for (uint16_t i = 0; i < inputs; i++) {
      for (uint16_t h = 0; h < hiddenNodes - 1; h++) {
         wFI[i][h] =
            useScheme ?
               schemeWeights[i*hiddenNodes + (h - 1)] :
               randomWeight(seed);
      }
   }

   for (uint16_t l = 0; l < hiddenLayers - 1; l++) {
      for (uint16_t hp = 0; hp < hiddenNodes; hp++) {
         for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
            wHL[l][hp][hn] =
               useScheme ?
                  schemeWeights[l*hiddenNodes + hp + hn] :
                  randomWeight(seed);
         }
      }
   }

   for (uint16_t h = 0; h < hiddenNodes; h++) {
      for (uint16_t o = 0; o < outputs; o++) {
         wTO[h][o] =
            useScheme ?
               schemeWeights[o*hiddenNodes +
                             h +
                             (hiddenNodes - 1)*inputs] :
               randomWeight(seed);
      }
   }
}

inline vecdo flatten(vecvecdo const& toFlatten) {
   vecdo flat;
   for (vecdo sub : toFlatten) {
      flat.insert(end(flat), begin(sub), end(sub));
   }
   return flat;
}

inline vecdo flatten(vector< vecvecdo > const& toFlatten) {
   vecdo flat;
   for (auto& sub : toFlatten) {
      vecdo flatSub = flatten(sub);
      flat.insert(end(flat), begin(flatSub), end(flatSub));
   }
   return flat;
}

inline void pullScheme(Network& n) {
   /*
    * "Pull" the weights of the network together according to the scheme of the network.
    * This means that the weights which have been assigned the same letter will get a
    * small nudge to come closer to each other.
    * The nudge is half the distance to the average of their weights.
    */
    string scheme = n.scheme;
    unsigned int schemeLength = scheme.length();
    vecdo weightSums(schemeLength, 0.0);
    auto wFIFlat = flatten(n.weightsFromInputs);
    auto wHLFlat = flatten(n.weightsHiddenLayers);
    auto wTOFlat = flatten(n.weightsToOutput);
    vecdo allWeightsFlat = flatten({wFIFlat, wHLFlat, wTOFlat});
    vector<unsigned int> letterCount(schemeLength, 0);
    unsigned int index = 0;
    
    for (unsigned int i = 0; i < schemeLength; i++) {
       index = scheme[i] - 'A';
       weightSums[index] += allWeightsFlat[i];
       letterCount[index]++;
    }
    
    // Take the averages of the weightSums
    vecdo weightAverages(schemeLength, 0.0);
    for (unsigned int j = 0; j < schemeLength; j++) {
       index = scheme[j] - 'A';
       weightAverages[index] = letterCount[index] > 0 ? weightSums[index] / 
       letterCount[index] : 0;
    }
    
    // Then use these to nudge the weights
    for (unsigned int k = 0; k < schemeLength; k++) {
      index = scheme[k] - 'A';
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
    initialiseWeights(n.weightsFromInputs,
                      n.weightsHiddenLayers,
                      n.weightsToOutput,
                      n.weightsFromInputs.size(), //amount of input nodes
                      n.weightsHiddenLayers.size(), //amount of hidden layers
                      n.weightsHiddenLayers[0].size(), //amount hidden nodes
                      n.weightsToOutput[0].size(), //amount of output nodes
                      0, //seed (not relevant in this case)
                      allWeightsFlat); //scheme weights
}

inline void testTheNetwork(Network& n) {
   /*
    * Basically a forward propagation through the network.
    * n.calculatedOutput contains the result of the
    * propagation.
    */
   const uint16_t hiddenLayers = n.hiddenLayers.size();
   const uint16_t hiddenNodes = n.hiddenLayers[0].size();

   for (uint16_t h = 0; h < hiddenNodes - 1; h++) {
      //bias has value -1
      n.hiddenLayers[0][h + 1] = -n.weightsFromInputs[0][h];
      for (uint16_t i = 1; i < n.inputs.size(); i++) {
         n.hiddenLayers[0][h + 1] +=
            n.weightsFromInputs[i][h] * n.inputs[i];
      }
   }

   //hp is hidden previous
   //hn is hidden next
   //for the previous and next hidden layer
   for (uint16_t l = 0; l < hiddenLayers - 1; l++) {
      for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
         //bias has value -1
         n.hiddenLayers[l + 1][hn + 1] = -n.weightsHiddenLayers[l + 1][0][hn];
         for (uint16_t hp = 1; hp < hiddenNodes; hp++) {
            n.hiddenLayers[l + 1][hn + 1] +=
               n.weightsHiddenLayers[l][hp][hn] * sigmoid(n.hiddenLayers[l][hp]);
         }
      }
   }

   // only 1 output
   n.calculatedOutput = -n.weightsToOutput[0][0];
   for (uint16_t h = 1; h < hiddenNodes; h++) {
      n.calculatedOutput +=
         n.weightsToOutput[h][0] *
         sigmoid(n.hiddenLayers[hiddenLayers - 1][h]);
   }
}


void trainTheNetwork(Network& n) {
   /*
    * Both forward and backward propagation through the
    * network. First the forward propagation is done in
    * testTheNetwork(), as testing it is done by forward
    * propagation.
    * For the backward propagation some optimisation may
    * be possible, but it works for now.
    */
   const uint16_t hiddenLayers = n.hiddenLayers.size();
   const uint16_t hiddenNodes = n.hiddenLayers[0].size();

   // Forward
   testTheNetwork(n);

   // Backward
   const double deltaOutput =
      sigmoid_d(n.calculatedOutput) *
      (n.expectedOutput - sigmoid(n.calculatedOutput));
   vecvecdo deltas(hiddenLayers, vecdo(hiddenNodes, 0.0));

   for (uint16_t h = 0; h < hiddenNodes; h++) {
      deltas[hiddenLayers - 1][h] += n.weightsToOutput[h][0] * deltaOutput;
      deltas[hiddenLayers - 1][h] *=
         sigmoid_d(n.hiddenLayers[hiddenLayers - 1][h]);
      n.weightsToOutput[h][0] +=
         n.alpha * sigmoid(n.hiddenLayers[hiddenLayers - 1][h]) * deltaOutput;
   }

   for (int16_t l = hiddenLayers - 2; l >= 0; l--) {
      for (uint16_t hp = 0; hp < hiddenNodes; hp++) {
         for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
            deltas[l][hp] +=
               n.weightsHiddenLayers[l][hp][hn] * deltas[l + 1][hn + 1];
         }
         deltas[l][hp] *=
            sigmoid_d(n.hiddenLayers[l][hp]);
         for (uint16_t hn = 0; hn < hiddenNodes - 1; hn++) {
            n.weightsHiddenLayers[l][hp][hn] +=
               n.alpha * sigmoid(n.hiddenLayers[l][hp]) * deltas[l + 1][hn + 1];
         }
      }
   }

   for (uint16_t h = 0; h < hiddenNodes - 1; h++) {
      for (uint16_t i = 0; i < n.inputs.size(); i++) {
         // Totally not cheating around small numbers
         double weight = n.weightsFromInputs[i][h];
         double addition = n.alpha * n.inputs[i] * deltas[0][h + 1];
         if (weight > 0 && weight < pow(10, -100)) {
            n.weightsFromInputs[i][h] = addition;
         } else if (weight < 0 && weight > pow(-10, -100)) {
            n.weightsFromInputs[i][h] = addition;
         } else {
            n.weightsFromInputs[i][h] += addition;
         }
      }
   }
}

inline void XOR(vecdo& inputs, double& output) {
   /*
    * Create input and expected output for the XOR
    * problem. Two numbers are generated, either 0 or 1,
    * and if they are equal then the expected output is 0,
    * otherwise 1.
    * If the numbers are 0, they are changed to -1 so the
    * network can use these numbers.
    */
   int a = (rand ( ) % 2 == 0);
   int b = (rand ( ) % 2 == 0);
   output = (a + b) % 2;
   if (a == 0) { a = -1; }
   if (b == 0) { b = -1; }
   inputs = {-1.0, static_cast<double>(a), static_cast<double>(b)};
}

void XORTest(Network n,
             const bool toFile = false,
             string filename = "",
             const string writeMode = "w",
             const bool seedTest = false,
             const int seed = -1,
             const string addition = "") {
   /*
    * Given the trained network, calculate the error by
    * doing one forward propagation and comparing the
    * output with the expected output for each possible
    * input.
    * The error is then either printed to a file or to
    * the terminal.
    */
   if (filename == "") {
      filename = "i" + to_string(n.inputs.size( )) +
                 "l" + to_string(n.hiddenLayers.size()) +
                 "h" + to_string(n.hiddenLayers[0].size( )) +
                 "a" + to_string(n.alpha) +
                 ".xoroutput";
   }
   filename.insert(filename.find(".xoroutput"), addition);
   FILE * of = fopen(filename.c_str(), writeMode.c_str());
   double error = 0.0;
   for (int8_t i = -1; i <= 1; i += 2) {
      for (int8_t j = -1; j <= 1; j += 2) {
         n.inputs = {-1.0, static_cast<float>(i), static_cast<float>(j)};
         n.expectedOutput = (i != j);
         testTheNetwork(n);
         error += abs(n.expectedOutput -
                      sigmoid(n.calculatedOutput));
         if (!seedTest) {
            if (toFile) {
               fprintf(of,
                       "x: %d, y: %d, gives %.6f\n",
                       i, j, sigmoid(n.calculatedOutput));
            } else {
               printf("x: %d, y: %d, gives %.6f\n",
                      i, j, sigmoid(n.calculatedOutput)); 
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

Network makeNetwork(const uint16_t inputs,
                    const uint16_t hiddenLayers,
                    const uint16_t hiddenNodes,
                    const uint16_t outputs,
                    const double alpha,
                    const uint16_t seed,
                    const string scheme = "") {
   /*
    * Construct a network using the parameters.
    * This is basically only calling the functions
    * which set the weights.
    * If a scheme is used, first schemeVector is filled
    * so that the scheme is reflected in the weights.
    * The '+ 1' after inputs and hiddenNodes is to account
    * for the bias nodes, which are added to the network.
    */
   const uint16_t hiddenPlusBias = hiddenNodes + 1;
   vecvecdo wFI(inputs + 1, vecdo(hiddenNodes));
   vector< vecvecdo > wHL(hiddenLayers,
                          vecvecdo(hiddenPlusBias,
                                   vecdo(hiddenNodes)));
   vecvecdo wTO(hiddenPlusBias, vecdo(outputs));
   vecdo schemeVector = {};
   if (scheme.length() > 0) {
      schemeVector = initialiseWeightsByScheme(scheme, seed);
   }

   initialiseWeights(wFI, //first layer of weights
                     wHL, //layers of hidden weights
                     wTO, //last layer of weights
                     inputs + 1, //amount of input nodes
                     hiddenLayers, //amount of hidden layers
                     hiddenNodes + 1, //amount hidden nodes
                     outputs, //amount of output nodes
                     seed, //seed
                     schemeVector); //scheme weights

   return {vecdo(inputs + 1, 0), //inputs
           // Weights from the input layer to the first hidden layer
           wFI,
           // The nodes in the hidden layers
           vecvecdo(hiddenLayers, vecdo(hiddenNodes + 1, 0)),
           // The weights between each consequent pair of hidden layers
           wHL,
           // Weights from the last hidden layer to the output layer
           wTO,
           // The expected output
           0.0,
           // The learning rate alpha
           alpha,
           // The calculated output, or just whatever the network produces
           0.0,
           // The scheme according to which the weights have been distributed
           scheme};
}

unordered_set<string> generateInitialSchemes(string scheme, uint8_t multitask = 0) {
   /*
    * First generate all the needed schemes.
    * This may get hard for larger collections of weights
    * due to ASCII limitations, but when limited to about
    * 50 different possible weights this should work.
    * Each element in the scheme string may not be larger
    * than its index + 'A', so that there will be no
    * duplicate schemes.
    */
   unordered_set<string> schemes = {scheme};
   unordered_set<string> newSchemes;
   const uint16_t schemeLength = scheme.length();
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

unordered_set<string> multiTaskSchemes(uint16_t length) {
   /*
    * Generate the schemes using threads, so it will take less time.
    */
   unordered_set<string> allSchemes = {};
   char startLetter = 'A';
   string startScheme;
   vector< future< void > > threads(length);
   for(uint16_t i = 0; i < length; i++) {
      startScheme = startLetter * length;
      threads[i] = async(launch::async,
                         [allSchemes,
                          startScheme]
                         {
         generateInitialSchemes(startScheme, 1); 
      });
   }
}

void run(Network n,
         const uint64_t maxEpochs,
         const uint16_t seed,
         const bool toFile,
         string fileName = "") {
   /*
    * Given the Network, train the network on the
    * XOR problem in the given amount of epochs.
    * Afterwards, call the function to test it.
    */
   vecdo inputVector;
   double expectedOutput;
   uint64_t currentEpoch = 0;

   while (currentEpoch < maxEpochs) {
      XOR(inputVector, expectedOutput);
      n.inputs = inputVector;
      n.expectedOutput = expectedOutput;
      trainTheNetwork(n);
      currentEpoch++;
      if (currentEpoch % (maxEpochs / 20) == 0) {
         if (fileName != "") {
            fileName = regex_replace(fileName,
                                     regex("e" + to_string(maxEpochs)),
                                     "e" + to_string(currentEpoch));
         }
         pullScheme(n);
         XORTest(n, //network
                 toFile, //whether to write to a file
                 (fileName == "") ? "simple.xoroutput" :
                                    fileName, //filename
                 "a", //writing mode for the file
                 true, //seedtest
                 seed, //seed
                 "e" + to_string(currentEpoch) //amount of epochs
                 );
      }
   }

   //For the seedtest
   XORTest(n, //network
           toFile, //whether to write to a file
           (fileName == "") ? "simple.xoroutput" :
                              fileName, //filename
           "a", //writing mode for the file
           true, //seedtest
           seed); //seed
}

void runSchemes(const unordered_set<string> schemes,
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
   string fileName;
   const string folder = "nudgingtest2hidden3/";
   __attribute__((unused)) const uint16_t unused =
      system(("mkdir " + folder).c_str());
   for(string scheme : schemes) {
      fileName = folder +
                 "w" + scheme +
                 "e" + to_string(epochs) +
                 "a" + to_string_prec(alpha, 2) +
                 "i" + to_string(inputs) +
                 "l" + to_string(hiddenLayers) +
                 "h" + to_string(hiddenNodes) +
                 "o" + to_string(outputs) +
                 ".xoroutput";
      run(
         makeNetwork(inputs,
                     hiddenLayers,
                     hiddenNodes,
                     outputs,
                     alpha,
                     seed,
                     scheme),
         epochs, seed, toFile, fileName
      );
   }
}

float progress = 0.0; //global
void updateStatusBar(const float percent) {
   /*
    * Update the progressbar by percent percent, then
    * reprint it.
    * This seems to currently print on multiple lines,
    * but that might be platform specific.
    */
   progress += percent;

   const int barWidth = 70;
   const float amountProg = barWidth * progress;

   cout << "[" << string(amountProg, '#')
        << string(barWidth - amountProg, ' ') << "] "
        << int(progress * 100.0) << "%\r";
   cout.flush();
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
   const string initialScheme(amountWeights, 'A');
   const unordered_set<string> schemes = generateInitialSchemes(initialScheme);
   
   // Do we write the results to a file?
   const bool toFile = true;
   // Do we run the program for multiple seeds?
   const bool seedRun = true;

   if (seedRun) {
      float progress = 0.0;
      updateStatusBar(progress);
      const uint16_t startseed = 100, endseed = 1000, stepseed = 10;
      const uint32_t steps = (endseed / stepseed) - ((startseed - 1) / stepseed);

      vector< future< void > > threads(steps);

      for (uint16_t s = startseed; s <= endseed; s += stepseed) {
         threads[(s/10 - 10)] = async(launch::async,
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
            updateStatusBar(1.0 / (float) steps);
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
