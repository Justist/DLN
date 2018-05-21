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
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace std;

typedef vector< double > vecdo;
typedef vector< vecdo > vecvecdo;

struct Network {
   vecdo inputs;
   vecvecdo weightsFromInputs;
   vecdo hiddenLayer;
   // even though the inner vector is of length 1, this enables a uniform
   // initialisation function. In the future outputsize may differ as well.
   vecvecdo weightsToOutput;
   double expectedOutput;
   double alpha;
   double calculatedOutput;
};

// Function declarations so order doesn't matter.
double sigmoid(double);
double sigmoid_d(double);
vecdo initialiseWeightsByScheme(string);
void initialiseWeights(vecvecdo&, vecvecdo&, int, int, int, vecdo);
void XOR(vecdo&, double&);
void XORTest(Network, bool, string, string, bool, int);
void writeWeights(Network, int);
void trainTheNetwork(Network&);
void testTheNetwork(Network&);

template <typename T>
std::string to_string_prec(const T a_value, const int n = 3) {
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
         weights[i] = randomWeight(seed);
      }
   }
   return weights;
}

void initialiseWeights(vecvecdo& wFI,
                       vecvecdo& wTO,
                       const unsigned int inputs,
                       const unsigned int hiddens,
                       const unsigned int outputs,
                       const unsigned int seed,
                       const vecdo scheme = {}) {
   /*
    * Initialise the weights of the given weightLayers.
    * If a scheme is given, fill the layers using that
    * scheme, else fill it with random values.
    * The layers are returned by reference.
    */
   bool useScheme = false;
   if (!scheme.empty()) { useScheme = true; }
   for (uint16_t i = 0; i < inputs; i++) {
      for (uint16_t h = 1; h < hiddens; h++) {
         wFI[i][h] =
            useScheme ?
               scheme[i*hiddens + (h - 1)] :
               randomWeight(seed);
      }
   }
   for (uint16_t h = 0; h < hiddens; h++) {
      for (uint16_t o = 0; o < outputs; o++) {
         wTO[h][o] =
            useScheme ?
               scheme[o*hiddens +
                      h +
                      (hiddens - 1)*inputs] :
               randomWeight(seed);
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
             const int seed = -1) {
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
                 "-h" + to_string(n.hiddenLayer.size( )) +
                 "-a" + to_string(n.alpha) +
                 ".xoroutput";
   }
   FILE * of = fopen(filename.c_str(), writeMode.c_str());
   double error = 0.0;
   for (int16_t i = -1; i <= 1; i += 2) {
      for (int16_t j = -1; j <= 1; j += 2) {
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

void writeWeights(Network n, const int epoch) {
   /*
    * Write the weights of the network to a file.
    * Just a testing function.
    */
   FILE *of = fopen("testweights.output", "w");
   fprintf(of, "%d: ", epoch);
   for (uint16_t i = 0; i < n.inputs.size(); i++) {
      for (uint16_t h = 1; h < n.hiddenLayer.size(); h++) { //0 is bias
         fprintf(of, "%.6f ", n.weightsFromInputs[i][h]);
      }
   }
   fprintf(of, "| ");
   for (uint16_t h = 0; h < n.hiddenLayer.size(); h++) {
      fprintf(of, "%.6f ", n.weightsToOutput[h][0]); //only 1 output
   }
   fprintf(of, "\n");
   fclose(of);
}

/*int repairNetwork(Network& n) {
   / *
    * Hacky way to ensure the weights don't surpass the
    * limits. A penalty is incurred everytime this function
    * makes changes.
    * /
   int penalty = 0;
   const double barrier = exp(-16);
   for(vecdo v1 : n.weightsFromInputs) {
      for(double d1 : v1) {
         if(d1 > 0.0 && d1 < barrier) 
            { d1 = barrier; penalty++; }
         if(d1 < 0.0 && d1 > -barrier) 
            { d1 = -barrier; penalty++; }
      }
   }
   for(vecdo v2 : n.weightsToOutput) {
      for(double d2 : v2) {
         if(d2 > 0.0 && d2 < barrier) 
            { d2 = barrier; penalty++; }
         if(d2 < 0.0 && d2 > -barrier) 
            { d2 = -barrier; penalty++; }
      }
   }
   return penalty;
}*/

inline void testTheNetwork(Network& n) {
   /*
    * Basically a forward propagation through the network.
    * n.calculatedOutput contains the result of the 
    * propagation.
    */
   unsigned int hiddenSize = n.hiddenLayer.size();

   for (uint16_t h = 1; h < hiddenSize; h++) {
      //bias has value -1
      n.hiddenLayer[h] = -n.weightsFromInputs[0][h];
      for (uint16_t i = 1; i < n.inputs.size(); i++) {
         n.hiddenLayer[h] +=
            n.weightsFromInputs[i][h] * n.inputs[i];
      }
   }
   
   // only 1 output
   n.calculatedOutput = -n.weightsToOutput[0][0];
   for (unsigned int h = 1; h < hiddenSize; h++) {
      n.calculatedOutput +=
         n.weightsToOutput[h][0] *
         sigmoid(n.hiddenLayer[h]);
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
   unsigned int hiddenSize = n.hiddenLayer.size();
   // Forward
   testTheNetwork(n);
   
   // Backward
   const double deltaOutput =
      sigmoid_d(n.calculatedOutput) *
      (n.expectedOutput - sigmoid(n.calculatedOutput));
   vecdo delta(hiddenSize, 0.0);
   // We also update the delta and weight to hiddenLayer[0]
   //here, as that saves code, but those won't be 
   // used elsewhere
   for (uint16_t h = 0; h < hiddenSize; h++) {
      delta[h] += n.weightsToOutput[h][0] * deltaOutput;
      delta[h] *= sigmoid_d(n.hiddenLayer[h]);
      n.weightsToOutput[h][0] +=
         n.alpha * sigmoid(n.hiddenLayer[h]) * deltaOutput;
      for (uint16_t i = 0; i < n.inputs.size(); i++) {
         n.weightsFromInputs[i][h] +=
            n.alpha * n.inputs[i] * delta[h];
      }
   }
}

Network makeNetwork(const unsigned int inputs,
                    const unsigned int hiddenNodes,
                    const unsigned int outputs,
                    const double alpha,
                    const unsigned int seed,
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
   vecvecdo wFI(inputs + 1, vecdo(hiddenNodes + 1));
   vecvecdo wTO(hiddenNodes + 1, vecdo(outputs));
   vecdo schemeVector = {};
   if (scheme.length() > 0) {
      schemeVector = initialiseWeightsByScheme(scheme, seed);
   }
   
   initialiseWeights(wFI, //first layer of weights
                     wTO, //second layer of weights
                     inputs + 1, //amount of input nodes
                     hiddenNodes + 1, //amount hidden nodes
                     outputs, //amount of output nodes
                     seed, //seed
                     schemeVector); //scheme weights

   return {vecdo(inputs + 1, 0), //inputs
           wFI, //weightsFromInputs
           vecdo(hiddenNodes + 1, 0), //hiddenLayer
           wTO, //weightsToOutput
           0.0, //expectedOutput
           alpha, //alpha
           0.0}; //calculatedOutput
}

unordered_set<string> generateSchemes(string scheme) {
   /*
    * First generate all the needed schemes.
    * This may get hard for larger collections of weights,
    * but when limited to about 50 different possible 
    * weights this should work.
    * Each element in the scheme string may not be larger
    * than its index + 'A', so that there will be no
    * duplicate schemes.
    */
    //Filled to include the first scheme as well
   unordered_set<string> schemes = {scheme};
   unordered_set<string> newSchemes;
   for (int i = scheme.length() - 1; i >= 0; i--) {
      scheme[i]++;
      if (scheme[i] > ('A' + i) ||
         (i > 0 && scheme[i] >
                   (scheme[i-1] + 1))) {
         return schemes;
      }
      schemes.insert(scheme);
      newSchemes = generateSchemes(scheme);
      schemes.reserve(schemes.size() +
                      distance(newSchemes.begin(), newSchemes.end()));
      schemes.insert(newSchemes.begin(), newSchemes.end());
   }
   return schemes;
}

void run(Network n,
         const uint64_t epochs,
         const unsigned int seed,
         const bool toFile,
         const string fileName = "") {
   /*
    * Given the Network, train the network on the
    * XOR problem in the given amount of epochs.
    * Afterwards, call the function to test it.
    */
   vecdo inputVector;
   double expectedOutput;
   uint64_t e = 0;

   while (e < epochs) {
      XOR(inputVector, expectedOutput);
      n.inputs = inputVector;
      n.expectedOutput = expectedOutput;
      trainTheNetwork(n);
      e++;
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
                const unsigned int inputs,
                const unsigned int hiddenNodes,
                const unsigned int outputs,
                const uint64_t epochs,
                const unsigned int seed,
                const double alpha,
                const bool toFile) {
   /*
    * Given the set of schemes, run an identical network
    * on each of the schemes for the given seed.
    * It also creates the name of the file for the results
    * to be written to.
    */
   string fileName;
   const string folder = "schemetest/";
   for (string scheme : schemes) {
      fileName = folder +
                 "w" + scheme +
                 "e" + to_string(epochs) +
                 "a" + to_string_prec(alpha, 2) +
                 "i" + to_string(inputs) +
                 "h" + to_string(hiddenNodes) +
                 "o" + to_string(outputs) +
                 ".xoroutput";
      run(
         makeNetwork(inputs,
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
   const int inputs = 2;
   const int hiddenNodes = 1;
   const int outputs = 1;
   uint64_t epochs;
   double alpha;
   unsigned int seed;
   if (argc == 4) {
      epochs = atoi(argv[1]);
      alpha = atof(argv[2]);
      seed = atoi(argv[3]);
            /*static_cast<unsigned int>
            (std::chrono::high_resolution_clock::now().
            time_since_epoch().count());*/
   } else {
      epochs = 20000;
      alpha = 0.5;
      seed = 1230;
   }

   const unsigned int amountWeights =
      ((inputs + 1) * hiddenNodes) +
      ((hiddenNodes + 1) * outputs);
   const string initialScheme(amountWeights, 'A');
   const unordered_set<string> schemes =
      generateSchemes(initialScheme);

   // Do we write the results to a file?
   const bool toFile = true;
   // Do we run the program for multiple seeds?
   const bool seedRun = true;

   if (seedRun) {
      float progress = 0.0;
      updateStatusBar(progress);
      const int startseed = 100, endseed = 1000, stepseed = 10;
      const int steps = (endseed / stepseed) - ((startseed - 1) / stepseed);
      
      vector< future< void > > threads(steps);
      
      for (unsigned int s = startseed; s <= endseed; s += stepseed) {
         threads[(s/10 - 10)] = async(launch::async,
                                      [schemes, inputs, hiddenNodes,
                                       outputs, epochs, s, alpha, toFile] {
            //runSchemes({initialScheme}, inputs, hiddenNodes, outputs,
                         // epochs, s, alpha, toFile);
            runSchemes(schemes, inputs, hiddenNodes, outputs,
                       epochs, s, alpha, toFile);
            updateStatusBar(1.0 / (float) steps);
         });
      }
   } else {
      runSchemes(schemes, inputs, hiddenNodes, outputs, epochs, seed, alpha, toFile);
   }
   /*run(makeNetwork(inputs, hiddenNodes, outputs, alpha), epochs,
         hiddenNodes, seed, alpha, toFile);*/
   return 0;
}
