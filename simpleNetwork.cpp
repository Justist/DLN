#include <cstdlib>
#include <cstdio>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

using namespace std;

typedef vector< double > vecdo;
typedef vector< vecdo > vecvecdo;

// Global variable so it can be altered in a void function
bool sigintsent = false;

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
void writeWeights(Network, FILE *, int);
void trainTheNetwork(Network&);
void testTheNetwork(Network&);
void SIGINThandler (int);

template <typename T>
std::string to_string_prec(const T a_value, const int n = 3) {
   std::ostringstream out;
   out << std::setprecision(n) << a_value;
   return out.str();
}

double sigmoid(const double x) {
   return 1.0 / (1.0 + exp(-x));
}

double sigmoid_d(const double x) {
   double y = sigmoid(x);
   return y * (1.0 - y);
}

vecdo initialiseWeightsByScheme(string scheme) {
   /*
    * Function takes a scheme in format "aaabbbcccddd" etc,
    * where equal letters represent the same 'random' weight
    * in that position.
    * Then it makes a vector of equal length with on each
    * position a 'random' weight, according to this scheme.
    */
   char current = scheme[0];
   vecdo weights(scheme.length(), -1.0);
   weights[0] = -1 + 2 * ((double) rand() /RAND_MAX);
   for (unsigned int i = 1; i < scheme.length(); i++) {
      if (current == scheme[i]) {
         weights[i] = weights[i-1];
      } else {
         weights[i] = -1 + 2 * ((double) rand() /RAND_MAX);
      }
   }
   return weights;
}

void initialiseWeights(vecvecdo& wFI,
                       vecvecdo& wTO,
                       const int inputs,
                       const int hiddens,
                       const int outputs,
                       const vecdo scheme = {}) {
   bool useScheme = false;
   if (!scheme.empty()) { useScheme = true; }
   for (int i = 0; i < inputs; i++) {
      for (int h = 1; h < hiddens; h++) {
         wFI[i][h] = useScheme ? scheme[i*hiddens + h] : -1 + 2 * ((double) rand() /RAND_MAX);
      }
   }
   for (int h = 0; h < hiddens; h++) {
      for (int o = 0; o < outputs; o++) {
         wTO[h][o] = useScheme ? scheme[o*hiddens + h + (hiddens - 1)*inputs] : -1 + 2 * ((double) rand() /RAND_MAX);
      }
   }
}

void XOR(vecdo& inputs, double& output) {
   int a = (rand ( ) % 2 == 0);
   int b = (rand ( ) % 2 == 0);
   output = (a + b) % 2;
   if (a == 0) { a = -1; }
   if (b == 0) { b = -1; }
   inputs = {-1.0, (double) a, (double) b};
}

void XORTest(Network n, const bool toFile = false, string filename = "", const string writeMode = "w",
             const bool seedTest = false, const int seed = -1) {
   FILE * of;
   if (filename == "") {
      filename = "i" + to_string(n.inputs.size( )) +
                 "-h" + to_string(n.hiddenLayer.size( )) +
                 "-a" + to_string(n.alpha) +
                 ".xoroutput";
   }
   of = fopen(filename.c_str(), writeMode.c_str());
   double error = 0.0;
   for (int i = -1; i <= 1; i += 2) {
      for (int j = -1; j <= 1; j += 2) {
         n.inputs = {-1.0, (double) i, (double) j};
         n.expectedOutput = (i != j);
         testTheNetwork(n);
         error += abs(n.expectedOutput - sigmoid(n.calculatedOutput));
         if(!seedTest) {
            if (toFile) {
               fprintf(of, "x: %d, y: %d, gives %.6f\n", i, j, sigmoid(n.calculatedOutput));
               printf("x: %d, y: %d, gives %.6f\n", i, j, sigmoid(n.calculatedOutput));
            } else { printf("x: %d, y: %d, gives %.6f\n", i, j, sigmoid(n.calculatedOutput)); }
         }
      }
   }
   if (toFile) {
      if (seedTest) {
         fprintf(of, "seed: %d, error: %.6f\n", seed, error);
      } else {
         fprintf(of, "error: %.6f\n", error);
      }
   }
   else { printf("error: %.6f\n", error); }
   fclose(of);
}

void writeWeights(Network n, FILE * of, int epoch) {
   fprintf(of, "%d: ", epoch);
   for (unsigned int i = 0; i < n.inputs.size(); i++) {
      for (unsigned int h = 1; h < n.hiddenLayer.size(); h++) { //0 is bias
         fprintf(of, "%.6f ", n.weightsFromInputs[i][h]);
      }
   }
   fprintf(of, "| ");
   for (unsigned int h = 0; h < n.hiddenLayer.size(); h++) {
      fprintf(of, "%.6f ", n.weightsToOutput[h][0]); //only 1 output
   }
   fprintf(of, "\n");
}

void trainTheNetwork(Network& n) {
   unsigned int hiddenSize = n.hiddenLayer.size();
   // Forward
   testTheNetwork(n);
   
   // Backward
   double deltaOutput = sigmoid_d(n.calculatedOutput) * 
                        (n.expectedOutput - sigmoid(n.calculatedOutput));
   vecdo delta(hiddenSize, 0.0);
   // We also update the delta and weight to hiddenLayer[0] here, 
   // as that saves code, but those won't be used elsewhere
   for (unsigned int h = 0; h < hiddenSize; h++) {
      delta[h] += n.weightsToOutput[h][0] * deltaOutput;
      delta[h] *= sigmoid_d(n.hiddenLayer[h]);
      n.weightsToOutput[h][0] += n.alpha * sigmoid(n.hiddenLayer[h]) * deltaOutput;
      for (unsigned int i = 0; i < n.inputs.size(); i++) {
         n.weightsFromInputs[i][h] += n.alpha * n.inputs[i] * delta[h];
      }
   }
}

void testTheNetwork(Network& n) {
   unsigned int hiddenSize = n.hiddenLayer.size();

   for (unsigned int h = 1; h < hiddenSize; h++) {
      n.hiddenLayer[h] = -n.weightsFromInputs[0][h]; //bias has value -1
      for (unsigned int i = 1; i < n.inputs.size(); i++) {
         n.hiddenLayer[h] += n.weightsFromInputs[i][h] * n.inputs[i];
      }
   }
   
   n.calculatedOutput = -n.weightsToOutput[0][0]; // only 1 output
   for (unsigned int h = 1; h < hiddenSize; h++) {
      n.calculatedOutput += n.weightsToOutput[h][0] * sigmoid(n.hiddenLayer[h]);
   }
}

Network makeNetwork(const unsigned int inputs,
                    const unsigned int hiddenNodes,
                    const unsigned int outputs,
                    const double alpha,
                    const string scheme = "") {
   vecvecdo wFI(inputs + 1, vecdo(hiddenNodes + 1));
   vecvecdo wTO(hiddenNodes + 1, vecdo(outputs));
   vecdo schemeVector = {};
   if (scheme.length() > 0) {
      schemeVector = initialiseWeightsByScheme(scheme);
   }
   initialiseWeights(wFI, wTO, inputs + 1, hiddenNodes + 1, outputs, schemeVector);

   Network n = {vecdo(inputs + 1, 0), //inputs
                wFI, //weightsFromInputs
                vecdo(hiddenNodes + 1, 0), //hiddenLayer
                wTO, //weightsToOutput
                0.0, //expectedOutput
                alpha, //alpha
                0.0}; //calculatedOutput
   return n;
}

unordered_set<string> generateSchemes(string scheme) {
   /*
    * First generate all the needed schemes.
    * This may get hard for larger collections of weights,
    * but when limited to 26 different possible weights
    * this should work.
    */
   unordered_set<string> schemes = {scheme}; //to include the first scheme as well
   unordered_set<string> newSchemes = {};
   //printf("scheme: %s\n", scheme.c_str());
   for(int i = scheme.length() - 1; i >= 0; i--) {
      scheme[i]++;
      if(scheme[i] > ('A' + i) || (i > 0 && scheme[i] > (scheme[i-1] + 1))) {
         return schemes;
      }
      schemes.insert(scheme);
      newSchemes = generateSchemes(scheme);
      schemes.reserve(schemes.size() + distance(newSchemes.begin(),newSchemes.end()));
      schemes.insert(newSchemes.begin(),newSchemes.end());
   }
   return schemes;
}

void run(Network n,
         const unsigned long int epochs,
         const unsigned int seed,
         const bool toFile,
         const string fileName = "") {
   vecdo inputVector;
   double expectedOutput;
   unsigned long int e = 0;
   srand(seed);
   //printf("The program will run with %d hidden nodes, alpha %f, and seed %d\n", hiddenNodes, alpha, seed);

   while(/*!sigintsent*/e < epochs) {
      XOR(inputVector, expectedOutput);
      n.inputs = inputVector;
      n.expectedOutput = expectedOutput;
      trainTheNetwork(n);
      //writeWeights(n, of, e);
      e++;
   }

   //cout << "The program will now proceed to testing." << endl;

   //For the seedtest
   XORTest(n, toFile, (fileName == "") ? "simple.xoroutput" : fileName, "a", true, seed);
}

void runSchemes(unordered_set<string> schemes,
                const unsigned int inputs,
                const unsigned int hiddenNodes,
                const unsigned int outputs,
                const unsigned long int epochs,
                const unsigned int seed,
                const double alpha,
                const bool toFile) {
   string fileName;
   string folder = "schemetest/";
   for(auto scheme : schemes) {
      fileName = folder +
                 "w" + scheme + "e" + to_string(epochs) + "a" + to_string_prec(alpha, 2) +
                 "i" + to_string(inputs) + "h" + to_string(hiddenNodes) + "o" + to_string(outputs) +
                 ".xoroutput";
      //printf("The program will write to %s\n", fileName.c_str());
      run(makeNetwork(inputs, hiddenNodes, outputs, alpha, scheme), epochs, seed, toFile, fileName);
   }
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
   const int hiddenNodes = 4;
   const int outputs = 1;
   unsigned long int epochs;
   double alpha;
   unsigned int seed;
   if(argc == 4) {
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

   unsigned int amountWeights = ((inputs + 1) * hiddenNodes) + ((hiddenNodes + 1) * outputs);
   string initialScheme(amountWeights, 'A');
   unordered_set<string> schemes = generateSchemes(initialScheme);

   // Do we write the results to a file?
   bool toFile = true;
   // Do we run the program for multiple seeds?
   bool seedRun = true;

   if(seedRun) {
      const int barWidth = 70;
      float progress = 0.0;
      unsigned int amountProg = 0;
      const int startseed = 100, endseed = 1000, stepseed = 10;
      const int steps = (endseed / stepseed) - ((startseed - 1) / stepseed);
      for (unsigned int s = startseed; s <= endseed; s += stepseed) {
         amountProg = barWidth * progress;
         cout << "[" << string(amountProg, '#') << string(barWidth - amountProg, ' ') << "] "
              << int(progress * 100.0) << "%\r";
         cout.flush();
         //runSchemes({initialScheme}, inputs, hiddenNodes, outputs, epochs, s, alpha, toFile);
         runSchemes(schemes, inputs, hiddenNodes, outputs, epochs, s, alpha, toFile);
         progress += 1.0 / steps;
      }
   } else {
      runSchemes(schemes, inputs, hiddenNodes, outputs, epochs, seed, alpha, toFile);
   }
   /*run(makeNetwork(inputs, hiddenNodes, outputs, alpha), epochs, hiddenNodes, seed, alpha, toFile);*/

   /*for (unsigned int i = 0; i < n.inputs.size(); i++) {
      for (unsigned int h = 1; h < n.hiddenLayer.size(); h++) { //0 is bias
         printf("%.6f ", n.weightsFromInputs[i][h]);
      }
   }
   printf("| ");
   for (unsigned int h = 0; h < n.hiddenLayer.size(); h++) {
      printf("%.6f ", n.weightsToOutput[h][0]); //only 1 output
   }
   printf("\n");
   exit(0);*/

   /*FILE * of;
   string filename = "outputsimple.xoroutput";
   of = fopen(filename.c_str(), "a");*/
   return 0;
}
