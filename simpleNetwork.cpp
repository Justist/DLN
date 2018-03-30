#include <cstdlib>
#include <cstdio>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <iostream>
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
void initialiseWeights(int, int);
void XOR(vecdo&, double&);
void XORTest(Network, bool);
void writeWeights(Network, FILE *, int);
void trainTheNetwork(Network&);
void testTheNetwork(Network&);
void SIGINThandler (int);

double sigmoid(const double x) {
   return 1.0 / (1.0 + exp(-x));
}

double sigmoid_d(const double x) {
   double y = sigmoid(x);
   return y * (1.0 - y);
}

void initialiseWeights(vecvecdo& wFI,
                       vecvecdo& wTO,
                       const int inputs,
                       const int hiddens,
                       const int outputs) {
   /*const double initWeight = 0.5;
   vecvecdo weightLayer(originSize, 
                                          vecdo(targetSize, initWeight));
   */
   /*vecvecdo weightLayer(originSize,
                                          vecdo(targetSize, 
                                                           -1 + 2 * 
                                                           ((double) rand() / 
                                                            RAND_MAX)));*/
   for (int i = 0; i < inputs; i++) {
      for (int h = 1; h < hiddens; h++) {
         wFI[i][h] = -1 + 2 * ((double) rand() /RAND_MAX);
      }
   }
   for (int h = 0; h < hiddens; h++) {
      for (int o = 0; o < outputs; o++) {
         wTO[h][o] = -1 + 2 * ((double) rand() /RAND_MAX);
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

void XORTest(Network n, const bool toFile = false) {
   FILE * of;
   string filename = "i" + to_string(n.inputs.size()) + 
                     "-h" + to_string(n.hiddenLayer.size()) + 
                     "-a" + to_string(n.alpha) +
                     ".xoroutput";
   of = fopen(filename.c_str(), "w");
   double error = 0.0;
   for (int i = -1; i <= 1; i += 2) {
      for (int j = -1; j <= 1; j += 2) {
         n.inputs = {-1.0, (double) i, (double) j};
         n.expectedOutput = (i != j);
         testTheNetwork(n);
         error += abs(n.expectedOutput - sigmoid(n.calculatedOutput));
         if (toFile) {
            fprintf(of, "x: %d, y: %d, gives %.6f\n", i, j, sigmoid(n.calculatedOutput));
         } else { printf("x: %d, y: %d, gives %.6f\n", i, j, sigmoid(n.calculatedOutput)); }
      }
   }
   if (toFile) { fprintf(of, "error: %.6f\n", error); }
   else { printf("error: %.6f\n", error); }
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

void SIGINThandler (int s __attribute__((unused))) {
   /*
    * Handler to catch a SIGINT.
    * Used to escape the while-loop in the main function.
    */
   cout << "\nSIGINT caught!" << endl;
   sigintsent = true;
}

int main (const int argc, const char **argv) {

   // Raise an error when one of these float exceptions occur.
   feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);
   
   if (argc != 4) {
      printf("Usage: %s <epochs> <alpha> <seed>\n", argv[0]);
      return 1;
   }
   
   // + 1 for the bias node
   const int inputs = 2;
   const int hiddenNodes = 4;
   const int outputs = 1;
   const long int epochs = atoi(argv[1]);
   const double alpha = atof(argv[2]);
   // Seed unused for now, can be used for weight initialisation
   const int seed = atoi(argv[3]);
                       /*static_cast<unsigned int>
                       (std::chrono::high_resolution_clock::now().
                       time_since_epoch().count());*/
   srand(seed);
   
   bool toFile = false;
//   if (argc == 5) { toFile = (atoi(argv[4]) == 1); }
   
   printf("The program will run with %d hidden nodes, alpha %f, and seed %d\n", hiddenNodes, alpha, seed);
   //cout << "The program will train until Ctrl-C is pressed, after which the score will be presented." << endl;
   
   // Code to catch SIGINTs
   struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = SIGINThandler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
   sigaction(SIGINT, &sigIntHandler, nullptr);

   vecvecdo wFI(inputs + 1, vecdo(hiddenNodes + 1));
   vecvecdo wTO(hiddenNodes + 1, vecdo(outputs));
   initialiseWeights(wFI, wTO, inputs + 1, hiddenNodes + 1, outputs);
   
   Network n = {vecdo(inputs + 1, 0), //inputs
                wFI, //weightsFromInputs
                vecdo(hiddenNodes + 1, 0), //hiddenLayer
                wTO, //weightsToOutput
                0.0, //expectedOutput
                alpha, //alpha
                0.0}; //calculatedOutput

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

   FILE * of;
   string filename = "outputsimple.xoroutput";
   of = fopen(filename.c_str(), "w");

   vecdo inputVector;
   double expectedOutput;
   long int e = 0;
   while(/*!sigintsent*/e < epochs) {
      XOR(inputVector, expectedOutput);
      n.inputs = inputVector;
      n.expectedOutput = expectedOutput;
      trainTheNetwork(n);
      //writeWeights(n, of, e);
      fprintf(of, "%ld: %.6f\n", e, sigmoid(n.calculatedOutput));
      e++;
   }
   
   cout << "The program will now proceed to testing." << endl;
   
   XORTest(n, toFile);
}
