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

// Global variable so it can be altered in a void function
bool sigintsent = false;

struct Network {
   vector< double > inputs;
   vector< vector< double > > weightsFromInputs;
   vector< double > hiddenLayer;
   // even though the inner vector is of length 1, this enables a uniform
   // initialisation function. In the future outputsize may differ as well.
   vector< vector< double > > weightsToOutput;
   double expectedOutput;
   double alpha;
   double calculatedOutput;
};

// Function declarations so order doesn't matter.
double sigmoid(double);
double sigmoid_d(double);
vector< vector< double > > initialiseWeights(int, int);
void XOR(vector< double >&, double&);
void XORTest(Network, bool);
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

vector< vector< double > > initialiseWeights(const int originSize, 
                                             const int targetSize) {
   /*const double initWeight = 0.5;
   vector< vector< double > > weightLayer(originSize, 
                                          vector< double >(targetSize, initWeight));
   */
   vector< vector< double > > weightLayer(originSize, 
                                          vector< double >(targetSize, 
                                                           -1 + 2 * 
                                                           ((double) rand() / 
                                                            RAND_MAX)));
   return weightLayer;
}

void XOR(vector< double >& inputs, double& output) {
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
         n.expectedOutput = !(i == j);
         testTheNetwork(n);
         error += abs(n.expectedOutput - n.calculatedOutput);
         if (toFile) {
            fprintf(of, "x: %d, y: %d, gives %.6f\n", i, j, n.calculatedOutput);
         } else { printf("x: %d, y: %d, gives %.6f\n", i, j, n.calculatedOutput); }
      }
   }
   if (toFile) { fprintf(of, "error: %.6f\n", error); }
   else { printf("error: %.6f\n", error); }
}

void trainTheNetwork(Network& n) {
   unsigned int hiddenSize = n.hiddenLayer.size();
   // Forward
   testTheNetwork(n);
   
   // Backward
   double deltaOutput = sigmoid_d(n.calculatedOutput) * 
                        (n.expectedOutput - sigmoid(n.calculatedOutput));
   vector< double > delta(hiddenSize, 0.0);
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
   
   if (argc > 1 && !(argc == 4 || argc == 5)) {
      printf("Usage: %s <hidden> <alpha> <seed> [toFile]\n", argv[0]);
   }
   
   // + 1 for the bias node
   const int inputs = 2 + 1;
   const int hiddenNodes = atoi(argv[1]);
   const double alpha = (argc == 1) ? 0.5 : atof(argv[2]);
   // Seed unused for now, can be used for weight initialisation
   const unsigned int seed = (argc == 1) ? 1203 : atoi(argv[3]);
                       /*static_cast<unsigned int>
                       (std::chrono::high_resolution_clock::now().
                       time_since_epoch().count());*/
   srand(seed);
   
   bool toFile = false;
   if (argc == 5) { toFile = (atoi(argv[4]) == 1); }
   
   printf("The program will run with %d hidden nodes, alpha %f, and seed %d\n", hiddenNodes, alpha, seed);
   cout << "The program will train until Ctrl-C is pressed, after which the score will be presented." << endl;
   
   // Code to catch SIGINTs
   struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = SIGINThandler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
   sigaction(SIGINT, &sigIntHandler, nullptr);
   
   
   Network n = {vector< double >(inputs, 0), //inputs
                initialiseWeights(inputs, hiddenNodes), //weightsFromInputs
                vector< double >(hiddenNodes, 0), //hiddenLayer
                initialiseWeights(hiddenNodes, 1), //weightsToOutput
                0.0, //expectedOutput
                alpha, //alpha
                0.0}; //calculatedOutput
   
   vector< double > inputVector;
   double expectedOutput;
   while(!sigintsent) {
      XOR(inputVector, expectedOutput);
      n.inputs = inputVector;
      n.expectedOutput = expectedOutput;
      trainTheNetwork(n);
   }
   
   cout << "The program will now proceed to testing." << endl;
   
   XORTest(n, toFile);
}
