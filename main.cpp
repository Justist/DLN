#include "includes.hpp"
#include "network.hpp"

// Global variables so they can be altered in a void function.
bool sigintsent = false;
bool wantToExport = false;

float RandomNumber (float Min, float Max) {
   /*
    * As copied from https://stackoverflow.com/a/4310296/1762311
    * Creates a random float between the floats Min and Max, where
    * both Min and Max can be either positive or negative.
    * Inputs:
    *    Min: Minimal float to be generated.
    *    Max: Maximal float to be generated.
    */
   return float(rand()) / float(RAND_MAX) * (Max - Min) + Min;
}

void SIGINThandler (int s __attribute__((unused))) {
   /*
    * Handler to catch a SIGINT.
    * Used to escape the while-loop in the main function.
    */
   cout << "\33[4B\rSIGINT caught!" << endl;
   cout << "\nDo you want to export the network? y/N" << endl;
   std::string answer;
   std::cin >> answer;
   std::transform(answer.begin(), answer.end(), answer.begin(), ::tolower);
   if (answer == "y") { wantToExport = true; }
   sigintsent = true;
}

void AplusB (vector< long double >& m1, vector< long double >& m2) {
   /*
    * Generates input for the network.
    * The 'problem' is whether the sum of two generated numbers 
    * a and b is bigger than 0. If so, the network should return 1,
    * else 0.
    * Inputs:
    *    m1, the vector which will contain a and b.
    *    m2, the vector which will contain 1 if a + b > 0, else 0. 
    */
   srand(static_cast<unsigned int>
         (std::chrono::high_resolution_clock::now().
             time_since_epoch().count()));
   float a, b;
   a = RandomNumber(-1000, 1000);
   b = RandomNumber(-1000, 1000);
   
   m1 = {a/1000, b/1000}; // Just for testing
   if ((a + b) > 0) {
      m2 = {1.0}; //{1.0, 0.0};
   } else { m2 = {0.0}; } //{0.0, 1.0};
}

void XOR (vector< long double >& m1, vector< long double >& m2) {
   /*
    * Generates input for the network.
    * The 'problem' is if a xor b is equal to 1. 
    * If so, the network should return 1, else 0.
    * Inputs:
    *    m1, the vector which will contain a and b.
    *    m2, the vector which will contain 1 if a xor b == 1, else 0. 
    */
   srand(static_cast<unsigned int>
         (std::chrono::high_resolution_clock::now().
             time_since_epoch().count()));
   float a, b;
   a = rand() % 2;
   b = rand() % 2;
   
   m1 = {a, b};
   if ((a + b) == 1) {
      m2 = {1.0}; //{1.0, 0.0};
   } else { m2 = {0.0}; } //{0.0, 1.0};
}

void exportNetwork (Network network, const unsigned int outputlength) {
   /*
    * Creates a filename based on the current datetime and the length
    * of the output layer, then exports the network to a file with that name.
    * Input:
    *    outputlength, the length of the output layer
    */
   auto t = std::time(nullptr);
   auto tm = *std::localtime(&t);
   std::ostringstream oss;
   oss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
   network.exportNetwork("network" + oss.str() +
                         "-ol_" + std::to_string(outputlength));
}

int main (int argc __attribute__((unused)),
          char **argv __attribute__((unused))) {
//   cout << "Do you want to import a network? y/N" << endl;
//   std::string answer;
//   std::cin >> answer;
//   std::transform(answer.begin(), answer.end(), answer.begin(), ::tolower);
//   if(answer == "y") { 
//      cout << "Whcih file do you want to read from?" << endl;
//      std::string bestand;
//      std::cin >> bestand;
//      network.importNetwork(bestand);
//   }
   
   cout << endl << endl;
   
   // Code to catch SIGINTs
   struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = SIGINThandler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
   sigaction(SIGINT, &sigIntHandler, nullptr);
  
   unsigned int outputlength = 1;
   Network network(outputlength);
   vector< long double > m1(2), m2(outputlength);
   
   while (!sigintsent) {
      AplusB(m1, m2);
      //XOR(m1, m2);
      network.run(m1, m2);
   }
//   for (unsigned int i = 1; i < 6; i++) { //hidden layers
//      for (unsigned int j = 1; j <= 10; j++) { //layer length
//         for (float k = 0.0; k <= 1.0; k += 0.1) { //learning rate
//            Network network(outputlength, i, j, 420, k);
//            for (unsigned int l = 0; l < 100000; l++) { //iterations
//               AplusB(m1, m2);
//               //XOR(m1, m2);
//               network.run(m1, m2);
//            }
//            printf("hl: %d, ll: %d, lr: %f, acc: %f\n", i, j, k, network.accuracy);
//         }
//      }
//   }
   if (wantToExport) {
      exportNetwork(network, outputlength);
   }
   return 0;
}
