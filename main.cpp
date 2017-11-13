#include "network.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>

// Global variables so they can be altered in a void function.
bool sigintsent = false;
bool wantToExport = false;

float RandomNumber(float Min, float Max) {
   /*
    * As copied from https://stackoverflow.com/a/4310296/1762311
    * Creates a random float between the floats Min and Max, where
    * both Min and Max can be either positive or negative.
    * Inputs:
    *    Min: Minimal float to be generated.
    *    Max: Maximal float to be generated.
    */
   return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

void SIGINThandler(int s) {
   /*
    * Handler to catch a SIGINT.
    * Used to escape the while-loop in the main function.
    */
   cout << "\n\n\nSIGINT caught!" << endl;
   cout << "Do you want to export the network? y/N" << endl;
   std::string answer;
   std::cin >> answer;
   std::transform(answer.begin(), answer.end(), answer.begin(), ::tolower);
   if(answer == "y") { wantToExport = true; }
   sigintsent = true;
}

void AplusB(vector<double>& m1, vector<double>& m2) {
   srand(static_cast<long unsigned int>
            (std::chrono::high_resolution_clock::now().
                          time_since_epoch().count()));
   float a, b;
   a = RandomNumber(-1000, 1000);
   b = RandomNumber(-1000, 1000);
   
   m1 = {a, b};
   if((a + b) > 0) m2 = {1.0}; //{1.0, 0.0};
   else m2 = {0.0}; //{0.0, 1.0};
}

void XOR(vector<double>& m1, vector<double>& m2) {
   srand(static_cast<long unsigned int>
            (std::chrono::high_resolution_clock::now().
                          time_since_epoch().count()));
   float a, b;
   a = static_cast<int>(rand()%2);
   b = static_cast<int>(rand()%2);
   
   m1 = {a, b};
   if((a + b) == 1) m2 = {1.0}; //{1.0, 0.0};
   else m2 = {0.0}; //{0.0, 1.0};
}

void exportNetwork(Network network, const unsigned int outputlength) {
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

int main(int argc, char** argv) {
   unsigned int outputlength = 1;
   Network network(outputlength);
   vector<double> m1(2), m2(outputlength);
   
   
   cout << "Do you want to import a network? y/N" << endl;
   std::string answer;
   std::cin >> answer;
   std::transform(answer.begin(), answer.end(), answer.begin(), ::tolower);
   if(answer == "y") { 
      cout << "Welk bestand wilt u importeren?" << endl;
      std::string bestand;
      std::cin >> bestand;
      network.importNetwork(bestand);
   }
   
   cout << endl << endl;
   
   // Code to catch SIGINTs
   struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = SIGINThandler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
   sigaction(SIGINT, &sigIntHandler, NULL);
   
   using namespace std::chrono;
   
   
   while(sigintsent == false) {
      AplusB(m1, m2);
      network.run(m1, m2);
   }
   if(wantToExport) {
      exportNetwork(network, outputlength);
   }
   return 0;
}
