#include "network.hpp"

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
    */
   return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

void SIGINThandler(int s) {
   /*
    * Handler to catch a SIGINT.
    * Used to escape the while-loop in the main function.
    */
   cout << "\nSIGINT caught!" << endl;
   cout << "Do you want to export the network? y/N" << endl;
   std::string answer;
   std::cin >> answer;
   std::transform(answer.begin(), answer.end(), answer.begin(), ::tolower);
   if(answer == "y") { wantToExport = true; }
   sigintsent = true;
}

int main(int argc, char** argv) {
   int outputlength = 2;
   Network network(outputlength);
   vector<float> m1(2), m2(outputlength);
   float a, b;
   
   // Code to catch SIGINTs
   struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = SIGINThandler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
   sigaction(SIGINT, &sigIntHandler, NULL);
   
   while(sigintsent == false) {
      srand (static_cast <unsigned> (time(0)));
      a = RandomNumber(-1000, 1000);
      b = RandomNumber(-1000, 1000);
      m1 = {a, b};
      if(a + b > 0) m2 = {1.0, 0.0};
      else m2 = {0.0, 1.0};
      network.run(m1, m2);
   }
   if(wantToExport) {
      auto t = std::time(nullptr);
      auto tm = *std::localtime(&t);
      std::ostringstream oss;
      oss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
      network.exportNetwork("network" + oss.str());
   }
   return 0;
}
