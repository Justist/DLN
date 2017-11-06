#include "network.hpp"

#include <cstdlib>
#include <ctime>

float RandomNumber(float Min, float Max) {
   /*
    * As copied from https://stackoverflow.com/a/4310296/1762311
    */
   return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

int main(int argc, char** argv) {
   int outputlength = 2;
   Network network(outputlength);
   vector<float> m1(2), m2(outputlength);
   float a, b;
   while(true) {
      srand (static_cast <unsigned> (time(0)));
      a = RandomNumber(-1000, 1000);
      b = RandomNumber(-1000, 1000);
      m1 = {a, b};
      if(a + b > 0) m2 = {1.0, 0.0};
      else m2 = {0.0, 1.0};
      network.run(m1, m2);
   }
   return 0;
}
