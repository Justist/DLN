#include "network.hpp"

int main(int argc, char** argv) {
   Network network(2);
   vector<float> m1(4), m2(4);
   for(auto e1 = begin(m1), e2 = begin(m2), e3 = end(m1); e1 != e3; e1++, e2++) {
   	*e1 = 1;
   	*e2 = 1;
   }
   return 0;
}
