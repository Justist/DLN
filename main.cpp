#include "network.hpp"

int main(int argc, char** argv) {
   Network network(2);
   vector<float> m1(4, 1), m2(4, 2);
   for(auto one : network.createOutput(m1)) std::cout << one << std::endl;
   std::cout << "--------" << std::endl;
   for(auto two : network.createOutput(m2)) std::cout << two << std::endl;
   return 0;
}
