#ifndef INCLUDES_H
#define INCLUDES_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

struct Node {
   std::vector< double > weights;
   double value;
   double delta;
};

struct OutputNode {
   double value;
   double delta;
};

#endif

