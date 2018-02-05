#ifndef INCLUDES_H
#define INCLUDES_H

#include <algorithm>
#include <cfenv>
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
 
//#pragma STDC FENV_ACCESS ON

struct Node {
   std::vector< long double > weights;
   long double value;
   long double delta;
};

struct OutputNode {
   long double value;
   long double delta;
};

#endif

