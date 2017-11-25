#ifndef MAPFUNCTIONS_H
#define MAPFUNCTIONS_H

#include "includes.hpp"

using std::map;
using std::cout;
using std::endl;

class MapFunctions {
    private:
        
    public:
        void updateValue(weightMap& mapping, 
                         const nodeToNode location, 
                         const double update, 
                         const std::function<double(double, double)> oper);
};

#endif
