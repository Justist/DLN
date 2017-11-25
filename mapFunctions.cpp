#include "mapFunctions.hpp"

// Overloads


// Functions
void updateValue(weightMap& mapping, 
                 const nodeToNode location, 
                 const double update, 
                 const std::function<double(double, double)> oper) {
    // If the value does not yet exist in mapping, make it so
    auto mapIter = mapping.find(location);
    if(mapIter = mapping.end()) {
        mapping[location] = update;
    } else {
        (*mapIter).second = oper((*mapIter).second, update);
    }
}
