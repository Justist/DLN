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
#include <map>
#include <sstream>
#include <unordered_map>
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

//// Hash combine function so {a,b} isn't equal to {b,a}
//// Taken from https://stackoverflow.com/a/38140932/1762311
//inline void hash_combine(std::size_t& seed) { (void) seed; }

//template <typename T, typename... Rest>
//inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
//    std::hash<T> hasher;
//    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
//    hash_combine(seed, rest...);
//}

//// To enable the comparison of pairs of ints with each other.
//// As taken from https://stackoverflow.com/a/20602159/1762311
//struct pairhash {
//    public:
//        template <typename T, typename U>
//        std::size_t operator()(const std::pair<T, U> &x) const {
//            std::size_t h=0;
//            hash_combine(h, x.first, x.second);
//            return h;
//        }
//};

//// Pair of nodes, each one in a different layer
//typedef std::pair<int, int> nodeToNode;
//// Map of nodepairs to doubles, which is the weight between those nodes
//typedef std::unordered_map<nodeToNode, double, pairhash> weightMap;
//// Map of a layer to layer + 1 of weights between nodes
//typedef std::unordered_map<int, weightMap> weightLayer;

