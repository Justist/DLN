#ifndef GENERAL_CPP
#define GENERAL_CPP

#include "Includes.hpp"

namespace General {
   
   template <typename T>
   std::string to_string_prec(const T a_value, const uint8_t n = 3) {
      /*
       * Change the precision of a number T to a certain specified
       * value n. Then a string containing this number is returned.
       */
      std::ostringstream out;
      out << std::setprecision(n) << a_value;
      return out.str();
   }
   
   inline double sigmoid(const double x) {
      /*
       * Returns the y-value the default sigmoid has at coordinate x.
       */
      return 1.0 / (1.0 + exp(-x));
   }
   
   inline double sigmoid_d(const double x) {
      /*
       * Returns the y-value the derivative of the default sigmoid
       * has at coordinate x.
       */
      const double y = sigmoid(x);
      return y * (1.0 - y);
   }
   
   inline vecdo flatten(vecvecdo const& toFlatten) {
      /*
       * Flattens a vector of vectors of doubles toFlatten to a 
       * vector of doubles. This is done by concatenating all
       * vectors of doubles in toFlatten to a single vector of doubles, 
       * then returning the result.
       */
      vecdo flat;
      for (vecdo sub : toFlatten) {
         flat.insert(std::end(flat), std::begin(sub), std::end(sub));
      }
      return flat;
   }
   
   inline vecdo flatten(std::vector< vecvecdo > const& toFlatten) {
      /*
       * This function flattens a vector of vectors of vectors of doubles.
       * This is done by calling the function flatten, defined above, on
       * each vector of vectors of doubles, and then concatenating the
       * results into a single vector of doubles, which is then returned.
       */
      vecdo flat;
      for (auto& sub : toFlatten) {
         vecdo flatSub = flatten(sub);
         flat.insert(std::end(flat), std::begin(flatSub), std::end(flatSub));
      }
      return flat;
   }
   
   inline double randomWeight(unsigned int seed) {
      return -1 + 2 * (static_cast<double>(rand_r(&seed)) / RAND_MAX);
   }
   
   inline double trueRandomWeight(unsigned int seed, vecdo pastWeights) {
      /*
       * Ensure the generated weights are 'truly' different.
       * This is ensured by having all weights differ by at least 'margin'.
       */
      const double margin = 0.01; //change if needed
      double randomNumber = -1.0;
      bool stop = false;
      while (!stop) {
         stop = true;
         randomNumber = randomWeight(seed);
         for (double weight : pastWeights) {
            if (randomNumber == weight + margin || 
                randomNumber == weight - margin) {
               stop = false;
               break;
            }
         }
      }
      assert(randomNumber != -1.0);
      return randomNumber;
   }
   
   inline FILE *openFile (const std::string& filename, const char *writeMode = "w") {
      FILE *of = nullptr;
      unsigned int failcount = 0;
      while (true) {
         of = fopen(filename.c_str(), writeMode);
         if (of == nullptr) {
            failcount++;
            if (failcount > 1000000) {
               fprintf(stderr, "File could not be opened within 10 seconds!\nFilename: %s\n", filename.c_str());
               exit(1);
            }
            usleep(10); //Wait for the previous operation on the file to end
         } else {
            break;
         }
      }
      return of;
   }
}

#endif
