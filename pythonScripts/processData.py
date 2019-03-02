"""
Processes the data generated into 'summaries' (TODO: find a better description)
Basically all the earlier scripts are merged into one, so the execution of those
can be streamlined.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import defaultdict
from glob import glob
import os
import re
from time import sleep
import sys
import threading as thr

### Extraction functions

def initialFunction(inputdir, outputdir):
   """
   THIS FUNCTION SHOULD BE RUN FIRST!
   For each file in the experiment, this function takes
   the sum of the values in the file, and prints those
   to a seperate file. The created files form the basis
   for all other functions in this file.
   """
   inputdirectory = os.fsencode(inputdir)
   if not os.path.exists(outputdir):
       os.makedirs(outputdir)
   resultsdirectory = os.fsencode(outputdir)

   def threadfunctionfirst(fileInput):
       filename = os.fsdecode(fileInput)
       scheme = re.search(r"w(.*)e[0-9]+", filename).group(1)
       try:
           epoch = re.search(r"o1e(.*).xoroutput", filename).group(1)
       except AttributeError:
           return #those are the same as the e20000 anyway
       with open(outputdir + "e" + epoch + ".schemeerrors", "a") as sea:
           with open(inputdir + "/" + filename, "r") as fo:
               errorSum = 0.0
               for line in fo:
                   ls = line.split()
                   errorSum += float(ls[3])
               sea.write(scheme + "," + str(errorSum) + "\n")

   for filePointer in os.listdir(inputdirectory):
       t = thr.Thread(target = threadfunctionfirst, args = (filePointer, ))
       while thr.active_count() > 500:
           sleep(0.005)
       t.start()

   while thr.active_count() > 1:
       sleep(0.05)

   def threadfunctionsecond(result):
       filename = outputdir + "/" + os.fsdecode(result)
       with open(filename, "r") as a:
           lines = sorted(set(a.readlines()))
           with open(filename + ".sorted", "w") as b:
               for line in lines:
                   b.write(line)

   #can be shorter possibly, but it works
   for result in os.listdir(resultsdirectory):
       u = thr.Thread(target = threadfunctionsecond, args = (result, ))
       while thr.active_count() > 500:
           sleep(0.005)
       u.start()

def extractResults(outputdir):
   """
   For each epoch among the performed experiments, find the highest error value,
   the lowest, the first, and the last. The first two of these are to give a sense
   of the extremes in the dataset, and the latter two to see if those correspond to
   either the scheme with the least or with the most variation.
   """
   for fullfilename in glob(outputdir + "*.sorted"):
      filePointer = open(fullfilename, "r")
      epoch = os.path.basename(fullfilename).replace(".schemeerrors.sorted", "")[1:]

      lines = filePointer.readlines()

      highestvalue = 0.0
      highestscheme = ""
      lowestvalue = 2000.0 #put a higher value if needed
      lowestscheme = ""
      for line in lines:
         ls = line.split(',')
         scheme = ls[0]
         value = float(ls[1])
         if value > highestvalue:
            highestvalue = value
            highestscheme = scheme
         if value < lowestvalue:
            lowestvalue = value
            lowestscheme = scheme
      filePointer.close()
      with open(outputdir + "e" + epoch + ".extracted", "w") as n:
         n.write("first,"    + lines[0]      +
                 "last,"     + lines[-1]     +
                 "highest,"  + highestscheme + "," + str(highestvalue) +
                 "\nlowest," + lowestscheme  + "," + str(lowestvalue)  +
                 "\n\n")

def extractVariation(outputdir):
   """
   For each epoch among the performed experiments, calculate the average error
   for each amount of variation. These are kept in dictionaries, as in this way
   the sum of errors and the amount of values summed can be kept in storage with
   the same index, and dictionaries can be easily inserted to and reset when
   necessary.
   """
   variancevalues = defaultdict(lambda: 0)
   variancecounts = defaultdict(lambda: 0)
   for filename in glob(outputdir + "*.sorted"):
      filePointer = open(filename, "r")
      epoch = os.path.basename(filename).replace(".schemeerrors.sorted", "")
      for line in filePointer:
         ls = line.split(',')
         scheme = re.sub(r"(?![A-Z])(.*)", "", ls[0])
         value = float(ls[1])
         variance = ord(scheme[-1]) - ord('A')
         variancevalues[variance] += value
         variancecounts[variance] += 1
      filePointer.close()
      with open(outputdir + epoch + "variation.output", "a") as v:
         for k in variancecounts:
            v.write(str(k) + ": " + str(variancevalues[k]/variancecounts[k]) + "\n")
      variancevalues = defaultdict(lambda: 0)
      variancecounts = defaultdict(lambda: 0)

### Paper functions

def roundVariations(outputdir, regString = "*.variation.output"):
   """
   Just round the error values of the variations.
   This is a function to make the values more aesthetically
   pleasing in the paper.
   """
   for filename in glob(outputdir + regString):
      with open(filename, "r") as a:
         lines = a.readlines()
         for i in range(len(lines)):
            line = lines[i]
            if ":" in line:
               ls = line.split()
               if len(ls) == 4:
                  lines[i] = "{} {}    {} {}\n".format(ls[0],
                     round(float(ls[1]), 4), ls[2],
                     round(float(ls[3]), 4))
               else:
                  lines[i] = "{} {}\n".format(ls[0], round(float(ls[1]), 4))
      with open(filename + ".rounded", "w") as b:
         for l in lines:
            b.write(l)

def readableVariations(outputdir, regString = "*.rounded"):
   """
   Re-orders the results in a given file to be more
   aesthetically pleasing. Just adds a lot of whitespace.
   Made for the variation results.
   """
   for filename in glob(outputdir + regString):
      with open(filename, "r") as a:
          alllines = a.readlines()
          lenfile = len(alllines)
          halflenfile = (lenfile // 2) + 1
          for i in range(lenfile):
              line = alllines[i]
              if i >= halflenfile:
                  alllines[i - halflenfile] = alllines[i - halflenfile][:-1] + " " + line
          alllines = alllines[:halflenfile]
      with open(filename + ".readable", "w") as b:
          for line in alllines:
              b.write(line)
         
def developmentExtracted(outputdir):
   """
   Shows the development in the extreme values which have been extracted.
   """
   for filename in glob(outputdir + "*.extracted"):
      with open(filename, "r") as filePointer:
         name = os.path.basename(filename).replace(".extracted", "")
         epochLocation = name.find("e")
         epoch = "last"
         if epochLocation > 0:
            epoch = str(name[epochLocation + 1:])
            name = name[:epochLocation]

         lines = filePointer.readlines()

         i = 0
         for x in ["first", "last", "highest", "lowest"]:
            with open(outputdir + name + "-" + x, "a") as f:
               ls = lines[i][:-1].split(",")
               f.write(epoch + "," + ls[2] + "," + ls[1][:15] + "\n")
            i += 1
            
def makeVerbatim(outputdir, regString, captionString):
   """
   Envelops the results in a given file in verbatim.
   """
   toVerbate = glob(outputdir + regString)
   for filename in toVerbate:
      with open(filename, "r") as a:
         alllines = a.readlines()
         epoch = re.search(r"e(\d+)", filename).group(0)
         with open(filename + ".verbatim", "w") as b:
            b.write("""
            \\begin{figure}[!ht]
            \\begin{verbatim}
            """)
            for line in alllines:
               b.write(line)
            b.write("""
            \\end{verbatim}
            \\caption{"""+ "The {} of epoch ${}$.".format(captionString, epoch[1:]) +"""}
            \\end{figure}

            """)
            
def makeGraphExtremes(outputdir, epochGroups = 20, maxEpoch = 20000):
   first = ()
   last = ()
   high = ()
   low = ()
   
   for filename in glob(outputdir + "*.extracted"):
      with open(filename, "r") as filePointer:
         for line in filePointer:
            ls = line.split(",")
            if ls[0] == "first":
               first += (float(ls[2]),)
            elif ls[0] == "last":
               last += (float(ls[2]),)
            elif ls[0] == "highest":
               high += (float(ls[2]),)
            elif ls[0] == "lowest":
               low += (float(ls[2]),)
   
   fig, ax = plt.subplots(figsize=(20, 15))
   index = np.arange(epochGroups)
   barWidth = 0.15
   firstbars = ax.bar(index, first, barWidth, color=colors.to_rgb('xkcd:grapefruit'), label="First")
   firstbars = ax.bar(index + barWidth, last, barWidth, color='xkcd:yellow tan', label="Last")
   firstbars = ax.bar(index + 2 * barWidth, high, barWidth, color='xkcd:dark yellow green', label="Highest")
   firstbars = ax.bar(index + 3 * barWidth, low, barWidth, color='xkcd:dark blue grey', label="Lowest")
   
   ax.set_xlabel("Epochs")
   ax.set_ylabel("Sum of errors")
   ax.set_title("Extremes per epoch")
   ax.set_xticks(index + 2 * barWidth)
   ax.set_xticklabels([str(x / 1000) + "k" 
                       for x in range(maxEpoch // epochGroups, 
                                      maxEpoch, 
                                      maxEpoch // epochGroups)])
   ax.get_yaxis().get_major_formatter().set_useOffset(False)
   ax.set_ylim([min(low) - 5, max(high) + 5])
   ax.legend()
   fig.tight_layout()
   
   plt.show()
   plt.savefig(outputdir + "extremes.png")
   

### Main

def usage():
   print("""
   Usage: {} <inputdir> <outputdir>
   Where <inputdir> is the location the DLN has put out its results to.
   And where <outputdir> is the location this script will put out its results to.
   """.format(sys.argv[0]))

def main():
   if len(sys.argv) is not 3:
      usage()
      raise Exception("Wrong number of arguments given!")

   inputdir = sys.argv[1]
   outputdir = sys.argv[2]

   if inputdir[-1] is not "/":
      inputdir += "/"
   if outputdir[-1] is not "/":
      outputdir += "/"

   if not os.path.exists(inputdir):
      raise Exception("Inputdir does not exist! Please check your path!")

   initialFunction(inputdir, outputdir)
   extractResults(outputdir)
   extractVariation(outputdir)
   
   roundVariations(outputdir)
   readableVariations(outputdir)
   developmentExtracted(outputdir)
   makeVerbatim(outputdir, "*.readable", "average errors for each amount of variation")
   makeVerbatim(outputdir, "*.extracted", "extreme values of the summed errors")
   makeGraphExtremes(outputdir)

   
if __name__ == "__main__":
   main()
