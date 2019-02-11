import os
import re
import sys

if len(sys.argv) is not 3:
    raise Exception("Wrong number of arguments given!")

directoryname = sys.argv[1]
resultsdir = sys.argv[2]

directory = os.fsencode(directoryname)
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)
nodes = ""
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    scheme = re.search("w(.*)e[0-9]+", filename).group(1)
    nodes = re.search("a0.5(.*).xoroutput", filename).group(1)
    with open(resultsdir + nodes + ".schemeerrors", "a") as sea:
       with open(directoryname + "/" + filename, "r") as fo:
           sum = 0.0
           for line in fo:
               ls = line.split()
               sum += float(ls[3])
           sea.write(scheme + "," + str(sum) + "\n")

#can be shorter possibly, but it works
sea = open(resultsdir + nodes + ".schemeerrors", "r")
lines = sorted(set(sea.readlines()))
sea.close()
with open(resultsdir + nodes + ".schemeerrors", "w") as sea:
   for line in lines:
      sea.write(line)

