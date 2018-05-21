import os
import re

directoryname = "../schemetest"
directory = os.fsencode(directoryname)	
nodes = ""
for file in os.listdir(directory):
	filename = os.fsdecode(file)
	scheme = re.search("w(.*)e[0-9]+", filename).group(1)
	nodes = re.search("a0.5(.*).xoroutput", filename).group(1)
	with open("../results/" + nodes + ".schemeerrors", "a") as sea:
	   with open(directoryname + "/" + filename, "r") as fo:
		   sum = 0.0
		   for line in fo:
			   ls = line.split()
			   sum += float(ls[3])
		   sea.write(scheme + "," + str(sum) + "\n")
	
#can be shorter possibly, but it works	   
sea = open("../results/" + nodes + ".schemeerrors", "r")
lines = sorted(set(sea.readlines()))
sea.close()
with open("../results/" + nodes + ".schemeerrors", "w") as sea:
   for line in lines:
      sea.write(line)
   
