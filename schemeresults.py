import os
import re

directoryname = "schemetest"
directory = os.fsencode(directoryname)
with open("schemeerrors.average", "w") as sea:	
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		scheme = re.search("w(.*)e[0-9]+", filename).group(1)
		with open(directoryname + "/" + filename, "r") as fo:
			sum = 0.0
			for line in fo:
				ls = line.split()
				sum += float(ls[3])
			sea.write(scheme + "," + str(sum) + "\n")