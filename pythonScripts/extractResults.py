from glob import glob
import os

for fullfilename in glob("../results/*.schemeerrors"):
	file = open(fullfilename, "r")
	name = os.path.basename(fullfilename).replace(".schemeerrors", "")
	
	lines = file.readlines()
	first = lines[0]
	last = lines[-1]
	
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
			lowestvalue = value;
			lowestscheme = scheme
	file.close()
	with open("../results/" + name + ".extracted", "w") as n:
		n.write("first," + first + "last," + last + "highest," + highestscheme + "," + str(highestvalue) + "\nlowest" + lowestscheme + "," + str(lowestvalue) + 
"\n\n")
