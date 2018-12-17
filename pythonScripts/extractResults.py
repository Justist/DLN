from glob import glob
import os

targetdir = "../nudgingresults2hidden2/"

for fullfilename in glob(targetdir + "*.sorted"):
	file = open(fullfilename, "r")
	epoch = os.path.basename(fullfilename).replace(".schemeerrors.sorted", "")[1:]

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
	with open(targetdir + "e" + epoch + ".extracted", "w") as n:
		n.write("first," + first + "last," + last + "highest," + highestscheme + "," +
		str(highestvalue) + "\nlowest," + lowestscheme + "," + str(lowestvalue) +
"\n\n")
