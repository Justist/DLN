from glob import glob
from collections import defaultdict
import re

variancevalues = defaultdict(lambda: 0)
variancecounts = defaultdict(lambda: 0)
for file in glob("../results/*.schemeerrors"):
	with open(file, "r") as f:
		for line in f:
			ls = line.split(',')
			scheme = ls[0]
			value = float(ls[1])
			variance = ord(ls[0][-1]) - ord('A')
			variancevalues[variance] += value
			variancecounts[variance] += 1
#print({k: variancevalues[k]/variancecounts[k] for k in variancecounts})
with open("../results/variances.output", "w") as v:
	for k in variancecounts:
		v.write(str(k) + ": " + str(variancevalues[k]/variancecounts[k]) + "\n")
	
