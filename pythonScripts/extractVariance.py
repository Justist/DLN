from glob import glob
from collections import defaultdict
import os
import re
import sys

if len(sys.argv) is not 2:
    raise Exception("Wrong number of arguments given!")

targetdir = sys.argv[1]

variancevalues = defaultdict(lambda: 0)
variancecounts = defaultdict(lambda: 0)
for filename in glob(f"{targetdir}*.sorted"):
    with open(filename, "r") as file:
        epoch = os.path.basename(filename).replace(".schemeerrors.sorted", "")
        for line in file:
            ls = line.split(',')
            scheme = re.sub(r"(?![A-Z])(.*)", "", ls[0])
            value = float(ls[1])
            variance = ord(scheme[-1]) - ord('A')
            variancevalues[variance] += value
            variancecounts[variance] += 1
    with open(targetdir + epoch + "variances.output", "a") as v:
        for k in variancecounts:
            v.write(f"{str(k)}: {str(variancevalues[k] / variancecounts[k])}" + "\n")
    variancevalues = defaultdict(lambda: 0)
    variancecounts = defaultdict(lambda: 0)