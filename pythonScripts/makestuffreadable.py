import sys

if len(sys.argv) is not 2:
    raise Exception("Wrong number of arguments given!")

filename = sys.argv[1]
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