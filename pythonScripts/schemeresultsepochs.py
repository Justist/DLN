import os
import re
import sys
import threading as thr
from time import sleep

if len(sys.argv) is not 3:
    raise Exception("Wrong number of arguments given!")

directoryname = sys.argv[1]
resultsdirname = sys.argv[2]

directory = os.fsencode(directoryname)
if not os.path.exists(resultsdirname):
    os.makedirs(resultsdirname)
resultsdirectory = os.fsencode(resultsdirname)
epoch = ""

def threadfunctionfirst(file):
    filename = os.fsdecode(file)
    scheme = re.search(r"w(.*)e[0-9]+", filename).group(1)
    try:
        epoch = re.search(r"o1e(.*).(.*)output", filename).group(1)
    except AttributeError:
        return #those are the same as the e20000 anyway
    with open(resultsdirname + "e" + epoch + ".schemeerrors", "a") as sea:
        with open(directoryname + "/" + filename, "r") as fo:
            sum = 0.0
            for line in fo:
                ls = line.split()
                sum += float(ls[3])
            sea.write(scheme + "," + str(sum) + "\n")

for file in os.listdir(directory):
    t = thr.Thread(target = threadfunctionfirst, args = (file, ))
    while thr.active_count() > 500:
        sleep(0.005)
    t.start()

while thr.active_count() > 1:
    sleep(0.05)

def threadfunctionsecond(result):
    filename = resultsdirname + "/" + os.fsdecode(result)
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