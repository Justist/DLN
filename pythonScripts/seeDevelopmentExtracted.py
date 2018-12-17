from glob import glob
import os

targetdir = "../nudgingresults2hidden2/"

for fullfilename in glob(targetdir + "*.extracted"):
	with open(fullfilename, "r") as file:
		name = os.path.basename(fullfilename).replace(".extracted", "")
		epochLocation = name.find("e")
		epoch = "last"
		if epochLocation > 0:
			epoch = str(name[epochLocation + 1:])
			name = name[:epochLocation]

		lines = file.readlines()

		i = 0
		for x in ["first", "last", "highest", "lowest"]:
			with open(targetdir + name + "-" + x, "a") as f:
				ls = lines[i][:-1].split(",")
				f.write(epoch + "," + ls[2] + "," + ls[1][:15] + "\n")
				#f.write("{},{},{}".format(epoch,ls[2],ls[1]))
			i += 1