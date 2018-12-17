with open("verbatedvariances.txt", "r") as a:
	lines = a.readlines()
	for i in range(len(lines)):
		line = lines[i]
		if ":" in line:
			ls = line.split()
			if len(ls) == 4:
				lines[i] = "{} {}    {} {}\n".format(ls[0], round(float(ls[1]), 4), ls[2], 
				round(float(ls[3]), 4))
			else:
				lines[i] = "{} {}\n".format(ls[0], round(float(ls[1]), 4))
with open("roundverbvar.txt", "w") as b:
	for l in lines:
		b.write(l)