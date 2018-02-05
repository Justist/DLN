with open("XORoutput.txt", "r") as o:
	with open("XORsufficientaccuracies.txt", "w") as s:
		for line in o:
			ls = line.split()
			acc = float(ls[-1])
			if acc < 0.01:
				s.write(str(ls[:-2]) + " 0\n")
			elif acc > 0.99:
				s.write(str(ls[:-2]) + " 1\n")
