if len(sys.argv) is not 3:
    raise Exception("Wrong number of arguments given!")

readname = sys.argv[1]
writename = sys.argv[2]

with open(readname, "r") as a:
    lines = a.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if ":" in line:
            ls = line.split()
            if len(ls) == 4:
                lines[
                    i
                ] = f"{ls[0]} {round(float(ls[1]), 4)}    {ls[2]} {round(float(ls[3]), 4)}\n"
            else:
                lines[i] = f"{ls[0]} {round(float(ls[1]), 4)}\n"
with open(writename, "w") as b:
    for l in lines:
        b.write(l)